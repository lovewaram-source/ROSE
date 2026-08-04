import time

import torch
import torch.nn as nn
import transformers

from .sparsegpt_globalmask_reorder import SparseGPTGlobalMaskReorder


class SparseGPTGlobalMaskDynamic(SparseGPTGlobalMaskReorder):
    """Recompute global-mask column ordering after every compensated block."""

    def fasterprune(
        self,
        sparsity,
        prune_n=0,
        prune_m=0,
        blocksize=None,
        percdamp=0.01,
    ):
        if prune_n or prune_m:
            raise ValueError(
                "SparseGPTGlobalMaskDynamic supports only unstructured sparsity"
            )
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must satisfy 0 <= sparsity < 1")

        blocksize = self.slice_size if blocksize is None else blocksize
        if blocksize <= 0:
            raise ValueError("slice_size must be a positive integer")

        tick = time.perf_counter()
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        diagonal = torch.arange(self.columns, device=self.dev)
        H[diagonal, diagonal] += percdamp * torch.mean(torch.diag(H))
        H_inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))

        target_k = int(round(W.numel() * sparsity))
        committed_k = 0
        remaining_columns = torch.arange(self.columns, device=self.dev)
        W_state = W
        W_result = torch.zeros_like(W)
        round_budgets = []
        round_sparsities = []
        round_first_columns = []

        while remaining_columns.numel():
            Hinv = torch.linalg.cholesky(H_inverse, upper=True)
            remaining_target = target_k - committed_k
            if not 0 <= remaining_target <= W_state.numel():
                raise RuntimeError("Invalid remaining global pruning budget")

            # Rebuild an exact mask for every unprocessed column after all
            # compensation from the previously committed blocks.
            predicted_score = (
                W_state.pow(2)
                / torch.diag(Hinv).reshape(1, -1).pow(2)
            )
            predicted_mask = self._bottomk_mask(predicted_score, remaining_target)
            column_predicted_loss = (predicted_score * predicted_mask).sum(dim=0)
            permutation = torch.argsort(column_predicted_loss, descending=True)

            W_state = W_state[:, permutation]
            predicted_mask = predicted_mask[:, permutation]
            remaining_columns = remaining_columns[permutation]
            H_inverse = H_inverse[:, permutation][permutation, :]
            Hinv = torch.linalg.cholesky(H_inverse, upper=True)

            width = min(blocksize, remaining_columns.numel())
            block_k = int(predicted_mask[:, :width].sum().item())
            block_ratio = block_k / (self.rows * width)
            round_budgets.append(block_k)
            round_sparsities.append(block_ratio)
            round_first_columns.append(int(remaining_columns[0].item()))

            W1 = W_state[:, :width].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[:width, :width]
            local_score = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2)
            mask1 = self._bottomk_mask(local_score, block_k)

            for column in range(width):
                weight = W1[:, column]
                divisor = Hinv1[column, column]
                quantized = weight.clone()
                quantized[mask1[:, column]] = 0
                Q1[:, column] = quantized

                error = (weight - quantized) / divisor
                W1[:, column:] -= error.unsqueeze(1).matmul(
                    Hinv1[column, column:].unsqueeze(0)
                )
                Err1[:, column] = error

            selected_columns = remaining_columns[:width]
            W_result[:, selected_columns] = Q1
            committed_k += block_k

            if width < remaining_columns.numel():
                # Apply this block's SparseGPT error to the remaining columns
                # before their global prediction is rebuilt in the next round.
                W_state = W_state[:, width:]
                W_state -= Err1.matmul(Hinv[:width, width:])
                remaining_columns = remaining_columns[width:]
                trailing_factor = Hinv[width:, width:]
                H_inverse = trailing_factor.t().matmul(trailing_factor)
                H_inverse = (H_inverse + H_inverse.t()) / 2
            else:
                remaining_columns = remaining_columns[:0]

        if committed_k != target_k:
            raise RuntimeError("Dynamic global-mask pruning did not meet target budget")

        self.last_block_budgets = round_budgets
        self.last_block_sparsities = round_sparsities
        self.last_column_permutation = round_first_columns
        self.last_rounds = len(round_budgets)

        if isinstance(self.layer, transformers.Conv1D):
            W_result = W_result.t()
        self.layer.weight.data = W_result.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if W_result.is_cuda:
            torch.cuda.synchronize()
        if self.verbose:
            print(
                "SparseGPTGlobalMaskDynamic "
                f"target={target_k / self.layer.weight.numel():.6f} "
                f"actual_range=[{min(round_sparsities):.6f}, {max(round_sparsities):.6f}] "
                f"rounds={self.last_rounds} blocksize={blocksize} "
                f"time={time.perf_counter() - tick:.2f}s"
            )
