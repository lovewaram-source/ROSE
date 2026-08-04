import time

import torch
import torch.nn as nn
import transformers

from .sparsegpt import SparseGPT


class SparseGPTGlobalMaskReorder(SparseGPT):
    """Global-mask guided column reordering with blockwise SparseGPT compensation.

    A layer-wide Hessian score first predicts one exact pruning mask.  The
    predicted loss of the masked entries ranks columns from high to low loss.
    Columns are then repacked into fixed-width blocks.  Each repacked block
    inherits its pruning count from the global prediction, while the actual
    within-block mask is recomputed during SparseGPT compensation.
    """

    def __init__(self, layer, slice_size=128, verbose=False):
        super().__init__(layer)
        self.slice_size = slice_size
        self.verbose = verbose
        self.last_block_budgets = None
        self.last_block_sparsities = None
        self.last_column_permutation = None

    @staticmethod
    def _bottomk_mask(score, k):
        k = max(0, min(int(k), score.numel()))
        mask = torch.zeros_like(score, dtype=torch.bool)
        if k == 0:
            return mask
        if k == score.numel():
            mask.fill_(True)
            return mask
        indices = torch.topk(score.reshape(-1), k, largest=False, sorted=False).indices
        mask.reshape(-1)[indices] = True
        return mask

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
                "SparseGPTGlobalMaskReorder supports only unstructured sparsity"
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
        Hinv_original = torch.linalg.cholesky(H_inverse, upper=True)

        # Predict one exact global mask before any block partition exists.
        predicted_score = W.pow(2) / torch.diag(Hinv_original).reshape(1, -1).pow(2)
        target_k = int(round(predicted_score.numel() * sparsity))
        predicted_mask = self._bottomk_mask(predicted_score, target_k)

        # A column is important to compensate early when its predicted pruned
        # weights have a large aggregate Hessian loss.
        column_predicted_loss = (predicted_score * predicted_mask).sum(dim=0)
        permutation = torch.argsort(column_predicted_loss, descending=True)
        self.last_column_permutation = permutation.detach().cpu()

        W = W[:, permutation]
        predicted_mask = predicted_mask[:, permutation]
        H_inverse = H_inverse[:, permutation][permutation, :]
        Hinv = torch.linalg.cholesky(H_inverse, upper=True)

        block_budgets = []
        block_widths = []
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            block_budgets.append(int(predicted_mask[:, i1:i2].sum().item()))
            block_widths.append(i2 - i1)

        if sum(block_budgets) != target_k:
            raise RuntimeError("Global predicted mask does not match target pruning budget")

        self.last_block_budgets = block_budgets
        self.last_block_sparsities = [
            budget / (self.rows * width)
            for budget, width in zip(block_budgets, block_widths)
        ]

        for block_index, (i1, width, block_k) in enumerate(
            zip(range(0, self.columns, blocksize), block_widths, block_budgets)
        ):
            i2 = i1 + width
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # Recompute the local mask after previous blocks have compensated
            # the remaining columns, while preserving this block's predicted
            # global pruning count.
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

            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        inverse_permutation = torch.argsort(permutation)
        W = W[:, inverse_permutation]
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if W.is_cuda:
            torch.cuda.synchronize()
        if self.verbose:
            ratios = self.last_block_sparsities
            print(
                "SparseGPTGlobalMaskReorder "
                f"target={target_k / self.layer.weight.numel():.6f} "
                f"actual_range=[{min(ratios):.6f}, {max(ratios):.6f}] "
                f"blocks={len(ratios)} blocksize={blocksize} "
                f"time={time.perf_counter() - tick:.2f}s"
            )
