import time

import torch
import torch.nn as nn
import transformers

from .sparsegpt_slice import SparseGPTSlice


class SparseGPTSliceReorder(SparseGPTSlice):
    """Hessian slice budgets with Wanda-guided compensation ordering."""

    def __init__(
        self,
        layer,
        reorder_mode,
        slice_size=128,
        min_sparsity=None,
        max_sparsity=None,
        allocation_step=0.01,
        reorder_threshold=0.5,
        verbose=False,
    ):
        super().__init__(
            layer,
            slice_size=slice_size,
            min_sparsity=min_sparsity,
            max_sparsity=max_sparsity,
            allocation_step=allocation_step,
            verbose=verbose,
        )
        if reorder_mode not in {"total", "mean"}:
            raise ValueError("reorder_mode must be 'total' or 'mean'")
        self.reorder_mode = reorder_mode
        self.reorder_threshold = reorder_threshold
        self.last_reordered = None
        self.last_relative_range = None

    @staticmethod
    def _bottomk_mask(score, k):
        k = max(0, min(int(k), score.numel()))
        mask = torch.zeros_like(score, dtype=torch.bool)
        if k == 0:
            return mask
        if k == score.numel():
            mask.fill_(True)
            return mask

        indices = torch.topk(
            score.reshape(-1), k=k, largest=False, sorted=False
        ).indices
        mask.reshape(-1)[indices] = True
        return mask

    def _calculate_wanda_reordering(self, W, slice_budgets, blocksize):
        wanda_score = torch.abs(W) * torch.sqrt(
            self.scaler_row.reshape(1, -1)
        )
        slice_priorities = []
        column_orders = []
        slice_widths = []

        for slice_index, i1 in enumerate(range(0, self.columns, blocksize)):
            i2 = min(i1 + blocksize, self.columns)
            score = wanda_score[:, i1:i2]
            budget = slice_budgets[slice_index]
            mask = self._bottomk_mask(score, budget)
            selected_score = score * mask

            if self.reorder_mode == "total":
                slice_priority = selected_score.sum()
                column_priority = selected_score.sum(dim=0)
            else:
                slice_priority = selected_score.sum() / max(budget, 1)
                selected_per_column = mask.sum(dim=0).clamp_min(1)
                column_priority = selected_score.sum(dim=0) / selected_per_column

            local_order = torch.argsort(column_priority, descending=True)
            original_columns = torch.arange(i1, i2, device=W.device)
            column_orders.append(original_columns[local_order])
            slice_priorities.append(slice_priority)
            slice_widths.append(i2 - i1)

        priorities = torch.stack(slice_priorities)
        max_priority = priorities.max().abs()
        if max_priority > 0:
            relative_range = (
                (priorities.max() - priorities.min()) / max_priority
            ).item()
        else:
            relative_range = 0.0

        reordered = relative_range > self.reorder_threshold
        if reordered:
            slice_order = torch.argsort(priorities, descending=True).cpu().tolist()
            permutation = torch.cat(
                [column_orders[slice_index] for slice_index in slice_order]
            )
        else:
            slice_order = list(range(len(slice_budgets)))
            permutation = torch.arange(self.columns, device=W.device)

        ordered_budgets = [slice_budgets[index] for index in slice_order]
        ordered_widths = [slice_widths[index] for index in slice_order]
        return (
            permutation,
            ordered_budgets,
            ordered_widths,
            relative_range,
            reordered,
        )

    def fasterprune(
        self,
        sparsity,
        prune_n=0,
        prune_m=0,
        blocksize=None,
        percdamp=0.01,
    ):
        if prune_n != 0 or prune_m != 0:
            raise ValueError(
                "SparseGPTSliceReorder currently supports only unstructured sparsity"
            )
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must satisfy 0 <= sparsity < 1")
        if self.slice_size <= 0:
            raise ValueError("slice_size must be a positive integer")
        if not 0.0 < self.allocation_step <= 1.0:
            raise ValueError("slice_step_ratio must satisfy 0 < value <= 1")
        if not 0.0 <= self.reorder_threshold <= 1.0:
            raise ValueError("slice_reorder_threshold must satisfy 0 <= value <= 1")

        blocksize = self.slice_size if blocksize is None else blocksize
        tick = time.perf_counter()

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, "quantizer") and not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H_inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))
        Hinv_original = torch.linalg.cholesky(H_inverse, upper=True)

        (
            slice_budgets,
            slice_sizes,
            target_k,
            min_sparsity,
            max_sparsity,
        ) = self._allocate_slice_budgets(
            W, Hinv_original, sparsity, blocksize
        )
        self.last_slice_budgets = list(slice_budgets)
        self.last_slice_sparsities = [
            budget / size for budget, size in zip(slice_budgets, slice_sizes)
        ]

        (
            permutation,
            ordered_budgets,
            ordered_widths,
            relative_range,
            reordered,
        ) = self._calculate_wanda_reordering(W, slice_budgets, blocksize)
        self.last_relative_range = relative_range
        self.last_reordered = reordered

        W = W[:, permutation]
        H_inverse = H_inverse[:, permutation][permutation, :]
        Hinv = torch.linalg.cholesky(H_inverse, upper=True)
        losses = torch.zeros(self.rows, device=self.dev)

        i1 = 0
        for slice_k, width in zip(ordered_budgets, ordered_widths):
            i2 = i1 + width
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            score = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2)
            mask1 = self._bottomk_mask(score, slice_k)

            for i in range(width):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q).pow(2) / d.pow(2)

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(
                    Hinv1[i, i:].unsqueeze(0)
                )
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            losses += Losses1.sum(dim=1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            i1 = i2

        W_restored = W[:, torch.argsort(permutation)]
        if isinstance(self.layer, transformers.Conv1D):
            W_restored = W_restored.t()
        self.layer.weight.data = W_restored.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if W.is_cuda:
            torch.cuda.synchronize()
        if self.verbose:
            ratios = self.last_slice_sparsities
            print(
                "SparseGPTSliceReorder "
                f"mode={self.reorder_mode} "
                f"target={target_k / self.layer.weight.numel():.6f} "
                f"bounds=[{min_sparsity:.6f}, {max_sparsity:.6f}] "
                f"actual_range=[{min(ratios):.6f}, {max(ratios):.6f}] "
                f"relative_range={relative_range:.6f} "
                f"threshold={self.reorder_threshold:.6f} "
                f"reordered={reordered} "
                f"slices={len(ratios)} "
                f"time={time.perf_counter() - tick:.2f}s"
            )
