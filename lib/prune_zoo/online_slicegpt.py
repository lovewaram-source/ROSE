import time

import torch
import torch.nn as nn
import transformers

from .rose_dynamic import ROSEDynamic
from .slice_budget import allocate_exact_marginal_budgets


class OnlineSliceGPT(ROSEDynamic):
    """Reallocate the remaining slice budget after every compensated slice."""

    def __init__(
        self,
        layer,
        slice_size=128,
        min_sparsity=None,
        max_sparsity=None,
        allocation_step=0.01,
        verbose=False,
    ):
        super().__init__(layer, blocksize=slice_size, interval=1, verbose=verbose)
        self.slice_size = slice_size
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.allocation_step = allocation_step
        self.last_slice_budgets = None
        self.last_slice_sparsities = None

    def _resolve_bounds(self, sparsity):
        minimum = (
            max(0.0, sparsity - 0.15)
            if self.min_sparsity is None
            else self.min_sparsity
        )
        maximum = (
            min(1.0, sparsity + 0.15)
            if self.max_sparsity is None
            else self.max_sparsity
        )
        if not 0.0 <= minimum <= sparsity <= maximum <= 1.0:
            raise ValueError("Invalid OnlineSliceGPT sparsity bounds")
        return minimum, maximum

    def fasterprune(
        self,
        sparsity,
        prune_n=0,
        prune_m=0,
        blocksize=None,
        percdamp=0.01,
    ):
        if prune_n != 0 or prune_m != 0:
            raise ValueError("OnlineSliceGPT supports only unstructured sparsity")
        blocksize = self.slice_size if blocksize is None else blocksize
        if blocksize <= 0 or not 0.0 <= sparsity < 1.0:
            raise ValueError("Invalid OnlineSliceGPT pruning configuration")
        minimum, maximum = self._resolve_bounds(sparsity)
        tick = time.perf_counter()

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H.clone()
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += percdamp * torch.mean(torch.diag(H))
        H_inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))

        groups = []
        for group_id, i1 in enumerate(range(0, self.columns, blocksize)):
            i2 = min(i1 + blocksize, self.columns)
            groups.append(
                {
                    "id": group_id,
                    "indices": torch.arange(i1, i2, device=self.dev),
                    "width": i2 - i1,
                }
            )

        target_k = int(round(W.numel() * sparsity))
        remaining_target = target_k
        selected_budgets = [0] * len(groups)
        W_state = W.clone()
        W_result = torch.zeros_like(W)
        remaining_groups = groups

        while remaining_groups:
            Hinv = torch.linalg.cholesky(H_inverse, upper=True)
            diagonal = torch.diag(Hinv)
            score_blocks = []
            offset = 0
            for group in remaining_groups:
                width = group["width"]
                score_blocks.append(
                    W_state[:, group["indices"]].pow(2)
                    / diagonal[offset : offset + width].reshape(1, -1).pow(2)
                )
                offset += width

            budgets = allocate_exact_marginal_budgets(
                score_blocks,
                remaining_target,
                minimum,
                maximum,
                self.allocation_step,
            )
            selected = dict(remaining_groups[0])
            selected["budget"] = budgets[0]
            selected_budgets[selected["id"]] = budgets[0]

            current_columns = torch.cat(
                [group["indices"] for group in remaining_groups]
            )
            W_round = W_state[:, current_columns].clone()
            W_round, finalized, selected_width = self._prune_selected_prefix(
                W_round, Hinv, [selected]
            )
            W_result[:, selected["indices"]] = finalized
            remaining_target -= selected["budget"]

            remaining_groups = remaining_groups[1:]
            if remaining_groups:
                remaining_columns = current_columns[selected_width:]
                W_state[:, remaining_columns] = W_round[:, selected_width:]
                trailing = Hinv[selected_width:, selected_width:]
                H_inverse = trailing.t().matmul(trailing)
                H_inverse = (H_inverse + H_inverse.t()) / 2

        if remaining_target != 0 or sum(selected_budgets) != target_k:
            raise RuntimeError("OnlineSliceGPT did not consume the exact target budget")
        sizes = [self.rows * group["width"] for group in groups]
        self.last_slice_budgets = selected_budgets
        self.last_slice_sparsities = [
            budget / size for budget, size in zip(selected_budgets, sizes)
        ]

        if isinstance(self.layer, transformers.Conv1D):
            W_result = W_result.t()
        self.layer.weight.data = W_result.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if W.is_cuda:
            torch.cuda.synchronize()
        if self.verbose:
            print(
                "OnlineSliceGPT "
                f"target={target_k / W.numel():.6f} "
                f"actual_range=[{min(self.last_slice_sparsities):.6f}, "
                f"{max(self.last_slice_sparsities):.6f}] "
                f"slices={len(groups)} time={time.perf_counter() - tick:.2f}s"
            )
