import heapq
import math
import time

import torch
import torch.nn as nn
import transformers

from .sparsegpt import SparseGPT


class SparseGPTSlice(SparseGPT):
    """SparseGPT with loss-aware, per-slice dynamic pruning budgets."""

    def __init__(
        self,
        layer,
        slice_size=128,
        min_sparsity=None,
        max_sparsity=None,
        allocation_step=0.01,
        verbose=False,
    ):
        super().__init__(layer)
        self.slice_size = slice_size
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.allocation_step = allocation_step
        self.verbose = verbose
        self.last_slice_budgets = None
        self.last_slice_sparsities = None

    def _resolve_sparsity_bounds(self, sparsity):
        min_sparsity = self.min_sparsity
        max_sparsity = self.max_sparsity

        if min_sparsity is None:
            min_sparsity = max(0.0, sparsity - 0.15)
        if max_sparsity is None:
            max_sparsity = min(1.0, sparsity + 0.15)

        if not 0.0 <= min_sparsity <= sparsity:
            raise ValueError(
                "slice_min_ratio must satisfy 0 <= slice_min_ratio <= sparsity_ratio"
            )
        if not sparsity <= max_sparsity <= 1.0:
            raise ValueError(
                "slice_max_ratio must satisfy sparsity_ratio <= slice_max_ratio <= 1"
            )

        return min_sparsity, max_sparsity

    def _allocate_slice_budgets(self, W, Hinv, sparsity, blocksize):
        """Allocate an exact layer budget using slice marginal pruning losses."""
        min_sparsity, max_sparsity = self._resolve_sparsity_bounds(sparsity)
        target_k = int(round(W.numel() * sparsity))
        hinv_diag = torch.diag(Hinv)

        budgets = []
        max_budgets = []
        slice_sizes = []
        candidate_chunks = []

        for block_index, i1 in enumerate(range(0, self.columns, blocksize)):
            i2 = min(i1 + blocksize, self.columns)
            slice_numel = W[:, i1:i2].numel()
            # Ceil/floor keep the realized integer sparsity inside the bounds.
            min_k = int(math.ceil(slice_numel * min_sparsity))
            max_k = int(math.floor(slice_numel * max_sparsity))
            min_k = max(0, min(min_k, slice_numel))
            max_k = max(0, min(max_k, slice_numel))
            if min_k > max_k:
                raise ValueError(
                    "A slice is too small to represent the requested sparsity bounds: "
                    f"size={slice_numel}, min={min_sparsity}, max={max_sparsity}"
                )

            budgets.append(min_k)
            max_budgets.append(max_k)
            slice_sizes.append(slice_numel)
            candidate_chunks.append([])

            if min_k == max_k:
                continue

            scores = W[:, i1:i2].pow(2) / hinv_diag[i1:i2].reshape(1, -1).pow(2)
            sorted_scores = torch.sort(scores.flatten()).values
            cumulative = torch.cumsum(sorted_scores, dim=0)

            step_k = max(1, int(round(slice_numel * self.allocation_step)))
            starts = torch.arange(
                min_k,
                max_k,
                step_k,
                device=W.device,
                dtype=torch.long,
            )
            ends = torch.clamp(starts + step_k, max=max_k)

            left_cost = torch.zeros_like(starts, dtype=cumulative.dtype)
            has_left = starts > 0
            left_cost[has_left] = cumulative[starts[has_left] - 1]
            marginal_cost = (
                cumulative[ends - 1] - left_cost
            ) / (ends - starts).to(cumulative.dtype)

            starts_cpu = starts.cpu().tolist()
            ends_cpu = ends.cpu().tolist()
            costs_cpu = marginal_cost.cpu().tolist()
            candidate_chunks[block_index].extend(
                (cost, block_index, start, end)
                for cost, start, end in zip(costs_cpu, starts_cpu, ends_cpu)
            )

        min_total = sum(budgets)
        max_total = sum(max_budgets)
        if not min_total <= target_k <= max_total:
            raise ValueError(
                "Slice sparsity bounds cannot satisfy the layer target: "
                f"min={min_total}, target={target_k}, max={max_total}"
            )

        remaining = target_k - min_total
        chunk_positions = [0] * len(candidate_chunks)
        available_chunks = []
        for chunks in candidate_chunks:
            if chunks:
                heapq.heappush(available_chunks, chunks[0])

        while remaining > 0 and available_chunks:
            _, block_index, start, end = heapq.heappop(available_chunks)
            if budgets[block_index] != start:
                raise RuntimeError("Slice marginal-loss chunks are not contiguous")

            chunk_size = end - start
            allocated = min(chunk_size, remaining)
            budgets[block_index] += allocated
            remaining -= allocated

            if allocated == chunk_size:
                chunk_positions[block_index] += 1
                next_position = chunk_positions[block_index]
                if next_position < len(candidate_chunks[block_index]):
                    heapq.heappush(
                        available_chunks,
                        candidate_chunks[block_index][next_position],
                    )

        if remaining != 0:
            raise RuntimeError(
                f"Failed to allocate the complete slice pruning budget; remaining={remaining}"
            )
        if sum(budgets) != target_k:
            raise RuntimeError("Allocated slice budgets do not match the layer target")

        return budgets, slice_sizes, target_k, min_sparsity, max_sparsity

    def fasterprune(
        self,
        sparsity,
        prune_n=0,
        prune_m=0,
        blocksize=None,
        percdamp=0.01,
    ):
        if prune_n != 0 or prune_m != 0:
            raise ValueError("SparseGPTSlice currently supports only unstructured sparsity")
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must satisfy 0 <= sparsity < 1")
        if self.slice_size <= 0:
            raise ValueError("slice_size must be a positive integer")
        if not 0.0 < self.allocation_step <= 1.0:
            raise ValueError("slice_step_ratio must satisfy 0 < value <= 1")

        blocksize = self.slice_size if blocksize is None else blocksize
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, "quantizer"):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()
        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        (
            slice_budgets,
            slice_sizes,
            target_k,
            min_sparsity,
            max_sparsity,
        ) = self._allocate_slice_budgets(W, Hinv, sparsity, blocksize)

        self.last_slice_budgets = list(slice_budgets)
        self.last_slice_sparsities = [
            budget / size for budget, size in zip(slice_budgets, slice_sizes)
        ]

        losses = torch.zeros(self.rows, device=self.dev)

        for block_index, i1 in enumerate(range(0, self.columns, blocksize)):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            slice_k = slice_budgets[block_index]
            mask1 = torch.zeros_like(W1, dtype=torch.bool)
            if slice_k == W1.numel():
                mask1.fill_(True)
            elif slice_k > 0:
                current_score = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2)
                prune_indices = torch.topk(
                    current_score.flatten(),
                    k=slice_k,
                    largest=False,
                    sorted=False,
                ).indices
                mask1.view(-1)[prune_indices] = True

            for i in range(count):
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
            losses += torch.sum(Losses1, dim=1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if W.is_cuda:
            torch.cuda.synchronize()

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if self.verbose:
            ratios = self.last_slice_sparsities
            print(
                "SparseGPTSlice "
                f"target={target_k / self.layer.weight.numel():.6f} "
                f"bounds=[{min_sparsity:.6f}, {max_sparsity:.6f}] "
                f"actual_range=[{min(ratios):.6f}, {max(ratios):.6f}] "
                f"slices={len(ratios)} time={time.time() - tick:.2f}s"
            )
