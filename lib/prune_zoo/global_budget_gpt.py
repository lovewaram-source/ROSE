import heapq
import math
import time

import torch
import torch.nn as nn
import transformers

from .sparsegpt import SparseGPT


def build_global_profile(
    wrapped_layer,
    min_sparsity,
    max_sparsity,
    allocation_step,
    percdamp=0.01,
):
    """Compress one sublayer's Hessian loss curve into marginal chunks."""
    W = wrapped_layer.layer.weight.data.clone()
    if isinstance(wrapped_layer.layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(wrapped_layer.layer, transformers.Conv1D):
        W = W.t()
    W = W.float()

    H = wrapped_layer.H
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    diagonal = torch.arange(wrapped_layer.columns, device=wrapped_layer.dev)
    H[diagonal, diagonal] += percdamp * torch.mean(torch.diag(H))
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))
    factor = torch.linalg.cholesky(inverse, upper=True)
    scores = W.pow(2) / torch.diag(factor).reshape(1, -1).pow(2)
    sorted_scores = torch.sort(scores.reshape(-1)).values
    cumulative = torch.cumsum(sorted_scores, dim=0)

    size = scores.numel()
    minimum = max(0, min(int(math.ceil(size * min_sparsity)), size))
    maximum = max(0, min(int(math.floor(size * max_sparsity)), size))
    if minimum > maximum:
        raise ValueError("A sublayer is too small for the global budget bounds")

    chunks = []
    if minimum < maximum:
        step = max(1, int(round(size * allocation_step)))
        starts = torch.arange(
            minimum, maximum, step, device=scores.device, dtype=torch.long
        )
        ends = torch.clamp(starts + step, max=maximum)
        left = torch.zeros_like(starts, dtype=cumulative.dtype)
        has_left = starts > 0
        left[has_left] = cumulative[starts[has_left] - 1]
        costs = (cumulative[ends - 1] - left) / (ends - starts).to(
            cumulative.dtype
        )
        chunks = list(zip(costs.cpu().tolist(), starts.cpu().tolist(), ends.cpu().tolist()))

    return {
        "size": size,
        "minimum": minimum,
        "maximum": maximum,
        "chunks": chunks,
    }


def allocate_global_budgets(profiles, sparsity):
    """Allocate one exact integer pruning budget across all profiled sublayers."""
    target = int(round(sum(profile["size"] for profile in profiles) * sparsity))
    budgets = [profile["minimum"] for profile in profiles]
    maximums = [profile["maximum"] for profile in profiles]
    if not sum(budgets) <= target <= sum(maximums):
        raise ValueError("Global sparsity bounds cannot satisfy the model target")

    positions = [0] * len(profiles)
    heap = []
    for profile_id, profile in enumerate(profiles):
        if profile["chunks"]:
            cost, start, end = profile["chunks"][0]
            heapq.heappush(heap, (cost, profile_id, start, end))

    remaining = target - sum(budgets)
    while remaining > 0 and heap:
        _, profile_id, start, end = heapq.heappop(heap)
        if budgets[profile_id] != start:
            raise RuntimeError("Global marginal chunks are not contiguous")
        amount = min(end - start, remaining)
        budgets[profile_id] += amount
        remaining -= amount
        if budgets[profile_id] == end:
            positions[profile_id] += 1
            position = positions[profile_id]
            chunks = profiles[profile_id]["chunks"]
            if position < len(chunks):
                cost, start, end = chunks[position]
                heapq.heappush(heap, (cost, profile_id, start, end))

    if remaining != 0 or sum(budgets) != target:
        raise RuntimeError("Failed to allocate the exact global pruning budget")
    return budgets, target


class GlobalBudgetSparseGPT(SparseGPT):
    """SparseGPT compensation using an externally assigned exact sublayer budget."""

    def __init__(self, layer, target_k, verbose=False):
        super().__init__(layer)
        self.target_k = int(target_k)
        self.verbose = verbose

    def fasterprune(
        self,
        sparsity,
        prune_n=0,
        prune_m=0,
        blocksize=128,
        percdamp=0.01,
    ):
        del sparsity
        if prune_n or prune_m:
            raise ValueError("GlobalBudgetGPT supports only unstructured sparsity")
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
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))
        Hinv = torch.linalg.cholesky(inverse, upper=True)

        score = W.pow(2) / torch.diag(Hinv).reshape(1, -1).pow(2)
        mask = torch.zeros_like(score, dtype=torch.bool)
        if self.target_k:
            indices = torch.topk(
                score.reshape(-1),
                k=self.target_k,
                largest=False,
                sorted=False,
            ).indices
            mask.reshape(-1)[indices] = True

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            errors = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            mask1 = mask[:, i1:i2]
            for column in range(i2 - i1):
                weight = W1[:, column]
                divisor = Hinv1[column, column]
                quantized = weight.clone()
                quantized[mask1[:, column]] = 0
                Q1[:, column] = quantized
                error = (weight - quantized) / divisor
                W1[:, column:] -= error.unsqueeze(1).matmul(
                    Hinv1[column, column:].unsqueeze(0)
                )
                errors[:, column] = error
            W[:, i1:i2] = Q1
            W[:, i2:] -= errors.matmul(Hinv[i1:i2, i2:])

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if W.is_cuda:
            torch.cuda.synchronize()
        if self.verbose:
            print(
                "GlobalBudgetGPT "
                f"budget={self.target_k}/{mask.numel()} "
                f"ratio={self.target_k / mask.numel():.6f} "
                f"time={time.perf_counter() - tick:.2f}s"
            )
