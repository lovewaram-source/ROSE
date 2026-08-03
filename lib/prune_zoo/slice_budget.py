import heapq
import math

import torch


def allocate_exact_marginal_budgets(
    score_blocks,
    target_k,
    min_sparsity,
    max_sparsity,
    allocation_step,
):
    """Allocate an exact integer budget using marginal score chunks."""
    budgets = []
    max_budgets = []
    candidates = []

    for block_index, scores in enumerate(score_blocks):
        size = scores.numel()
        min_k = max(0, min(int(math.ceil(size * min_sparsity)), size))
        max_k = max(0, min(int(math.floor(size * max_sparsity)), size))
        if min_k > max_k:
            raise ValueError(
                "A slice is too small to represent the requested sparsity bounds"
            )

        budgets.append(min_k)
        max_budgets.append(max_k)
        candidates.append([])
        if min_k == max_k:
            continue

        sorted_scores = torch.sort(scores.reshape(-1)).values
        cumulative = torch.cumsum(sorted_scores, dim=0)
        step_k = max(1, int(round(size * allocation_step)))
        starts = torch.arange(
            min_k, max_k, step_k, device=scores.device, dtype=torch.long
        )
        ends = torch.clamp(starts + step_k, max=max_k)
        left = torch.zeros_like(starts, dtype=cumulative.dtype)
        has_left = starts > 0
        left[has_left] = cumulative[starts[has_left] - 1]
        cost = (cumulative[ends - 1] - left) / (ends - starts).to(
            cumulative.dtype
        )
        candidates[block_index].extend(
            (float(c), block_index, int(start), int(end))
            for c, start, end in zip(
                cost.cpu().tolist(), starts.cpu().tolist(), ends.cpu().tolist()
            )
        )

    if not sum(budgets) <= target_k <= sum(max_budgets):
        raise ValueError(
            "Slice bounds cannot satisfy the requested remaining target budget"
        )

    remaining = target_k - sum(budgets)
    positions = [0] * len(candidates)
    heap = []
    for chunks in candidates:
        if chunks:
            heapq.heappush(heap, chunks[0])

    while remaining > 0 and heap:
        _, block_index, start, end = heapq.heappop(heap)
        if budgets[block_index] != start:
            raise RuntimeError("Marginal budget chunks are not contiguous")
        allocated = min(end - start, remaining)
        budgets[block_index] += allocated
        remaining -= allocated
        if budgets[block_index] == end:
            positions[block_index] += 1
            position = positions[block_index]
            if position < len(candidates[block_index]):
                heapq.heappush(heap, candidates[block_index][position])

    if remaining != 0 or sum(budgets) != target_k:
        raise RuntimeError("Failed to allocate the exact marginal budget")
    return budgets
