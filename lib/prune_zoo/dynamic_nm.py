import time

import torch

from .ca_rose import CAROSE


class DynamicNM(CAROSE):
    """Dynamic block ordering with strict N:M masks in original column groups."""

    def __init__(
        self,
        layer,
        blocksize=128,
        interval=4,
        reorder_threshold=0.0,
        verbose=False,
    ):
        super().__init__(
            layer,
            blocksize=blocksize,
            interval=interval,
            reorder_threshold=reorder_threshold,
            verbose=verbose,
        )
        self.prune_n = None
        self.prune_m = None

    def _build_groups(self, W, H_inverse, sparsity, blocksize):
        del H_inverse, sparsity
        if blocksize % self.prune_m:
            raise ValueError("dynamic_nm_blocksize must be divisible by M")
        if self.columns % self.prune_m:
            raise ValueError("The input dimension must be divisible by M")
        groups = []
        target = 0
        for group_id, i1 in enumerate(range(0, self.columns, blocksize)):
            i2 = min(i1 + blocksize, self.columns)
            width = i2 - i1
            budget = self.rows * (width // self.prune_m) * self.prune_n
            target += budget
            groups.append(
                {
                    "id": group_id,
                    "indices": torch.arange(i1, i2, device=self.dev),
                    "width": width,
                    "budget": budget,
                }
            )
        return groups, target

    def _structured_mask(self, score):
        mask = torch.zeros_like(score, dtype=torch.bool)
        for i1 in range(0, score.shape[1], self.prune_m):
            block = score[:, i1 : i1 + self.prune_m]
            indices = torch.topk(
                block, self.prune_n, dim=1, largest=False, sorted=False
            ).indices
            mask.scatter_(1, i1 + indices, True)
        return mask

    def _rank_compensation_aware(self, W, groups, H_inverse):
        del H_inverse
        priorities = []
        column_orders = []
        for group in groups:
            indices = group["indices"]
            score = torch.abs(W[:, indices]) * torch.sqrt(
                self.scaler_row[indices].reshape(1, -1)
            )
            mask = self._structured_mask(score)
            priorities.append((score * mask).sum())
            # Keep physical M-tuples intact; only whole blocks are reordered.
            column_orders.append(torch.arange(group["width"], device=self.dev))
        priorities = torch.stack(priorities)
        scale = priorities.abs().max().clamp_min(torch.finfo(priorities.dtype).eps)
        relative_range = (priorities.max() - priorities.min()) / scale
        reordered = bool(relative_range.item() >= self.reorder_threshold)
        self.last_reorder_score = relative_range.item()
        self.last_reorder_applied = reordered
        ranking = (
            torch.argsort(priorities, descending=True).cpu().tolist()
            if reordered
            else list(range(len(groups)))
        )
        return ranking, column_orders, priorities

    def _prune_selected_prefix(self, W, Hinv, groups):
        offset = 0
        finalized = []
        for group in groups:
            width = group["width"]
            i1, i2 = offset, offset + width
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            errors = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            score = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2)
            mask = self._structured_mask(score)
            for column in range(width):
                weight = W1[:, column]
                divisor = Hinv1[column, column]
                quantized = weight.clone()
                quantized[mask[:, column]] = 0
                Q1[:, column] = quantized
                error = (weight - quantized) / divisor
                W1[:, column:] -= error.unsqueeze(1).matmul(
                    Hinv1[column, column:].unsqueeze(0)
                )
                errors[:, column] = error
            W[:, i1:i2] = Q1
            W[:, i2:] -= errors.matmul(Hinv[i1:i2, i2:])
            finalized.append(Q1)
            offset = i2
        return W, torch.cat(finalized, dim=1), offset

    def fasterprune(
        self,
        sparsity,
        prune_n=0,
        prune_m=0,
        blocksize=None,
        percdamp=0.01,
    ):
        if prune_n <= 0 or prune_m <= 0 or prune_n >= prune_m:
            raise ValueError("DynamicNM requires a valid structured N:M pattern")
        if abs(sparsity - prune_n / prune_m) > 1e-9:
            raise ValueError("sparsity_ratio must equal N/M for DynamicNM")
        self.prune_n, self.prune_m = prune_n, prune_m
        tick = time.perf_counter()
        super().fasterprune(
            sparsity,
            prune_n=0,
            prune_m=0,
            blocksize=blocksize,
            percdamp=percdamp,
        )
        if self.verbose:
            print(
                f"DynamicNM pattern={prune_n}:{prune_m} "
                f"time={time.perf_counter() - tick:.2f}s"
            )
