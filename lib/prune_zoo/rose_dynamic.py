import time

import torch
import torch.nn as nn
import transformers

from .rose import ROSE


class ROSEDynamic(ROSE):
    """ROSE with online block re-ranking after Hessian compensation."""

    def __init__(
        self,
        layer,
        blocksize=128,
        interval=4,
        reorder_threshold=0.0,
        verbose=False,
    ):
        super().__init__(layer)
        if not 0.0 <= reorder_threshold <= 1.0:
            raise ValueError("rose_dynamic_reorder_threshold must be in [0, 1]")
        self.blocksize = blocksize
        self.interval = interval
        self.reorder_threshold = reorder_threshold
        self.verbose = verbose
        self.last_group_order = None
        self.last_rounds = None
        self.last_reorder_score = None
        self.last_reorder_applied = None

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

    def _rank_remaining_groups(self, W, groups):
        priorities = []
        column_orders = []

        for group in groups:
            indices = group["indices"]
            score = torch.abs(W[:, indices]) * torch.sqrt(
                self.scaler_row[indices].reshape(1, -1)
            )
            mask = self._bottomk_mask(score, group["budget"])
            selected_score = score * mask

            priorities.append(selected_score.sum())
            local_order = torch.argsort(
                selected_score.sum(dim=0), descending=True
            )
            column_orders.append(indices[local_order])

        priorities = torch.stack(priorities)
        scale = priorities.abs().max().clamp_min(torch.finfo(priorities.dtype).eps)
        relative_range = (priorities.max() - priorities.min()) / scale
        reordered = bool(relative_range.item() >= self.reorder_threshold)
        self.last_reorder_score = relative_range.item()
        self.last_reorder_applied = reordered
        if reordered:
            ranking = torch.argsort(priorities, descending=True).cpu().tolist()
        else:
            ranking = list(range(len(groups)))
            column_orders = [group["indices"] for group in groups]
        return ranking, column_orders, priorities

    def _prune_selected_prefix(self, W, Hinv, groups):
        offset = 0
        finalized = []

        for group in groups:
            width = group["width"]
            i1 = offset
            i2 = i1 + width
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            score = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2)
            mask = self._bottomk_mask(score, group["budget"])

            for i in range(width):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()
                q[mask[:, i]] = 0

                Q1[:, i] = q
                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1).matmul(
                    Hinv1[i, i:].unsqueeze(0)
                )
                Err1[:, i] = err

            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
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
        if prune_n != 0 or prune_m != 0:
            raise ValueError("ROSEDynamic currently supports only unstructured sparsity")
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must satisfy 0 <= sparsity < 1")

        blocksize = self.blocksize if blocksize is None else blocksize
        if blocksize <= 0:
            raise ValueError("rose_dynamic_blocksize must be a positive integer")
        if self.interval <= 0:
            raise ValueError("rose_dynamic_interval must be a positive integer")

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

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        groups = []
        for group_id, i1 in enumerate(range(0, self.columns, blocksize)):
            i2 = min(i1 + blocksize, self.columns)
            width = i2 - i1
            groups.append(
                {
                    "id": group_id,
                    "indices": torch.arange(i1, i2, device=self.dev),
                    "width": width,
                    "budget": int(self.rows * width * sparsity),
                }
            )

        W_state = W.clone()
        W_result = torch.zeros_like(W)
        remaining_groups = groups
        processed_group_ids = []
        rounds = 0

        while remaining_groups:
            rounds += 1
            ranking, column_orders, priorities = self._rank_remaining_groups(
                W_state, remaining_groups
            )
            selected_positions = ranking[: self.interval]
            selected_groups = [remaining_groups[pos] for pos in selected_positions]
            selected_ids = {group["id"] for group in selected_groups}
            selected_columns = torch.cat(
                [column_orders[pos] for pos in selected_positions]
            )

            next_remaining_groups = [
                group
                for group in remaining_groups
                if group["id"] not in selected_ids
            ]
            if next_remaining_groups:
                remaining_columns = torch.cat(
                    [group["indices"] for group in next_remaining_groups]
                )
                permutation = torch.cat([selected_columns, remaining_columns])
            else:
                permutation = selected_columns

            W_round = W_state[:, permutation].clone()
            H_round = H[:, permutation][permutation, :]
            H_inverse = torch.cholesky_inverse(torch.linalg.cholesky(H_round))
            Hinv = torch.linalg.cholesky(H_inverse, upper=True)

            W_round, finalized, selected_width = self._prune_selected_prefix(
                W_round, Hinv, selected_groups
            )
            W_result[:, selected_columns] = finalized
            processed_group_ids.extend(group["id"] for group in selected_groups)

            if next_remaining_groups:
                W_state[:, remaining_columns] = W_round[:, selected_width:]

            if self.verbose:
                selected_priority = [
                    priorities[pos].item() for pos in selected_positions
                ]
                print(
                    "ROSEDynamicRound "
                    f"round={rounds} "
                    f"selected={[group['id'] for group in selected_groups]} "
                    f"priority={[f'{value:.6e}' for value in selected_priority]} "
                    f"relative_range={self.last_reorder_score:.6f} "
                    f"reordered={self.last_reorder_applied} "
                    f"remaining={len(next_remaining_groups)}"
                )

            remaining_groups = next_remaining_groups

        self.last_group_order = processed_group_ids
        self.last_rounds = rounds

        if isinstance(self.layer, transformers.Conv1D):
            W_result = W_result.t()
        self.layer.weight.data = W_result.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if W.is_cuda:
            torch.cuda.synchronize()
        if self.verbose:
            print(
                "ROSEDynamic "
                f"blocksize={blocksize} "
                f"interval={self.interval} "
                f"reorder_threshold={self.reorder_threshold:.6f} "
                f"rounds={rounds} "
                f"time={time.perf_counter() - tick:.2f}s"
            )
