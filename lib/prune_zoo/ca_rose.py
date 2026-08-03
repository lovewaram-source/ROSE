import math
import time

import torch
import torch.nn as nn
import transformers

from .rose_dynamic import ROSEDynamic


class CAROSE(ROSEDynamic):
    """Compensation-aware ROSE with incremental inverse-Hessian updates."""

    def __init__(self, layer, blocksize=128, interval=4, verbose=False):
        super().__init__(
            layer,
            blocksize=blocksize,
            interval=interval,
            verbose=verbose,
        )
        self.last_target_k = None

    @staticmethod
    def _allocate_exact_budgets(rows, widths, sparsity):
        sizes = [rows * width for width in widths]
        target_k = int(round(sum(sizes) * sparsity))
        raw_budgets = [size * sparsity for size in sizes]
        budgets = [int(math.floor(value)) for value in raw_budgets]

        remaining = target_k - sum(budgets)
        fractional_order = sorted(
            range(len(widths)),
            key=lambda index: raw_budgets[index] - budgets[index],
            reverse=True,
        )
        for index in fractional_order[:remaining]:
            budgets[index] += 1

        if sum(budgets) != target_k:
            raise RuntimeError("CA-ROSE failed to allocate the exact layer budget")
        return budgets, target_k

    def _rank_compensation_aware(self, W, groups, H_inverse):
        priorities = []
        local_column_orders = []
        offset = 0

        for group in groups:
            width = group["width"]
            i1 = offset
            i2 = i1 + width
            indices = group["indices"]

            wanda_score = torch.abs(W[:, indices]) * torch.sqrt(
                self.scaler_row[indices].reshape(1, -1)
            )
            candidate_mask = self._bottomk_mask(
                wanda_score, group["budget"]
            )
            candidate_error = W[:, indices] * candidate_mask

            inverse_block = H_inverse[i1:i2, i1:i2]
            inverse_block = (inverse_block + inverse_block.t()) / 2
            residual_curvature = torch.cholesky_inverse(
                torch.linalg.cholesky(inverse_block)
            )

            weighted_error = candidate_error.matmul(residual_curvature)
            element_contribution = candidate_error * weighted_error / 2
            priorities.append(element_contribution.sum())
            column_priority = element_contribution.sum(dim=0)
            local_column_orders.append(
                torch.argsort(column_priority, descending=True)
            )
            offset = i2

        priorities = torch.stack(priorities)
        ranking = torch.argsort(priorities, descending=True).cpu().tolist()
        return ranking, local_column_orders, priorities

    def fasterprune(
        self,
        sparsity,
        prune_n=0,
        prune_m=0,
        blocksize=None,
        percdamp=0.01,
    ):
        if prune_n != 0 or prune_m != 0:
            raise ValueError("CA-ROSE currently supports only unstructured sparsity")
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must satisfy 0 <= sparsity < 1")

        blocksize = self.blocksize if blocksize is None else blocksize
        if blocksize <= 0:
            raise ValueError("ca_rose_blocksize must be a positive integer")
        if self.interval <= 0:
            raise ValueError("ca_rose_interval must be a positive integer")

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
        H_inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))

        widths = [
            min(i1 + blocksize, self.columns) - i1
            for i1 in range(0, self.columns, blocksize)
        ]
        budgets, target_k = self._allocate_exact_budgets(
            self.rows, widths, sparsity
        )
        self.last_target_k = target_k

        groups = []
        for group_id, (i1, width, budget) in enumerate(
            zip(range(0, self.columns, blocksize), widths, budgets)
        ):
            groups.append(
                {
                    "id": group_id,
                    "indices": torch.arange(
                        i1, i1 + width, device=self.dev
                    ),
                    "width": width,
                    "budget": budget,
                }
            )

        W_state = W.clone()
        W_result = torch.zeros_like(W)
        remaining_groups = groups
        processed_group_ids = []
        rounds = 0

        while remaining_groups:
            rounds += 1
            current_columns = torch.cat(
                [group["indices"] for group in remaining_groups]
            )
            ranking, column_orders, priorities = self._rank_compensation_aware(
                W_state, remaining_groups, H_inverse
            )
            selected_positions = ranking[: self.interval]
            selected_groups = [remaining_groups[pos] for pos in selected_positions]
            selected_ids = {group["id"] for group in selected_groups}

            group_local_positions = []
            offset = 0
            for group in remaining_groups:
                positions = torch.arange(
                    offset, offset + group["width"], device=self.dev
                )
                group_local_positions.append(positions)
                offset += group["width"]

            selected_local = torch.cat(
                [
                    group_local_positions[pos][column_orders[pos]]
                    for pos in selected_positions
                ]
            )
            next_remaining_groups = [
                group
                for group in remaining_groups
                if group["id"] not in selected_ids
            ]
            remaining_positions = [
                group_local_positions[pos]
                for pos, group in enumerate(remaining_groups)
                if group["id"] not in selected_ids
            ]
            if remaining_positions:
                local_permutation = torch.cat(
                    [selected_local, torch.cat(remaining_positions)]
                )
            else:
                local_permutation = selected_local

            global_permutation = current_columns[local_permutation]
            selected_width = selected_local.numel()
            selected_columns = global_permutation[:selected_width]
            remaining_columns = global_permutation[selected_width:]

            inverse_permuted = H_inverse[:, local_permutation][
                local_permutation, :
            ]
            Hinv = torch.linalg.cholesky(inverse_permuted, upper=True)
            W_round = W_state[:, global_permutation].clone()
            W_round, finalized, finalized_width = self._prune_selected_prefix(
                W_round, Hinv, selected_groups
            )
            if finalized_width != selected_width:
                raise RuntimeError("CA-ROSE selected width mismatch")

            W_result[:, selected_columns] = finalized
            processed_group_ids.extend(group["id"] for group in selected_groups)

            if next_remaining_groups:
                W_state[:, remaining_columns] = W_round[:, selected_width:]
                trailing_factor = Hinv[selected_width:, selected_width:]
                H_inverse = trailing_factor.t().matmul(trailing_factor)
                H_inverse = (H_inverse + H_inverse.t()) / 2

            if self.verbose:
                selected_priority = [
                    priorities[pos].item() for pos in selected_positions
                ]
                print(
                    "CAROSERound "
                    f"round={rounds} "
                    f"selected={[group['id'] for group in selected_groups]} "
                    f"residual_loss={[f'{value:.6e}' for value in selected_priority]} "
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
            actual_k = int((self.layer.weight.data == 0).sum().item())
            print(
                "CAROSE "
                f"target={target_k / W.numel():.6f} "
                f"actual={actual_k / W.numel():.6f} "
                f"blocksize={blocksize} "
                f"interval={self.interval} "
                f"rounds={rounds} "
                f"time={time.perf_counter() - tick:.2f}s"
            )
