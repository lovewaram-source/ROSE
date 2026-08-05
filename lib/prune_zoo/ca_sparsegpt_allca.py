import math

import torch

from .ca_sparsegpt_globalmin import CASparseGPTGlobalMin
from .slice_budget import allocate_exact_marginal_budgets


class CASparseGPTAllCA(CASparseGPTGlobalMin):
    """All-CA slice pruning with SparseGPT error compensation.

    All pruning decisions use a compensation-aware quadratic proxy.  Exact
    mask optimization is combinatorial, so masks are built by chunked forward
    greedy selection.  SparseGPT's sequential Hessian compensation remains the
    execution mechanism after the mask has been selected.
    """

    def __init__(
        self,
        layer,
        slice_size=128,
        allocation_step=0.01,
        interval=4,
        reorder_threshold=0.0,
        greedy_steps=2,
        verbose=False,
        block_verbose=False,
    ):
        if greedy_steps <= 0:
            raise ValueError("allca_greedy_steps must be a positive integer")
        super().__init__(
            layer,
            slice_size=slice_size,
            allocation_step=allocation_step,
            interval=interval,
            reorder_threshold=reorder_threshold,
            verbose=verbose,
            block_verbose=block_verbose,
        )
        self.greedy_steps = greedy_steps

    @staticmethod
    def _curvature_from_inverse(inverse_block):
        inverse_block = (inverse_block + inverse_block.t()) / 2
        return torch.cholesky_inverse(torch.linalg.cholesky(inverse_block))

    @staticmethod
    def _ca_diagonal_score(W, curvature):
        return W.pow(2) * torch.diag(curvature).reshape(1, -1) / 2

    def _ca_greedy_mask(self, W, curvature, budget):
        """Approximate the cardinality-constrained CA quadratic mask."""
        budget = max(0, min(int(budget), W.numel()))
        mask = torch.zeros_like(W, dtype=torch.bool)
        if budget == 0:
            return mask
        if budget == W.numel():
            mask.fill_(True)
            return mask

        error = torch.zeros_like(W)
        selected = 0
        steps = min(self.greedy_steps, budget)
        diagonal_term = (
            W.pow(2) * torch.diag(curvature).reshape(1, -1) / 2
        )

        for step in range(steps):
            remaining = budget - selected
            remaining_steps = steps - step
            take = int(math.ceil(remaining / remaining_steps))

            # Adding W_ij to the current pruning error changes
            # 1/2 E C E^T by W_ij (E C)_ij + 1/2 W_ij^2 C_jj.
            interaction = error.matmul(curvature)
            marginal = W * interaction + diagonal_term
            marginal = marginal.masked_fill(mask, float("inf"))
            indices = torch.topk(
                marginal.reshape(-1),
                k=take,
                largest=False,
                sorted=False,
            ).indices
            mask.reshape(-1)[indices] = True
            error = W * mask
            selected += take

        if int(mask.sum().item()) != budget:
            raise RuntimeError("All-CA greedy mask failed to meet its budget")
        return mask

    def _build_groups(self, W, H_inverse, sparsity, blocksize):
        score_blocks = []
        widths = []
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            inverse_block = H_inverse[i1:i2, i1:i2]
            curvature = self._curvature_from_inverse(inverse_block)
            score_blocks.append(
                self._ca_diagonal_score(W[:, i1:i2], curvature)
            )
            widths.append(i2 - i1)

        # The global CA mask uses the same block-conditional diagonal proxy as
        # the marginal allocator.  Physical column order is left unchanged.
        global_score = torch.cat(score_blocks, dim=1)
        target_k = int(round(W.numel() * sparsity))
        predicted_mask = self._bottomk_mask(global_score, target_k)
        predicted_sparsities = []
        for i1, width in zip(range(0, self.columns, blocksize), widths):
            size = self.rows * width
            predicted_sparsities.append(
                float(predicted_mask[:, i1:i1 + width].sum().item()) / size
            )

        auto_min = min(predicted_sparsities) if predicted_sparsities else 0.0
        budgets = allocate_exact_marginal_budgets(
            score_blocks,
            target_k,
            auto_min,
            1.0,
            self.allocation_step,
        )
        sizes = [self.rows * width for width in widths]
        allocated_sparsities = [
            budget / size for budget, size in zip(budgets, sizes)
        ]

        self.last_predicted_mask_sparsities = predicted_sparsities
        self.last_auto_min_sparsity = auto_min
        self.last_slice_budgets = list(budgets)
        self.last_slice_sparsities = allocated_sparsities

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

        if self.verbose:
            print(
                "CASparseGPTAllCA "
                f"target={target_k / W.numel():.6f} "
                "predicted_range="
                f"[{min(predicted_sparsities):.6f}, "
                f"{max(predicted_sparsities):.6f}] "
                f"auto_min={auto_min:.6f} "
                "allocated_range="
                f"[{min(allocated_sparsities):.6f}, "
                f"{max(allocated_sparsities):.6f}] "
                f"slices={len(groups)} greedy_steps={self.greedy_steps}"
            )
        if self.block_verbose:
            for group_id, (predicted, budget, allocated) in enumerate(
                zip(predicted_sparsities, budgets, allocated_sparsities)
            ):
                print(
                    "AllCAGlobalBlock "
                    f"block={group_id} predicted={predicted:.6f} "
                    f"budget={budget} allocated={allocated:.6f}"
                )

        return groups, target_k

    def _rank_compensation_aware(self, W, groups, H_inverse):
        priorities = []
        local_column_orders = []
        offset = 0

        for group in groups:
            width = group["width"]
            i1 = offset
            i2 = i1 + width
            indices = group["indices"]
            curvature = self._curvature_from_inverse(
                H_inverse[i1:i2, i1:i2]
            )
            candidate_error_source = W[:, indices]
            candidate_mask = self._ca_greedy_mask(
                candidate_error_source,
                curvature,
                group["budget"],
            )
            candidate_error = candidate_error_source * candidate_mask
            weighted_error = candidate_error.matmul(curvature)
            element_contribution = candidate_error * weighted_error / 2
            priorities.append(element_contribution.sum())
            local_column_orders.append(
                torch.argsort(
                    element_contribution.sum(dim=0), descending=True
                )
            )
            offset = i2

        priorities = torch.stack(priorities)
        scale = priorities.abs().max().clamp_min(
            torch.finfo(priorities.dtype).eps
        )
        relative_range = (priorities.max() - priorities.min()) / scale
        reorder = bool(relative_range.item() >= self.reorder_threshold)
        self.last_reorder_score = relative_range.item()
        self.last_reorder_applied = reorder
        if reorder:
            ranking = torch.argsort(priorities, descending=True).cpu().tolist()
        else:
            ranking = list(range(len(groups)))
            local_column_orders = [
                torch.arange(group["width"], device=self.dev)
                for group in groups
            ]
        return ranking, local_column_orders, priorities

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

            conditional_inverse = Hinv1.t().matmul(Hinv1)
            curvature = self._curvature_from_inverse(conditional_inverse)
            mask = self._ca_greedy_mask(W1, curvature, group["budget"])

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
                Err1[:, column] = error

            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            finalized.append(Q1)
            offset = i2

        return W, torch.cat(finalized, dim=1), offset
