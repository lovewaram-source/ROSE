import torch

from .ca_sparsegpt_auto_range import CASparseGPTAutoRange
from .slice_budget import allocate_exact_marginal_budgets


class CASparseGPTAutoRangeOnline(CASparseGPTAutoRange):
    """CA auto-range with budget refresh after each compensation round.

    The initial global-mask prediction supplies the first set of slice
    budgets.  After every ``interval`` committed slices, the remaining weight
    state and conditional inverse Hessian are used to predict fresh budgets
    for only the remaining slices.  Already committed slices are immutable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_budget_history = []

    def _refresh_remaining_groups(
        self, W, H_inverse, groups, remaining_target_k, round_index
    ):
        columns = torch.cat([group["indices"] for group in groups])
        W_remaining = W[:, columns]
        Hinv = torch.linalg.cholesky(H_inverse, upper=True)
        predicted_score = (
            W_remaining.pow(2) / torch.diag(Hinv).reshape(1, -1).pow(2)
        )
        predicted_mask = self._bottomk_mask(predicted_score, remaining_target_k)

        score_blocks = []
        predicted_sparsities = []
        offset = 0
        for group in groups:
            width = group["width"]
            i1, i2 = offset, offset + width
            size = self.rows * width
            score_blocks.append(predicted_score[:, i1:i2])
            predicted_sparsities.append(
                float(predicted_mask[:, i1:i2].sum().item()) / size
            )
            offset = i2

        remaining_sparsity = remaining_target_k / predicted_score.numel()
        minimum, maximum = self._auto_bounds(
            predicted_sparsities, remaining_sparsity
        )
        budgets = allocate_exact_marginal_budgets(
            score_blocks,
            remaining_target_k,
            minimum,
            maximum,
            self.allocation_step,
        )
        allocated_sparsities = [
            budget / (self.rows * group["width"])
            for group, budget in zip(groups, budgets)
        ]
        for group, budget in zip(groups, budgets):
            group["budget"] = budget

        self.last_predicted_mask_sparsities = predicted_sparsities
        self.last_auto_min_sparsity = minimum
        self.last_auto_max_sparsity = maximum
        self.last_slice_budgets = list(budgets)
        self.last_slice_sparsities = allocated_sparsities
        self.last_budget_history.append(
            {
                "round": round_index,
                "remaining_target_k": remaining_target_k,
                "predicted_sparsities": list(predicted_sparsities),
                "allocated_sparsities": list(allocated_sparsities),
                "minimum": minimum,
                "maximum": maximum,
            }
        )

        if self.verbose:
            print(
                "CASparseGPTAutoRangeOnline "
                f"round={round_index} "
                f"remaining_target={remaining_target_k} "
                f"target={remaining_sparsity:.6f} "
                "predicted_range="
                f"[{min(predicted_sparsities):.6f}, {max(predicted_sparsities):.6f}] "
                f"bounds=[{minimum:.6f}, {maximum:.6f}] "
                "allocated_range="
                f"[{min(allocated_sparsities):.6f}, {max(allocated_sparsities):.6f}] "
                f"slices={len(groups)}"
            )
        if self.block_verbose:
            for group, predicted, allocated in zip(
                groups, predicted_sparsities, allocated_sparsities
            ):
                print(
                    "CAAutoRangeOnlineBlock "
                    f"round={round_index} block={group['id']} "
                    f"predicted={predicted:.6f} allocated={allocated:.6f}"
                )
        return groups
