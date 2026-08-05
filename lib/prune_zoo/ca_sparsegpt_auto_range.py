import torch

from .ca_sparsegpt_slice import CASparseGPTSlice
from .slice_budget import allocate_exact_marginal_budgets


class CASparseGPTAutoRange(CASparseGPTSlice):
    """CA-Slice with a predicted, bounded per-layer sparsity range.

    SparseGPT's initial Hessian score predicts how a *physical* slice would
    be pruned by a layer-wide mask.  The lowest predicted slice sparsity is
    the automatic lower bound; ``safety_cap`` is the only upper bound.  CA
    still ranks slices with Wanda candidate errors and actual
    pruning/compensation remains SparseGPT, exactly as in
    :class:`CASparseGPTSlice`.

    The automatic upper bound is a hard constraint.  It prevents the budget
    allocator from putting nearly all pruning into a few attention slices.
    """

    def __init__(
        self,
        layer,
        slice_size=128,
        allocation_step=0.01,
        interval=4,
        reorder_threshold=0.0,
        safety_cap=0.95,
        verbose=False,
        block_verbose=False,
    ):
        super().__init__(
            layer,
            slice_size=slice_size,
            min_sparsity=None,
            max_sparsity=None,
            allocation_step=allocation_step,
            interval=interval,
            reorder_threshold=reorder_threshold,
            verbose=verbose,
        )
        if not 0.0 < safety_cap <= 1.0:
            raise ValueError("dynamic_range_safety_cap must be in (0, 1]")

        self.safety_cap = safety_cap
        self.block_verbose = block_verbose
        self.last_predicted_mask_sparsities = None
        self.last_auto_min_sparsity = None
        self.last_auto_max_sparsity = None

    def _auto_bounds(self, predicted_sparsities, sparsity):
        if sparsity > self.safety_cap:
            raise ValueError(
                "Target sparsity exceeds dynamic_range_safety_cap; raise the cap "
                "only if near-total slice pruning is intentional"
            )

        minimum = min(predicted_sparsities)
        maximum = self.safety_cap
        if maximum < sparsity:
            raise ValueError("Dynamic range upper bound cannot satisfy target sparsity")
        return minimum, maximum

    def _build_groups(self, W, H_inverse, sparsity, blocksize):
        Hinv = torch.linalg.cholesky(H_inverse, upper=True)
        predicted_score = W.pow(2) / torch.diag(Hinv).reshape(1, -1).pow(2)
        target_k = int(round(W.numel() * sparsity))
        predicted_mask = self._bottomk_mask(predicted_score, target_k)

        widths = []
        score_blocks = []
        predicted_sparsities = []
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            width = i2 - i1
            size = self.rows * width
            widths.append(width)
            score_blocks.append(predicted_score[:, i1:i2])
            predicted_sparsities.append(
                float(predicted_mask[:, i1:i2].sum().item()) / size
            )

        minimum, maximum = self._auto_bounds(predicted_sparsities, sparsity)
        budgets = allocate_exact_marginal_budgets(
            score_blocks,
            target_k,
            minimum,
            maximum,
            self.allocation_step,
        )
        sizes = [self.rows * width for width in widths]
        allocated_sparsities = [
            budget / size for budget, size in zip(budgets, sizes)
        ]

        self.last_predicted_mask_sparsities = predicted_sparsities
        self.last_auto_min_sparsity = minimum
        self.last_auto_max_sparsity = maximum
        self.last_slice_budgets = list(budgets)
        self.last_slice_sparsities = allocated_sparsities

        groups = []
        for group_id, (i1, width, budget) in enumerate(
            zip(range(0, self.columns, blocksize), widths, budgets)
        ):
            groups.append(
                {
                    "id": group_id,
                    "indices": torch.arange(i1, i1 + width, device=self.dev),
                    "width": width,
                    "budget": budget,
                }
            )

        if self.verbose:
            print(
                "CASparseGPTAutoRange "
                f"target={target_k / W.numel():.6f} "
                "predicted_range="
                f"[{min(predicted_sparsities):.6f}, {max(predicted_sparsities):.6f}] "
                f"bounds=[{minimum:.6f}, {maximum:.6f}] "
                "allocated_range="
                f"[{min(allocated_sparsities):.6f}, {max(allocated_sparsities):.6f}] "
                f"slices={len(groups)}"
            )
        if self.block_verbose:
            for group_id, (predicted, allocated) in enumerate(
                zip(predicted_sparsities, allocated_sparsities)
            ):
                print(
                    "CAAutoRangeBlock "
                    f"block={group_id} predicted={predicted:.6f} "
                    f"allocated={allocated:.6f}"
                )

        return groups, target_k
