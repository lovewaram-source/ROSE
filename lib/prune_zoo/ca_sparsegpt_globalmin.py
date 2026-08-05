import torch

from .ca_sparsegpt_slice import CASparseGPTSlice
from .slice_budget import allocate_exact_marginal_budgets


class CASparseGPTGlobalMin(CASparseGPTSlice):
    """CA-Slice with a global-mask-derived automatic lower bound.

    Columns remain in their original order.  One exact layer-wide SparseGPT
    mask is predicted before slicing.  The lowest predicted sparsity among the
    physical slices becomes the common allocation lower bound; the natural
    upper bound is one.  Exact marginal Hessian allocation then assigns the
    actual per-slice budgets, while CAROSE supplies online compensation-aware
    slice ordering and SparseGPT error compensation.
    """

    def __init__(
        self,
        layer,
        slice_size=128,
        allocation_step=0.01,
        interval=4,
        reorder_threshold=0.0,
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
        self.block_verbose = block_verbose
        self.last_predicted_mask_sparsities = None
        self.last_auto_min_sparsity = None

    def _build_groups(self, W, H_inverse, sparsity, blocksize):
        Hinv = torch.linalg.cholesky(H_inverse, upper=True)
        predicted_score = (
            W.pow(2) / torch.diag(Hinv).reshape(1, -1).pow(2)
        )
        target_k = int(round(W.numel() * sparsity))
        predicted_mask = self._bottomk_mask(predicted_score, target_k)

        score_blocks = []
        widths = []
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
                "CASparseGPTGlobalMin "
                f"target={target_k / W.numel():.6f} "
                "predicted_range="
                f"[{min(predicted_sparsities):.6f}, "
                f"{max(predicted_sparsities):.6f}] "
                f"auto_min={auto_min:.6f} "
                "allocated_range="
                f"[{min(allocated_sparsities):.6f}, "
                f"{max(allocated_sparsities):.6f}] "
                f"slices={len(groups)}"
            )
        if self.block_verbose:
            for group_id, (predicted, budget, allocated) in enumerate(
                zip(predicted_sparsities, budgets, allocated_sparsities)
            ):
                print(
                    "CAGlobalMinBlock "
                    f"block={group_id} predicted={predicted:.6f} "
                    f"budget={budget} allocated={allocated:.6f}"
                )

        return groups, target_k
