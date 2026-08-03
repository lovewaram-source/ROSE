import torch

from .ca_rose import CAROSE
from .slice_budget import allocate_exact_marginal_budgets


class CASparseGPTSlice(CAROSE):
    """Hessian slice budgets with compensation-aware online ordering."""

    def __init__(
        self,
        layer,
        slice_size=128,
        min_sparsity=None,
        max_sparsity=None,
        allocation_step=0.01,
        interval=4,
        verbose=False,
    ):
        super().__init__(
            layer,
            blocksize=slice_size,
            interval=interval,
            verbose=verbose,
        )
        self.slice_size = slice_size
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.allocation_step = allocation_step
        self.last_slice_budgets = None
        self.last_slice_sparsities = None

    def _resolve_bounds(self, sparsity):
        minimum = (
            max(0.0, sparsity - 0.15)
            if self.min_sparsity is None
            else self.min_sparsity
        )
        maximum = (
            min(1.0, sparsity + 0.15)
            if self.max_sparsity is None
            else self.max_sparsity
        )
        if not 0.0 <= minimum <= sparsity <= maximum <= 1.0:
            raise ValueError("Invalid CA-SparseGPT-Slice sparsity bounds")
        return minimum, maximum

    def _build_groups(self, W, H_inverse, sparsity, blocksize):
        minimum, maximum = self._resolve_bounds(sparsity)
        Hinv = torch.linalg.cholesky(H_inverse, upper=True)
        score_blocks = []
        widths = []
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            widths.append(i2 - i1)
            score_blocks.append(
                W[:, i1:i2].pow(2)
                / torch.diag(Hinv)[i1:i2].reshape(1, -1).pow(2)
            )

        target_k = int(round(W.numel() * sparsity))
        budgets = allocate_exact_marginal_budgets(
            score_blocks,
            target_k,
            minimum,
            maximum,
            self.allocation_step,
        )
        sizes = [self.rows * width for width in widths]
        self.last_slice_budgets = list(budgets)
        self.last_slice_sparsities = [
            budget / size for budget, size in zip(budgets, sizes)
        ]

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
        return groups, target_k
