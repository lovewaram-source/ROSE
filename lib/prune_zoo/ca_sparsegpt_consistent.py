import torch

from .ca_sparsegpt_slice import CASparseGPTSlice


class CASparseGPTConsistent(CASparseGPTSlice):
    """CA-Slice whose ranking candidate mask matches SparseGPT's metric.

    The original CA-Slice uses a Wanda candidate mask to estimate residual
    loss, but the execution stage recomputes a SparseGPT Hessian mask.  This
    class uses the local SparseGPT score for the CA candidate as well.  It
    retains physical column order inside each slice so the selected first
    slice has the same local coordinate system at ranking and execution time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_candidate_metric = "sparsegpt"

    def _rank_compensation_aware(self, W, groups, H_inverse):
        priorities = []
        local_column_orders = []
        offset = 0

        for group in groups:
            width = group["width"]
            i1 = offset
            i2 = i1 + width
            indices = group["indices"]
            inverse_block = H_inverse[i1:i2, i1:i2]
            inverse_block = (inverse_block + inverse_block.t()) / 2

            # This is the same local Hessian score used by SparseGPT when a
            # slice is actually committed.  In particular, it replaces the
            # Wanda score used by CAROSE for its candidate mask.
            inverse_factor = torch.linalg.cholesky(inverse_block, upper=True)
            sparsegpt_score = (
                W[:, indices].pow(2)
                / torch.diag(inverse_factor).reshape(1, -1).pow(2)
            )
            candidate_mask = self._bottomk_mask(
                sparsegpt_score, group["budget"]
            )
            candidate_error = W[:, indices] * candidate_mask
            residual_curvature = torch.cholesky_inverse(inverse_factor)

            weighted_error = candidate_error.matmul(residual_curvature)
            element_contribution = candidate_error * weighted_error / 2
            priorities.append(element_contribution.sum())

            # Do not reorder columns within a physical slice.  This preserves
            # the local Hessian coordinate system used to form candidate_mask.
            local_column_orders.append(
                torch.arange(width, device=self.dev)
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
        ranking = (
            torch.argsort(priorities, descending=True).cpu().tolist()
            if reorder
            else list(range(len(groups)))
        )
        return ranking, local_column_orders, priorities
