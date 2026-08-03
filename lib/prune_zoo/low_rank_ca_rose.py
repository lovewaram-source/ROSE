import torch

from .ca_rose import CAROSE


class LowRankCAROSE(CAROSE):
    """CA-ROSE with a Nyström-plus-diagonal inverse-Hessian approximation."""

    def __init__(self, layer, rank=128, **kwargs):
        super().__init__(layer, **kwargs)
        if rank <= 0:
            raise ValueError("low_rank_ca_rank must be positive")
        self.rank = rank

    def _initial_inverse(self, H):
        size = H.shape[0]
        rank = min(self.rank, size)
        sample_indices = torch.linspace(
            0, size - 1, steps=rank, device=H.device
        ).round().long().unique()
        C = H[:, sample_indices]
        core = H[:, sample_indices][sample_indices, :]
        core_inverse = torch.cholesky_inverse(torch.linalg.cholesky(core))

        approximation_diagonal = (C.matmul(core_inverse) * C).sum(dim=1)
        residual = (torch.diag(H) - approximation_diagonal).clamp_min(
            torch.mean(torch.diag(H)) * 1e-4
        )
        inverse_residual = residual.reciprocal()
        scaled_C = inverse_residual[:, None] * C
        middle = core + C.t().matmul(scaled_C)
        middle_inverse = torch.cholesky_inverse(torch.linalg.cholesky(middle))
        inverse = torch.diag(inverse_residual) - scaled_C.matmul(
            middle_inverse
        ).matmul(scaled_C.t())
        inverse = (inverse + inverse.t()) / 2
        jitter = torch.mean(torch.diag(inverse)).abs() * 1e-6
        inverse.diagonal().add_(jitter)
        return inverse
