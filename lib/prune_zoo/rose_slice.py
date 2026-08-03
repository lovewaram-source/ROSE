import torch

from .sparsegpt_slice import SparseGPTSlice


class ROSESlice(SparseGPTSlice):
    """Dynamic slice sparsity using Wanda scores for budget allocation.

    No columns or blocks are reordered.  Wanda is used only to decide how much
    each slice should be pruned; the actual masks and compensation remain the
    Hessian-based SparseGPT procedure inherited from ``SparseGPTSlice``.
    """

    def _allocation_scores(self, W, Hinv, i1, i2):
        del Hinv
        return torch.abs(W[:, i1:i2]) * torch.sqrt(
            self.scaler_row[i1:i2].reshape(1, -1)
        )
