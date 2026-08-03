import torch
import torch.nn as nn
import transformers

from .sparsegpt_slice import SparseGPTSlice


class RobustSliceGPT(SparseGPTSlice):
    """Slice allocation using mean-plus-uncertainty Hessian scores."""

    def __init__(self, layer, robust_groups=4, uncertainty_weight=0.5, **kwargs):
        super().__init__(layer, **kwargs)
        if robust_groups <= 1:
            raise ValueError("robust_groups must be greater than one")
        self.robust_groups = robust_groups
        self.uncertainty_weight = uncertainty_weight
        self._group_hessian_sums = [
            torch.zeros_like(self.H) for _ in range(robust_groups)
        ]
        self._group_counts = [0] * robust_groups
        self._robust_batch_index = 0
        self._robust_scores = None

    def _prepare_input(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        count = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        elif isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp).permute([1, 0, 2]).flatten(1)
        return inp.float(), count

    def add_batch(self, inp, out):
        prepared, count = self._prepare_input(inp)
        group = self._robust_batch_index % self.robust_groups
        self._group_hessian_sums[group] += prepared.matmul(prepared.t())
        self._group_counts[group] += count
        self._robust_batch_index += 1
        super().add_batch(inp, out)

    def _allocation_scores(self, W, Hinv, i1, i2):
        del W, Hinv
        return self._robust_scores[:, i1:i2]

    def fasterprune(self, sparsity, **kwargs):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        group_scores = []
        for hessian_sum, count in zip(
            self._group_hessian_sums, self._group_counts
        ):
            if count == 0:
                continue
            H = 2.0 * hessian_sum / count
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += 0.01 * torch.mean(torch.diag(H))
            inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))
            factor = torch.linalg.cholesky(inverse, upper=True)
            group_scores.append(
                W.pow(2) / torch.diag(factor).reshape(1, -1).pow(2)
            )
        if len(group_scores) < 2:
            raise RuntimeError("RobustSliceGPT needs calibration data in two groups")

        stacked = torch.stack(group_scores)
        self._robust_scores = stacked.mean(dim=0) + self.uncertainty_weight * stacked.std(
            dim=0, unbiased=False
        )
        try:
            return super().fasterprune(sparsity, **kwargs)
        finally:
            self._robust_scores = None
            self._group_hessian_sums = None
