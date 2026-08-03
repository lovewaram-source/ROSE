import torch
import torch.nn as nn
import transformers

from .sparsegpt_slice import SparseGPTSlice


class ClusterSparseGPT(SparseGPTSlice):
    """SparseGPTSlice over Hessian-correlation column clusters."""

    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.last_cluster_permutation = None

    def _cluster_permutation(self, H, cluster_size):
        diagonal = torch.diag(H).clamp_min(1e-12).sqrt()
        correlation = torch.abs(H) / diagonal[:, None] / diagonal[None, :]
        available = torch.ones(self.columns, dtype=torch.bool, device=self.dev)
        clusters = []

        while available.any():
            seed = torch.nonzero(available, as_tuple=False)[0, 0]
            candidates = torch.nonzero(available, as_tuple=False).flatten()
            count = min(cluster_size, candidates.numel())
            values = correlation[seed, candidates]
            selected = candidates[
                torch.topk(values, k=count, largest=True, sorted=False).indices
            ]
            available[selected] = False
            clusters.append(selected)
        return torch.cat(clusters)

    def fasterprune(self, sparsity, blocksize=None, **kwargs):
        cluster_size = self.slice_size if blocksize is None else blocksize
        if cluster_size <= 0:
            raise ValueError("cluster_size must be positive")
        permutation = self._cluster_permutation(self.H, cluster_size)
        inverse = torch.argsort(permutation)
        self.last_cluster_permutation = permutation.detach().cpu()

        weight = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            canonical = weight.flatten(1)
            permuted = canonical[:, permutation].reshape_as(weight)
        elif isinstance(self.layer, transformers.Conv1D):
            canonical = weight.t()
            permuted = canonical[:, permutation].t()
        else:
            permuted = weight[:, permutation]

        original_scaler = self.scaler_row
        original_hessian = self.H
        original_weight = self.layer.weight.data.clone()
        succeeded = False
        self.layer.weight.data = permuted
        self.H = original_hessian[:, permutation][permutation, :]
        self.scaler_row = self.scaler_row[permutation]
        try:
            super().fasterprune(
                sparsity, blocksize=cluster_size, **kwargs
            )
            pruned = self.layer.weight.data
            if isinstance(self.layer, nn.Conv2d):
                restored = pruned.flatten(1)[:, inverse].reshape_as(pruned)
            elif isinstance(self.layer, transformers.Conv1D):
                restored = pruned.t()[:, inverse].t()
            else:
                restored = pruned[:, inverse]
            self.layer.weight.data = restored
            succeeded = True
        finally:
            if not succeeded:
                self.layer.weight.data = original_weight
                self.H = original_hessian
            self.scaler_row = original_scaler
