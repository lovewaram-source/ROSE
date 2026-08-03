import math
import time

import torch
import torch.nn as nn
import transformers

from .rose import ROSE


class ROSEBottomK(ROSE):
    """ROSE with exact bottom-k masks and safe block reordering.

    The original ROSE code uses a score threshold (``score <= threshold``).
    That can prune more weights than requested when the threshold is tied, and
    it also selects one extra element in the common no-tie case.  This version
    selects exactly the requested number of elements wherever a mask is built.
    """

    def __init__(self, layer, reorder_threshold=0.5, verbose=False):
        super().__init__(layer)
        self.reorder_threshold = reorder_threshold
        self.verbose = verbose

    @staticmethod
    def _bottomk_mask(score, k):
        """Return a boolean mask containing exactly the k smallest scores."""
        k = max(0, min(int(k), score.numel()))
        mask = torch.zeros_like(score, dtype=torch.bool)
        if k == 0:
            return mask
        if k == score.numel():
            mask.fill_(True)
            return mask

        indices = torch.topk(
            score.reshape(-1), k=k, largest=False, sorted=False
        ).indices
        mask.reshape(-1)[indices] = True
        return mask

    def _wanda_block_statistics(self, score, blocksize, sparsity):
        """Compute exact-bottom-k Wanda losses and the within-block order."""
        block_losses = []
        column_orders = []

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            block_score = score[:, i1:i2].float()
            mask = self._bottomk_mask(block_score, int(block_score.numel() * sparsity))
            selected_score = block_score * mask

            block_losses.append(selected_score.sum())
            local_order = torch.argsort(selected_score.sum(dim=0), descending=True)
            column_orders.append(torch.arange(i1, i2, device=score.device)[local_order])

        return torch.stack(block_losses), torch.cat(column_orders)

    def _nm_block_losses(self, score, prune_n, prune_m):
        """Predicted Wanda loss for each N:M group, using N smallest scores."""
        losses = []
        for i1 in range(0, self.columns, prune_m):
            i2 = min(i1 + prune_m, self.columns)
            group_score = score[:, i1:i2].float()
            n = min(prune_n, group_score.shape[1])
            losses.append(
                torch.topk(group_score, k=n, dim=1, largest=False, sorted=False).values.sum()
            )
        return torch.stack(losses)

    def _reorder_blocks(self, block_losses, blocksize):
        block_order = torch.argsort(block_losses, descending=True)
        return torch.cat(
            [
                torch.arange(
                    block_index.item() * blocksize,
                    min((block_index.item() + 1) * blocksize, self.columns),
                    device=self.dev,
                )
                for block_index in block_order
            ]
        )

    def _hessian_compensation(self, sparsity, W, Hinv, blocksize, prune_n, prune_m):
        losses = torch.zeros(self.rows, device=self.dev)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0:
                score = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2)
                mask1 = self._bottomk_mask(score, int(score.numel() * sparsity))
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)
                for local_i1 in range(0, count, prune_m):
                    local_i2 = min(local_i1 + prune_m, count)
                    group_score = W1[:, local_i1:local_i2].pow(2) / torch.diag(
                        Hinv1
                    )[local_i1:local_i2].reshape(1, -1).pow(2)
                    n = min(prune_n, group_score.shape[1])
                    indices = torch.topk(
                        group_score, k=n, dim=1, largest=False, sorted=False
                    ).indices
                    mask1.scatter_(1, local_i1 + indices, True)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                losses += (w - q).pow(2) / (2 * d.pow(2))

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        return W

    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=0.01
    ):
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must satisfy 0 <= sparsity < 1")
        if not 0.0 <= self.reorder_threshold <= 1.0:
            raise ValueError("reorder_threshold must satisfy 0 <= value <= 1")
        if prune_n != 0 and (prune_m <= 0 or prune_n > prune_m):
            raise ValueError("N:M pruning requires 0 < prune_n <= prune_m")

        tick = time.perf_counter()
        if prune_n != 0:
            blocksize = prune_m

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        wanda_score = torch.abs(W) * torch.sqrt(self.scaler_row.reshape(1, -1))
        if prune_n == 0:
            block_losses, column_order = self._wanda_block_statistics(
                wanda_score, blocksize, sparsity
            )
        else:
            block_losses = self._nm_block_losses(wanda_score, prune_n, prune_m)
            # Do not reorder columns inside an N:M group, otherwise the pattern changes.
            column_order = torch.arange(self.columns, device=self.dev)

        max_loss = block_losses.max().abs().clamp_min(1e-12)
        relative_range = ((block_losses.max() - block_losses.min()) / max_loss).item()
        reordered = relative_range > self.reorder_threshold
        if reordered:
            block_order = self._reorder_blocks(block_losses, blocksize)
            reordered_indices = column_order[block_order]
        else:
            reordered_indices = torch.arange(self.columns, device=self.dev)

        H = self.H.clone()
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H_inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))

        W = W[:, reordered_indices]
        H_inverse = H_inverse[:, reordered_indices][reordered_indices, :]
        Hinv = torch.linalg.cholesky(H_inverse, upper=True)
        W = self._hessian_compensation(
            sparsity, W, Hinv, blocksize, prune_n, prune_m
        )

        W_restored = W[:, torch.argsort(reordered_indices)]
        if isinstance(self.layer, transformers.Conv1D):
            W_restored = W_restored.t()
        self.layer.weight.data = W_restored.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if W.is_cuda:
            torch.cuda.synchronize()
        if self.verbose:
            print(
                "ROSEBottomK "
                f"relative_range={relative_range:.6f} "
                f"threshold={self.reorder_threshold:.6f} "
                f"reordered={reordered} "
                f"time={time.perf_counter() - tick:.2f}s"
            )
