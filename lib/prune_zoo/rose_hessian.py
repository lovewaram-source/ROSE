import time

import torch
import torch.nn as nn
import transformers

from .rose import ROSE


class ROSEHessian(ROSE):
    """ROSE with SparseGPT Hessian loss used for column/block ordering."""

    def __init__(
        self,
        layer,
        blocksize=128,
        reorder_threshold=0.5,
        verbose=False,
    ):
        super().__init__(layer)
        self.blocksize = blocksize
        self.reorder_threshold = reorder_threshold
        self.verbose = verbose
        self.last_relative_range = None
        self.last_reordered = None

    def _calculate_reordering(self, score, sparsity, blocksize):
        column_orders = []
        block_losses = []

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            block_score = score[:, i1:i2]
            target_k = int(round(block_score.numel() * sparsity))
            target_k = max(0, min(target_k, block_score.numel()))

            mask = torch.zeros_like(block_score, dtype=torch.bool)
            if target_k == block_score.numel():
                mask.fill_(True)
            elif target_k > 0:
                prune_indices = torch.topk(
                    block_score.flatten(),
                    k=target_k,
                    largest=False,
                    sorted=False,
                ).indices
                mask.view(-1)[prune_indices] = True

            predicted_loss = block_score * mask
            column_loss = predicted_loss.sum(dim=0)
            local_order = torch.argsort(column_loss, descending=True)
            column_orders.append(
                torch.arange(i1, i2, device=score.device)[local_order]
            )
            block_losses.append(predicted_loss.sum())

        block_losses = torch.stack(block_losses)
        max_loss = torch.max(block_losses)
        if max_loss > 0:
            relative_range = (
                (torch.max(block_losses) - torch.min(block_losses)) / max_loss
            ).item()
        else:
            relative_range = 0.0

        should_reorder = relative_range > self.reorder_threshold
        if should_reorder:
            block_order = torch.argsort(block_losses, descending=True).cpu().tolist()
            reordered_indices = torch.cat(
                [column_orders[block_index] for block_index in block_order]
            )
        else:
            reordered_indices = torch.arange(self.columns, device=score.device)

        return reordered_indices, relative_range, should_reorder

    def _hessian_compensation(self, sparsity, W, Hinv, blocksize):
        losses = torch.zeros(self.rows, device=self.dev)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            score = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2)
            threshold = torch.sort(score.flatten())[0][
                int(score.numel() * sparsity)
            ]
            mask1 = score <= threshold

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q).pow(2) / d.pow(2)

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(
                    Hinv1[i, i:].unsqueeze(0)
                )
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            losses += torch.sum(Losses1, dim=1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        return W

    def fasterprune(
        self,
        sparsity,
        prune_n=0,
        prune_m=0,
        blocksize=None,
        percdamp=0.01,
    ):
        if prune_n != 0 or prune_m != 0:
            raise ValueError("ROSEHessian currently supports only unstructured sparsity")
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must satisfy 0 <= sparsity < 1")
        if self.blocksize <= 0:
            raise ValueError("rose_hessian_blocksize must be a positive integer")
        if not 0.0 <= self.reorder_threshold <= 1.0:
            raise ValueError(
                "rose_hessian_reorder_threshold must satisfy 0 <= value <= 1"
            )

        blocksize = self.blocksize if blocksize is None else blocksize
        tick = time.perf_counter()

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H.clone()
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H_cholesky = torch.linalg.cholesky(H)
        H_inverse = torch.cholesky_inverse(H_cholesky)

        # The upper Cholesky factor matches SparseGPT's pruning-loss metric.
        Hinv_for_score = torch.linalg.cholesky(H_inverse, upper=True)
        score = W.pow(2) / torch.diag(Hinv_for_score).reshape(1, -1).pow(2)

        reordered_indices, relative_range, reordered = self._calculate_reordering(
            score,
            sparsity,
            blocksize,
        )
        self.last_relative_range = relative_range
        self.last_reordered = reordered

        W = W[:, reordered_indices]
        H_inverse = H_inverse[:, reordered_indices]
        H_inverse = H_inverse[reordered_indices, :]
        Hinv_reordered = torch.linalg.cholesky(H_inverse, upper=True)

        W = self._hessian_compensation(
            sparsity,
            W,
            Hinv_reordered,
            blocksize,
        )

        inverse_indices = torch.argsort(reordered_indices)
        W_restored = W[:, inverse_indices]

        if W.is_cuda:
            torch.cuda.synchronize()

        if isinstance(self.layer, transformers.Conv1D):
            W_restored = W_restored.t()
        self.layer.weight.data = W_restored.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if self.verbose:
            print(
                "ROSEHessian "
                f"relative_range={relative_range:.6f} "
                f"threshold={self.reorder_threshold:.6f} "
                f"reordered={reordered} "
                f"blocks={len(range(0, self.columns, blocksize))} "
                f"time={time.perf_counter() - tick:.2f}s"
            )
