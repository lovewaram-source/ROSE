import torch

from .ca_rose import CAROSE


class LookaheadROSE(CAROSE):
    """Greedy one-step block lookahead over top Wanda candidates."""

    def __init__(self, layer, candidate_count=3, **kwargs):
        super().__init__(layer, interval=1, **kwargs)
        if candidate_count <= 0:
            raise ValueError("lookahead_candidates must be positive")
        self.candidate_count = candidate_count

    def _rank_compensation_aware(self, W, groups, H_inverse):
        wanda_priorities = []
        group_positions = []
        column_orders = []
        current_losses = []
        current_columns = torch.cat([group["indices"] for group in groups])
        offset = 0
        for group in groups:
            width = group["width"]
            indices = group["indices"]
            wanda = torch.abs(W[:, indices]) * torch.sqrt(
                self.scaler_row[indices].reshape(1, -1)
            )
            wanda_mask = self._bottomk_mask(wanda, group["budget"])
            wanda_priorities.append((wanda * wanda_mask).sum())
            positions = torch.arange(offset, offset + width, device=self.dev)
            group_positions.append(positions)

            inverse_block = H_inverse[offset : offset + width, offset : offset + width]
            inverse_block = (inverse_block + inverse_block.t()) / 2
            factor = torch.linalg.cholesky(inverse_block, upper=True)
            score = W[:, indices].pow(2) / torch.diag(factor).reshape(1, -1).pow(2)
            mask = self._bottomk_mask(score, group["budget"])
            column_orders.append(torch.argsort((score * mask).sum(dim=0), descending=True))
            current_losses.append((score * mask).sum() / 2)
            offset += width

        wanda_priorities = torch.stack(wanda_priorities)
        candidate_positions = torch.topk(
            wanda_priorities,
            k=min(self.candidate_count, len(groups)),
            largest=True,
            sorted=False,
        ).indices.tolist()

        priorities = torch.full_like(wanda_priorities, float("-inf"))
        for position in candidate_positions:
            selected_positions = group_positions[position][column_orders[position]]
            remaining_positions = [
                positions
                for other, positions in enumerate(group_positions)
                if other != position
            ]
            permutation = torch.cat([selected_positions] + remaining_positions)
            inverse_permuted = H_inverse[:, permutation][permutation, :]
            factor = torch.linalg.cholesky(inverse_permuted, upper=True)
            W_trial = W[:, current_columns[permutation]].clone()
            W_trial, _, selected_width = self._prune_selected_prefix(
                W_trial, factor, [groups[position]]
            )

            # One-step objective: estimated loss of every remaining block after
            # applying this candidate's compensation update.
            objective = current_losses[position].clone()
            trailing_diagonal = torch.diag(factor)[selected_width:]
            remaining_W = W_trial[:, selected_width:]
            offset = 0
            for other, group in enumerate(groups):
                if other == position:
                    continue
                width = group["width"]
                score = remaining_W[:, offset : offset + width].pow(2) / (
                    trailing_diagonal[offset : offset + width].reshape(1, -1).pow(2)
                )
                mask = self._bottomk_mask(score, group["budget"])
                objective += (score * mask).sum() / 2
                offset += width
            priorities[position] = -objective

        candidate_objectives = -priorities[candidate_positions]
        scale = candidate_objectives.abs().max().clamp_min(
            torch.finfo(candidate_objectives.dtype).eps
        )
        relative_range = (
            candidate_objectives.max() - candidate_objectives.min()
        ) / scale
        reordered = bool(relative_range.item() >= self.reorder_threshold)
        self.last_reorder_score = relative_range.item()
        self.last_reorder_applied = reordered
        if reordered:
            ranking = torch.argsort(priorities, descending=True).cpu().tolist()
        else:
            ranking = list(range(len(groups)))
            column_orders = [
                torch.arange(group["width"], device=self.dev) for group in groups
            ]
        reported_objectives = -priorities
        reported_objectives[~torch.isfinite(reported_objectives)] = 0
        return ranking, column_orders, reported_objectives
