import unittest
import torch

from lib.prune_zoo.ca_sparsegpt_slice import CASparseGPTSlice
from lib.prune_zoo.cluster_sparsegpt import ClusterSparseGPT
from lib.prune_zoo.dynamic_nm import DynamicNM
from lib.prune_zoo.global_budget_gpt import (
    GlobalBudgetSparseGPT,
    allocate_global_budgets,
    build_global_profile,
)
from lib.prune_zoo.lookahead_rose import LookaheadROSE
from lib.prune_zoo.low_rank_ca_rose import LowRankCAROSE
from lib.prune_zoo.online_slicegpt import OnlineSliceGPT
from lib.prune_zoo.robust_slicegpt import RobustSliceGPT


def calibrated(pruner_type, **kwargs):
    torch.manual_seed(7)
    layer = torch.nn.Linear(16, 8, bias=False)
    pruner = pruner_type(layer, **kwargs)
    for _ in range(4):
        inputs = torch.randn(2, 4, 16)
        pruner.add_batch(inputs, layer(inputs))
    return layer, pruner


class ExperimentalPrunerTests(unittest.TestCase):
    def test_unstructured_experimental_pruner_exact_budget(self):
        configurations = [
            (CASparseGPTSlice, {"slice_size": 4, "interval": 2}),
            (OnlineSliceGPT, {"slice_size": 4}),
            (
                RobustSliceGPT,
                {
                    "slice_size": 4,
                    "robust_groups": 2,
                    "uncertainty_weight": 0.5,
                },
            ),
            (ClusterSparseGPT, {"slice_size": 4}),
            (LookaheadROSE, {"blocksize": 4, "candidate_count": 2}),
            (LowRankCAROSE, {"blocksize": 4, "interval": 2, "rank": 4}),
        ]
        for pruner_type, kwargs in configurations:
            with self.subTest(pruner=pruner_type.__name__):
                layer, pruner = calibrated(pruner_type, **kwargs)
                pruner.fasterprune(0.5, blocksize=4)
                self.assertEqual(
                    int((layer.weight == 0).sum()), layer.weight.numel() // 2
                )
    def test_global_budget_is_exact_across_sublayers(self):
        first_layer, first = calibrated(GlobalBudgetSparseGPT, target_k=0)
        second_layer, second = calibrated(GlobalBudgetSparseGPT, target_k=0)
        profiles = [
            build_global_profile(first, 0.25, 0.75, 0.1),
            build_global_profile(second, 0.25, 0.75, 0.1),
        ]
        budgets, target = allocate_global_budgets(profiles, 0.5)
        self.assertEqual(sum(budgets), target)
        self.assertEqual(target, 128)

        first.target_k, second.target_k = budgets
        first.fasterprune(0.5, blocksize=4)
        second.fasterprune(0.5, blocksize=4)
        actual = int(
            (first_layer.weight == 0).sum() + (second_layer.weight == 0).sum()
        )
        self.assertEqual(actual, target)
    def test_dynamic_nm_preserves_every_physical_group(self):
        layer, pruner = calibrated(DynamicNM, blocksize=8, interval=1)
        pruner.fasterprune(0.5, prune_n=2, prune_m=4, blocksize=8)
        mask = (layer.weight == 0).reshape(8, 4, 4)
        self.assertTrue(torch.all(mask.sum(dim=2) == 2))


if __name__ == "__main__":
    unittest.main()
