import unittest
import torch

from lib.prune_zoo.ca_sparsegpt_slice import CASparseGPTSlice
from lib.prune_zoo.ca_sparsegpt_consistent import CASparseGPTConsistent
from lib.prune_zoo.ca_sparsegpt_globalmin import CASparseGPTGlobalMin
from lib.prune_zoo.ca_sparsegpt_allca import CASparseGPTAllCA
from lib.prune_zoo.ca_rose import CAROSE
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
from lib.prune_zoo.sparsegpt_globalmask_reorder import SparseGPTGlobalMaskReorder
from lib.prune_zoo.sparsegpt_globalmask_dynamic import SparseGPTGlobalMaskDynamic
from lib.prune_zoo.rose_dynamic import ROSEDynamic


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
            (CASparseGPTConsistent, {"slice_size": 4, "interval": 2}),
            (CASparseGPTGlobalMin, {"slice_size": 4, "interval": 2}),
            (
                CASparseGPTAllCA,
                {"slice_size": 4, "interval": 2, "greedy_steps": 2},
            ),
            (
                CAROSE,
                {"blocksize": 4, "interval": 2, "reorder_threshold": 0.25},
            ),
            (OnlineSliceGPT, {"slice_size": 4}),
            (
                RobustSliceGPT,
                {
                    "slice_size": 4,
                    "robust_groups": 2,
                    "uncertainty_weight": 0.5,
                },
            ),
            (
                ClusterSparseGPT,
                {"slice_size": 4, "cluster_threshold": 0.25},
            ),
            (
                ROSEDynamic,
                {"blocksize": 4, "interval": 2, "reorder_threshold": 0.25},
            ),
            (
                LookaheadROSE,
                {
                    "blocksize": 4,
                    "candidate_count": 2,
                    "reorder_threshold": 0.25,
                },
            ),
            (LowRankCAROSE, {"blocksize": 4, "interval": 2, "rank": 4}),
            (SparseGPTGlobalMaskReorder, {"slice_size": 4}),
            (SparseGPTGlobalMaskDynamic, {"slice_size": 4}),
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

    def test_global_mask_reorder_derives_exact_block_budgets(self):
        layer, pruner = calibrated(SparseGPTGlobalMaskReorder, slice_size=4)
        pruner.fasterprune(0.5, blocksize=4)
        self.assertEqual(sum(pruner.last_block_budgets), layer.weight.numel() // 2)
        self.assertEqual(len(pruner.last_column_permutation), layer.in_features)
        self.assertEqual(int((layer.weight == 0).sum()), layer.weight.numel() // 2)

    def test_ca_globalmin_uses_physical_slices_and_exact_budget(self):
        layer, pruner = calibrated(
            CASparseGPTGlobalMin, slice_size=4, interval=2
        )
        pruner.fasterprune(0.5, blocksize=4)
        self.assertEqual(
            pruner.last_auto_min_sparsity,
            min(pruner.last_predicted_mask_sparsities),
        )
        self.assertEqual(sum(pruner.last_slice_budgets), layer.weight.numel() // 2)
        self.assertTrue(
            all(
                ratio >= pruner.last_auto_min_sparsity
                for ratio in pruner.last_slice_sparsities
            )
        )
        self.assertEqual(int((layer.weight == 0).sum()), layer.weight.numel() // 2)

    def test_consistent_ca_uses_sparsegpt_candidate_metric(self):
        layer, pruner = calibrated(
            CASparseGPTConsistent, slice_size=4, interval=1
        )
        pruner.fasterprune(0.5, blocksize=4)
        self.assertEqual(pruner.last_candidate_metric, "sparsegpt")
        self.assertEqual(int((layer.weight == 0).sum()), layer.weight.numel() // 2)

    def test_allca_uses_one_metric_and_exact_budget(self):
        layer, pruner = calibrated(
            CASparseGPTAllCA,
            slice_size=4,
            interval=2,
            greedy_steps=2,
        )
        target = int(round(layer.weight.numel() * 0.85))
        pruner.fasterprune(0.85, blocksize=4)
        self.assertEqual(pruner.greedy_steps, 2)
        self.assertEqual(sum(pruner.last_slice_budgets), target)
        self.assertEqual(int((layer.weight == 0).sum()), target)

    def test_dynamic_global_mask_reorders_every_block(self):
        layer, pruner = calibrated(SparseGPTGlobalMaskDynamic, slice_size=4)
        pruner.fasterprune(0.5, blocksize=4)
        self.assertEqual(pruner.last_rounds, layer.in_features // 4)
        self.assertEqual(sum(pruner.last_block_budgets), layer.weight.numel() // 2)
        self.assertEqual(int((layer.weight == 0).sum()), layer.weight.numel() // 2)
    def test_dynamic_nm_preserves_every_physical_group(self):
        layer, pruner = calibrated(
            DynamicNM, blocksize=8, interval=1, reorder_threshold=0.25
        )
        pruner.fasterprune(0.5, prune_n=2, prune_m=4, blocksize=8)
        mask = (layer.weight == 0).reshape(8, 4, 4)
        self.assertTrue(torch.all(mask.sum(dim=2) == 2))


if __name__ == "__main__":
    unittest.main()
