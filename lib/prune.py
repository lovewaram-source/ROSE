import torch
import torch.nn as nn

from .prune_zoo.wanda import Wanda
from .prune_zoo.dsnot import DSnoT
from .prune_zoo.sparsegpt import SparseGPT
from .prune_zoo.sparsegpt_slice import SparseGPTSlice
from .prune_zoo.sparsegpt_slice_reorder import SparseGPTSliceReorder
from .prune_zoo.rose_slice import ROSESlice
from .prune_zoo.rose import ROSE
from .prune_zoo.rose_dynamic import ROSEDynamic
from .prune_zoo.ca_rose import CAROSE
from .prune_zoo.ca_sparsegpt_slice import CASparseGPTSlice
from .prune_zoo.online_slicegpt import OnlineSliceGPT
from .prune_zoo.robust_slicegpt import RobustSliceGPT
from .prune_zoo.cluster_sparsegpt import ClusterSparseGPT
from .prune_zoo.lookahead_rose import LookaheadROSE
from .prune_zoo.low_rank_ca_rose import LowRankCAROSE
from .prune_zoo.dynamic_nm import DynamicNM
from .prune_zoo.global_budget_gpt import (
    GlobalBudgetSparseGPT,
    allocate_global_budgets,
    build_global_profile,
)
from .prune_zoo.rose_bottomk import ROSEBottomK
from .prune_zoo.rose_hessian import ROSEHessian
from .utils import find_layers
from .data import get_loaders


def _calibration_loader(args, tokenizer):
    return get_loaders(
        args.calibration_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        cache_dir=args.dataset_cache_dir or None,
        offline=args.offline_dataset,
        c4_train_path=args.c4_train_path or None,
        c4_validation_path=args.c4_validation_path or None,
        wikitext_train_path=args.wikitext_train_path or None,
        wikitext_test_path=args.wikitext_test_path or None,
    )[0]


def prepare_calibration_input(args, model, dataloader, device):
    """
    Collect calibration inputs for layer-wise pruning.

    Returns:
        inps: calibration inputs
        outs: placeholder outputs
        attention_mask
        position_embeddings
    """

    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False

    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise ValueError("Model must contain model.model.layers")

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)

    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device)

    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model.config, "dim", None)
        if hidden_size is None:
            raise ValueError("Cannot find hidden_size or dim in model config")

    inps = torch.zeros(
        (args.nsamples, model.seqlen, hidden_size),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False

    cache = {"i": 0, "attention_mask": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if cache["i"] < args.nsamples:
                inps[cache["i"]] = inp.detach()

            cache["i"] += 1
            if "attention_mask" in kwargs:
                cache["attention_mask"] = kwargs["attention_mask"]
            if "position_embeddings" in kwargs:
                cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError("Catcher stop forward")

    original_first_layer = layers[0]
    layers[0] = Catcher(layers[0])

    samples_collected = 0
    for batch in dataloader:
        if samples_collected >= args.nsamples:
            break
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
        samples_collected = min(cache["i"], args.nsamples)

    layers[0] = original_first_layer
    layers[0] = layers[0].to("cpu")
    model.model.embed_tokens = model.model.embed_tokens.to("cpu")
    model.model.norm = model.model.norm.to("cpu")

    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to("cpu")

    outs = torch.zeros_like(inps)
    if use_cache is not None:
        model.config.use_cache = use_cache

    torch.cuda.empty_cache()

    return inps, outs, cache["attention_mask"], cache["position_embeddings"]


@torch.no_grad()
def _profile_global_budget(
    args,
    model,
    inps,
    outs,
    attention_mask,
    position_embeddings,
    device,
):
    """Run a dense calibration pass and assign exact model-wide budgets."""
    profiles = []
    keys = []
    layers = model.model.layers
    minimum = (
        max(0.0, args.sparsity_ratio - 0.15)
        if args.global_min_ratio is None
        else args.global_min_ratio
    )
    maximum = (
        min(1.0, args.sparsity_ratio + 0.15)
        if args.global_max_ratio is None
        else args.global_max_ratio
    )

    for layer_index in range(len(layers)):
        layer = layers[layer_index].to(device)
        subset = find_layers(layer)
        wrappers = {name: SparseGPT(module) for name, module in subset.items()}

        def add_batch(name):
            def hook(_, inp, out):
                wrappers[name].add_batch(inp[0].data, out.data)

            return hook

        handles = [
            subset[name].register_forward_hook(add_batch(name)) for name in subset
        ]
        for sample in range(args.nsamples):
            outs[sample] = layer(
                inps[sample].unsqueeze(0).to(device),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )[0]
        for handle in handles:
            handle.remove()

        for name, wrapper in wrappers.items():
            profiles.append(
                build_global_profile(
                    wrapper,
                    minimum,
                    maximum,
                    args.global_step_ratio,
                )
            )
            keys.append((layer_index, name))
            wrapper.free()

        inps, outs = outs, inps
        layers[layer_index] = layer.to("cpu")
        del layer, wrappers
        torch.cuda.empty_cache()

    budgets, target = allocate_global_budgets(profiles, args.sparsity_ratio)
    assignment = dict(zip(keys, budgets))
    if args.global_budget_verbose:
        ratios = [
            budget / profile["size"]
            for budget, profile in zip(budgets, profiles)
        ]
        print(
            "GlobalBudgetAllocation "
            f"target={target / sum(p['size'] for p in profiles):.6f} "
            f"actual_range=[{min(ratios):.6f}, {max(ratios):.6f}] "
            f"sublayers={len(profiles)}"
        )
    return assignment


@torch.no_grad()
def prune_model(args, model, tokenizer, device=torch.device("cuda"), prune_n=0, prune_m=0):
    """
    Layer-wise pruning pipeline.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if not hasattr(model.model, "layers"):
        raise ValueError("Model must contain model.model.layers")

    layers = model.model.layers

    dataloader = _calibration_loader(args, tokenizer)
    inps, outs, attention_mask, position_embeddings = prepare_calibration_input(
        args, model, dataloader, device
    )

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if isinstance(position_embeddings, tuple):
        position_embeddings = tuple(e.to(device) for e in position_embeddings)

    global_budgets = None
    if args.prune_method == "global_budget_gpt":
        global_budgets = _profile_global_budget(
            args,
            model,
            inps,
            outs,
            attention_mask,
            position_embeddings,
            device,
        )
        # The profiling pass propagated dense hidden states through all layers.
        # Rebuild the original calibration inputs for the actual pruning pass.
        del inps, outs
        torch.cuda.empty_cache()
        dataloader = _calibration_loader(args, tokenizer)
        inps, outs, attention_mask, position_embeddings = prepare_calibration_input(
            args, model, dataloader, device
        )
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if isinstance(position_embeddings, tuple):
            position_embeddings = tuple(e.to(device) for e in position_embeddings)

    if args.prune_method == "magnitude":
        prune_fn = prune_magnitude
    elif args.prune_method == "wanda":
        prune_fn = prune_wanda
    elif args.prune_method in [
        "sparsegpt",
        "rose",
        "rose_bottomk",
        "global_budget_gpt",
    ]:
        prune_fn = prune_sparsegpt
    elif args.prune_method in [
        "sparsegpt_slice",
        "rose_slice",
        "sparsegpt_slice_reorder_total",
        "sparsegpt_slice_reorder_mean",
        "ca_sparsegpt_slice",
        "online_slicegpt",
        "robust_slicegpt",
        "cluster_sparsegpt",
    ]:
        prune_fn = prune_sparsegpt_slice
    elif args.prune_method == "rose_hessian":
        prune_fn = prune_rose_hessian
    elif args.prune_method == "rose_dynamic":
        prune_fn = prune_rose_dynamic
    elif args.prune_method in ["ca_rose", "lookahead_rose", "low_rank_ca_rose"]:
        prune_fn = prune_ca_rose
    elif args.prune_method == "dynamic_nm":
        prune_fn = prune_dynamic_nm
    elif args.prune_method == "dsnot":
        prune_fn = prune_dsnot
    else:
        raise ValueError(f"Unsupported prune_method: {args.prune_method}")

    for i in range(len(layers)):
        layer = layers[i].to(device)
        subset = find_layers(layer)

        wrapped_layers = {}

        for name in subset:
            if args.prune_method == "magnitude":
                wrapped_layers[name] = None
            elif args.prune_method == "wanda":
                wrapped_layers[name] = Wanda(subset[name])
            elif args.prune_method == "sparsegpt":
                wrapped_layers[name] = SparseGPT(subset[name])
            elif args.prune_method == "global_budget_gpt":
                wrapped_layers[name] = GlobalBudgetSparseGPT(
                    subset[name],
                    target_k=global_budgets[(i, name)],
                    verbose=args.global_budget_verbose,
                )
            elif args.prune_method == "sparsegpt_slice":
                wrapped_layers[name] = SparseGPTSlice(
                    subset[name],
                    slice_size=args.slice_size,
                    min_sparsity=args.slice_min_ratio,
                    max_sparsity=args.slice_max_ratio,
                    allocation_step=args.slice_step_ratio,
                    verbose=args.slice_verbose,
                )
            elif args.prune_method == "rose_slice":
                wrapped_layers[name] = ROSESlice(
                    subset[name],
                    slice_size=args.slice_size,
                    min_sparsity=args.slice_min_ratio,
                    max_sparsity=args.slice_max_ratio,
                    allocation_step=args.slice_step_ratio,
                    verbose=args.slice_verbose,
                )
            elif args.prune_method == "ca_sparsegpt_slice":
                wrapped_layers[name] = CASparseGPTSlice(
                    subset[name],
                    slice_size=args.slice_size,
                    min_sparsity=args.slice_min_ratio,
                    max_sparsity=args.slice_max_ratio,
                    allocation_step=args.slice_step_ratio,
                    interval=args.ca_slice_interval,
                    verbose=args.slice_verbose,
                )
            elif args.prune_method == "online_slicegpt":
                wrapped_layers[name] = OnlineSliceGPT(
                    subset[name],
                    slice_size=args.slice_size,
                    min_sparsity=args.slice_min_ratio,
                    max_sparsity=args.slice_max_ratio,
                    allocation_step=args.slice_step_ratio,
                    verbose=args.slice_verbose,
                )
            elif args.prune_method == "robust_slicegpt":
                wrapped_layers[name] = RobustSliceGPT(
                    subset[name],
                    slice_size=args.slice_size,
                    min_sparsity=args.slice_min_ratio,
                    max_sparsity=args.slice_max_ratio,
                    allocation_step=args.slice_step_ratio,
                    robust_groups=args.robust_groups,
                    uncertainty_weight=args.robust_uncertainty_weight,
                    verbose=args.slice_verbose,
                )
            elif args.prune_method == "cluster_sparsegpt":
                wrapped_layers[name] = ClusterSparseGPT(
                    subset[name],
                    slice_size=args.slice_size,
                    min_sparsity=args.slice_min_ratio,
                    max_sparsity=args.slice_max_ratio,
                    allocation_step=args.slice_step_ratio,
                    verbose=args.slice_verbose,
                )
            elif args.prune_method in [
                "sparsegpt_slice_reorder_total",
                "sparsegpt_slice_reorder_mean",
            ]:
                wrapped_layers[name] = SparseGPTSliceReorder(
                    subset[name],
                    reorder_mode=(
                        "total"
                        if args.prune_method.endswith("_total")
                        else "mean"
                    ),
                    slice_size=args.slice_size,
                    min_sparsity=args.slice_min_ratio,
                    max_sparsity=args.slice_max_ratio,
                    allocation_step=args.slice_step_ratio,
                    reorder_threshold=args.slice_reorder_threshold,
                    verbose=args.slice_reorder_verbose,
                )
            elif args.prune_method == "dsnot":
                wrapped_layers[name] = DSnoT(subset[name], layer_name=name)
            elif args.prune_method == "rose":
                wrapped_layers[name] = ROSE(subset[name])
            elif args.prune_method == "rose_dynamic":
                wrapped_layers[name] = ROSEDynamic(
                    subset[name],
                    blocksize=args.rose_dynamic_blocksize,
                    interval=args.rose_dynamic_interval,
                    verbose=args.rose_dynamic_verbose,
                )
            elif args.prune_method == "ca_rose":
                wrapped_layers[name] = CAROSE(
                    subset[name],
                    blocksize=args.ca_rose_blocksize,
                    interval=args.ca_rose_interval,
                    verbose=args.ca_rose_verbose,
                )
            elif args.prune_method == "lookahead_rose":
                wrapped_layers[name] = LookaheadROSE(
                    subset[name],
                    blocksize=args.ca_rose_blocksize,
                    candidate_count=args.lookahead_candidates,
                    verbose=args.ca_rose_verbose,
                )
            elif args.prune_method == "low_rank_ca_rose":
                wrapped_layers[name] = LowRankCAROSE(
                    subset[name],
                    blocksize=args.ca_rose_blocksize,
                    interval=args.ca_rose_interval,
                    rank=args.low_rank_ca_rank,
                    verbose=args.ca_rose_verbose,
                )
            elif args.prune_method == "dynamic_nm":
                wrapped_layers[name] = DynamicNM(
                    subset[name],
                    blocksize=args.dynamic_nm_blocksize,
                    interval=args.dynamic_nm_interval,
                    verbose=args.dynamic_nm_verbose,
                )
            elif args.prune_method == "rose_bottomk":
                wrapped_layers[name] = ROSEBottomK(
                    subset[name],
                    reorder_threshold=args.rose_bottomk_reorder_threshold,
                    verbose=args.rose_bottomk_verbose,
                )
            elif args.prune_method == "rose_hessian":
                wrapped_layers[name] = ROSEHessian(
                    subset[name],
                    blocksize=args.rose_hessian_blocksize,
                    reorder_threshold=args.rose_hessian_reorder_threshold,
                    verbose=args.rose_hessian_verbose,
                )
            else:
                raise ValueError("Invalid prune_method during wrapping")

        handles = []
        if args.prune_method != "magnitude":
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0).to(device),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )[0]

        for h in handles:
            h.remove()

        for name in subset:
            prune_fn(
                subset[name],
                wrapped_layers[name],
                args.sparsity_ratio,
                prune_n,
                prune_m,
            )
            print(f"Pruning layer {i} - {name}")

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0).to(device),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )[0]

        inps, outs = outs, inps
        layers[i] = layer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# =========================================================
#  Pruning Methods
# =========================================================

def prune_magnitude(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """
    magnitude pruning.
    """
    if layer is None:
        raise ValueError("Layer cannot be None")

    layer = layer.to("cuda")
    W = layer.weight.data
    W_metric = torch.abs(W)

    if prune_n != 0:
        W_mask = torch.zeros_like(W_metric) == 1
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii:(ii + prune_m)].float()
                idx = torch.topk(tmp, prune_n, dim=1, largest=True)[1]
                W_mask.scatter_(1, ii + idx, True)
    else:
        thresh = torch.sort(W_metric.flatten())[0][int(W.numel() * sparsity_ratio)]
        W_mask = W_metric > thresh

    layer.weight.data[W_mask] = 0


def prune_wanda(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """
    Wanda pruning.
    """
    if wrapped_layer is None:
        raise ValueError("wrapped_layer cannot be None for Wanda")

    W_metric = torch.abs(layer.weight.data) * torch.sqrt(wrapped_layer.scaler_row.reshape((1, -1)))
    W_mask = torch.zeros_like(W_metric) == 1

    if prune_n != 0:
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii:(ii + prune_m)].float()
                idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                W_mask.scatter_(1, ii + idx, True)
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True, descending=True)
        indices = sort_res[1][:, :int(W_metric.shape[1] * (1 - sparsity_ratio))]
        W_mask.scatter_(1, indices, True)

    layer.weight.data[W_mask] = 0
    wrapped_layer.free()


def prune_sparsegpt(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """
    SparseGPT / ROSE pruning.
    """
    if wrapped_layer is None:
        raise ValueError("wrapped_layer required for SparseGPT/ROSE")

    wrapped_layer.fasterprune(
        sparsity_ratio,
        prune_n=prune_n,
        prune_m=prune_m,
        percdamp=0.01,
        blocksize=128,
    )
    wrapped_layer.free()


def prune_dsnot(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """
    DSnoT pruning.
    """
    if wrapped_layer is None:
        raise ValueError("wrapped_layer required for DSnoT")

    wrapped_layer.fasterprune(
        sparsity=sparsity_ratio,
        prune_n=prune_n,
        prune_m=prune_m,
        max_cycle_time=50,
        update_threshold=0.1,
        pow_of_var_regrowing=1.0,
        pow_of_var_pruning=1.0,
        without_DSnoT=False,
        skip_layer="mlp",
        skip_sub_layer="no_skip",
        without_same_sign=True,
    )
    wrapped_layer.free()


def prune_sparsegpt_slice(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """SparseGPT pruning with dynamically allocated per-slice sparsity."""
    if wrapped_layer is None:
        raise ValueError("wrapped_layer required for SparseGPTSlice")

    wrapped_layer.fasterprune(
        sparsity_ratio,
        prune_n=prune_n,
        prune_m=prune_m,
        percdamp=0.01,
        blocksize=wrapped_layer.slice_size,
    )
    wrapped_layer.free()


def prune_rose_hessian(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """ROSE pruning with Hessian-aware column and block ordering."""
    if wrapped_layer is None:
        raise ValueError("wrapped_layer required for ROSEHessian")

    wrapped_layer.fasterprune(
        sparsity_ratio,
        prune_n=prune_n,
        prune_m=prune_m,
        percdamp=0.01,
        blocksize=wrapped_layer.blocksize,
    )
    wrapped_layer.free()


def prune_rose_dynamic(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """ROSE with online re-ranking of remaining blocks."""
    if wrapped_layer is None:
        raise ValueError("wrapped_layer required for ROSEDynamic")

    wrapped_layer.fasterprune(
        sparsity_ratio,
        prune_n=prune_n,
        prune_m=prune_m,
        percdamp=0.01,
        blocksize=wrapped_layer.blocksize,
    )
    wrapped_layer.free()


def prune_ca_rose(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """Compensation-aware ROSE with incremental Hessian updates."""
    if wrapped_layer is None:
        raise ValueError("wrapped_layer required for CAROSE")

    wrapped_layer.fasterprune(
        sparsity_ratio,
        prune_n=prune_n,
        prune_m=prune_m,
        percdamp=0.01,
        blocksize=wrapped_layer.blocksize,
    )
    wrapped_layer.free()


def prune_dynamic_nm(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """Strict N:M pruning with compensation-aware dynamic block ordering."""
    if wrapped_layer is None:
        raise ValueError("wrapped_layer required for DynamicNM")

    wrapped_layer.fasterprune(
        sparsity_ratio,
        prune_n=prune_n,
        prune_m=prune_m,
        percdamp=0.01,
        blocksize=wrapped_layer.blocksize,
    )
    wrapped_layer.free()
