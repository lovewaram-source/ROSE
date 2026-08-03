import argparse
import os
import re
import time
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

from lib.prune import prune_model
from lib.eval import eval_ppl,eval_zero_shot
from lib.utils import check_sparsity, distribute_model

# from smilelogging import Logger  
# from smilelogging import argparser as parser

def auto_or_int(value):
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Must be 'auto' or an integer, got '{value}'") 

def get_llm(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto", 
        low_cpu_mem_usage=True, 
        device_map="cpu"   
    )
    model.seqlen = 2048
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_path, use_fast=False,unk_token="<unk>")
    tokenizer.pad_token = tokenizer.eos_token    
    return model, tokenizer


PPL_HEADER = ["Dataset", "Model", "Sparsity", "Method", "PPL", "PruneTime(s)"]
LEGACY_PPL_ROW = re.compile(
    r"^\s*(wikitext2|c4|ptb)(.+?)\s+(\d+(?:\.\d+)?%)\s*"
    r"(\S+?)\s+(\d+(?:\.\d+)?)\s*(N/A|\d+(?:\.\d+)?)?\s*$"
)


def _read_ppl_rows(filename):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return []

    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if lines and "|" in lines[0]:
        rows = []
        for line in lines[2:]:
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split("|")]
            if len(fields) != len(PPL_HEADER):
                raise ValueError(f"Cannot parse PPL result row: {line}")
            rows.append(fields)
        return rows

    rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        match = LEGACY_PPL_ROW.match(line)
        if match is None:
            raise ValueError(f"Cannot migrate legacy PPL result row: {line}")
        dataset, model, sparsity, method, ppl, prune_time = match.groups()
        rows.append([dataset, model.strip(), sparsity, method, ppl, prune_time or "N/A"])
    return rows


def append_ppl_result(filename, data_items):
    """Append a row and rewrite the PPL result table with compact aligned columns."""
    rows = _read_ppl_rows(filename)
    rows.append([str(item) for item in data_items])

    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(PPL_HEADER)
    ]

    def format_row(row):
        return " | ".join(
            f"{value:<{width}}" if index < 4 else f"{value:>{width}}"
            for index, (value, width) in enumerate(zip(row, widths))
        )

    header_line = format_row(PPL_HEADER)
    divider_line = "-+-".join("-" * width for width in widths)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(header_line + "\n")
        f.write(divider_line + "\n")
        for row in rows:
            f.write(format_row(row) + "\n")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default="/home/sumingluo/models/llama2-7b", help="Path to the pretrained model directory.")
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples used for pruning.')
    
    parser.add_argument('--sparsity_ratio', type=float, default=0.7, help='Target sparsity ratio.')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"], help='Type of sparsity pattern: unstructured or structured')
    parser.add_argument("--prune_method", type=str, default="sparsegpt", choices=["magnitude", "wanda", "sparsegpt", "sparsegpt_slice", "sparsegpt_slice_reorder_total", "sparsegpt_slice_reorder_mean", "rose_slice", "dsnot", "rose", "rose_dynamic", "ca_rose", "rose_bottomk", "rose_hessian", "dense"], help="Pruning method to apply.")
    parser.add_argument("--slice_size", type=int, default=128, help="Number of consecutive input columns in each SparseGPTSlice slice.")
    parser.add_argument("--slice_min_ratio", type=float, default=None, help="Minimum sparsity of each slice. Defaults to target sparsity minus 0.15.")
    parser.add_argument("--slice_max_ratio", type=float, default=None, help="Maximum sparsity of each slice. Defaults to target sparsity plus 0.15.")
    parser.add_argument("--slice_step_ratio", type=float, default=0.01, help="Budget allocation step as a fraction of each slice.")
    parser.add_argument("--slice_verbose", action="store_true", help="Print the allocated SparseGPTSlice sparsity range for every pruned sublayer.")
    parser.add_argument("--slice_reorder_threshold", type=float, default=0.5, help="Minimum relative Wanda slice-priority range required to activate slice reordering.")
    parser.add_argument("--slice_reorder_verbose", action="store_true", help="Print SparseGPTSliceReorder allocation and ordering statistics for every pruned sublayer.")
    parser.add_argument("--rose_hessian_blocksize", type=int, default=128, help="Column block size used by ROSEHessian ordering and compensation.")
    parser.add_argument("--rose_hessian_reorder_threshold", type=float, default=0.5, help="Minimum relative block-loss range required to activate ROSEHessian reordering.")
    parser.add_argument("--rose_hessian_verbose", action="store_true", help="Print ROSEHessian ordering statistics for every pruned sublayer.")
    parser.add_argument("--rose_bottomk_reorder_threshold", type=float, default=0.5, help="Minimum relative Wanda block-loss range required to activate ROSEBottomK reordering.")
    parser.add_argument("--rose_bottomk_verbose", action="store_true", help="Print ROSEBottomK ordering statistics for every pruned sublayer.")
    parser.add_argument("--rose_dynamic_blocksize", type=int, default=128, help="Number of consecutive input columns in each ROSEDynamic block.")
    parser.add_argument("--rose_dynamic_interval", type=int, default=4, help="Number of blocks pruned before ROSEDynamic re-ranks the remaining blocks.")
    parser.add_argument("--rose_dynamic_verbose", action="store_true", help="Print ROSEDynamic round and timing statistics.")
    parser.add_argument("--ca_rose_blocksize", type=int, default=128, help="Number of consecutive input columns in each CA-ROSE block.")
    parser.add_argument("--ca_rose_interval", type=int, default=4, help="Number of blocks processed before CA-ROSE recomputes residual-loss priorities.")
    parser.add_argument("--ca_rose_verbose", action="store_true", help="Print CA-ROSE residual-loss, sparsity, and timing statistics.")

    parser.add_argument("--calibration_dataset", type=str, default="c4", choices=["c4", "wikitext2", "ptb"], help="Dataset used to collect pruning calibration activations.")
    parser.add_argument("--eval_dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"], help="Dataset used for perplexity evaluation.")
    parser.add_argument("--dataset_cache_dir", type=str, default="", help="Hugging Face datasets cache directory.")
    parser.add_argument("--offline_dataset", action="store_true", help="Forbid dataset downloads and use only local files/cache.")
    parser.add_argument("--c4_train_path", type=str, default="", help="Local C4 train load_from_disk directory or Arrow/Parquet/JSON file.")
    parser.add_argument("--c4_validation_path", type=str, default="", help="Local C4 validation load_from_disk directory or Arrow/Parquet/JSON file.")
    parser.add_argument("--wikitext_train_path", type=str, default="", help="Local WikiText2 train load_from_disk directory or Arrow/Parquet/JSON file.")
    parser.add_argument("--wikitext_test_path", type=str, default="", help="Local WikiText2 test load_from_disk directory or Arrow/Parquet/JSON file.")
    
    parser.add_argument("--tasks", type=str, nargs="+", default=["winogrande","boolq","piqa","openbookqa","hellaswag","arc_easy","arc_challenge"], help="List of evaluation tasks.")
    parser.add_argument("--eval_zero_shot", action="store_true", help="Enable zero-shot evaluation mode.")
    parser.add_argument("--lm_eval_batch_size",type=auto_or_int,default="auto",help="LM eval batch size to evaluate")
    
    parser.add_argument('--save_model', type=str, default="", help='Path to save the pruned model. If empty, model will not be saved.')
    parser.add_argument("--distribute",action="store_true",help="Distribute the model on multiple GPUs for evaluation.")

    args = parser.parse_args()
    # logger = Logger(args, overwrite_print=True)  

    if not 0.0 <= args.sparsity_ratio < 1.0:
        parser.error("--sparsity_ratio must satisfy 0 <= value < 1")
    if args.prune_method in ["sparsegpt_slice", "rose_slice", "sparsegpt_slice_reorder_total", "sparsegpt_slice_reorder_mean"]:
        if args.sparsity_type != "unstructured":
            parser.error(f"{args.prune_method} currently supports only unstructured sparsity")
        if args.slice_size <= 0:
            parser.error("--slice_size must be a positive integer")
        if not 0.0 < args.slice_step_ratio <= 1.0:
            parser.error("--slice_step_ratio must satisfy 0 < value <= 1")

        effective_min = (
            max(0.0, args.sparsity_ratio - 0.15)
            if args.slice_min_ratio is None
            else args.slice_min_ratio
        )
        effective_max = (
            min(1.0, args.sparsity_ratio + 0.15)
            if args.slice_max_ratio is None
            else args.slice_max_ratio
        )
        if not 0.0 <= effective_min <= args.sparsity_ratio:
            parser.error("--slice_min_ratio must be between 0 and --sparsity_ratio")
        if not args.sparsity_ratio <= effective_max <= 1.0:
            parser.error("--slice_max_ratio must be between --sparsity_ratio and 1")
        if args.prune_method in [
            "sparsegpt_slice_reorder_total",
            "sparsegpt_slice_reorder_mean",
        ] and not 0.0 <= args.slice_reorder_threshold <= 1.0:
            parser.error(
                "--slice_reorder_threshold must satisfy 0 <= value <= 1"
            )
    if args.prune_method == "rose_hessian":
        if args.sparsity_type != "unstructured":
            parser.error("rose_hessian currently supports only unstructured sparsity")
        if args.rose_hessian_blocksize <= 0:
            parser.error("--rose_hessian_blocksize must be a positive integer")
        if not 0.0 <= args.rose_hessian_reorder_threshold <= 1.0:
            parser.error(
                "--rose_hessian_reorder_threshold must satisfy 0 <= value <= 1"
            )
    if args.prune_method == "rose_bottomk":
        if not 0.0 <= args.rose_bottomk_reorder_threshold <= 1.0:
            parser.error(
                "--rose_bottomk_reorder_threshold must satisfy 0 <= value <= 1"
            )
    if args.prune_method == "rose_dynamic":
        if args.sparsity_type != "unstructured":
            parser.error("rose_dynamic currently supports only unstructured sparsity")
        if args.rose_dynamic_blocksize <= 0:
            parser.error("--rose_dynamic_blocksize must be a positive integer")
        if args.rose_dynamic_interval <= 0:
            parser.error("--rose_dynamic_interval must be a positive integer")
    if args.prune_method == "ca_rose":
        if args.sparsity_type != "unstructured":
            parser.error("ca_rose currently supports only unstructured sparsity")
        if args.ca_rose_blocksize <= 0:
            parser.error("--ca_rose_blocksize must be a positive integer")
        if args.ca_rose_interval <= 0:
            parser.error("--ca_rose_interval must be a positive integer")

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model_path.split("/")[-1]    
    print(f"loading llm model {model_name}")

    model,tokenizer = get_llm(args.model_path)
    model.eval()    
    device = torch.device("cuda")
    prune_seconds = None

    if args.prune_method != "dense" and args.sparsity_ratio > 0:
        print("pruning starts")
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        prune_start = time.perf_counter()
        prune_model(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        prune_seconds = time.perf_counter() - prune_start
        print(f"total pruning time: {prune_seconds:.2f}s ({prune_seconds / 60:.2f}min)")
    else:
        pass
    print("*"*30)
     
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    
    if args.distribute:
        distribute_model(model)
    else:
        model.to(device)
    
    # =======================
    # PPL Evaluation
    # =======================
    os.makedirs("results/ppl", exist_ok=True)
    ppl_filename = f"results/ppl/{model_name}.txt"
    dataset = args.eval_dataset
    ppl_wikitext = eval_ppl(model, tokenizer, dataset, args=args)

    prune_time_text = f"{prune_seconds:.2f}" if prune_seconds is not None else "N/A"
    ppl_data_items = [
        dataset,
        model_name,
        f"{args.sparsity_ratio:.1%}",
        args.prune_method,
        f"{ppl_wikitext:.4f}",
        prune_time_text,
    ]
    append_ppl_result(ppl_filename, ppl_data_items)

    # =======================
    # Zero-shot Evaluation
    # =======================
    if args.eval_zero_shot:
        os.makedirs("results/acc", exist_ok=True)
        acc_filename = f"results/acc/{model_name}.txt"

        metric_vals = eval_zero_shot(model, tokenizer, args)
        metric_keys = list(metric_vals.keys())  

        col_width = 15
        header_items = ["Model", "Sparsity", "Method"] + metric_keys
        header_line = "".join(f"{item:>{col_width}}" for item in header_items)

        values = [f"{100 * metric_vals[k]:.2f}" for k in metric_keys]
        data_items = [model_name, f"{args.sparsity_ratio:.1%}", args.prune_method] + values
        data_line = "".join(f"{item:>{col_width}}" for item in data_items)

        with open(acc_filename, 'a') as f:
            if not os.path.exists(acc_filename) or os.path.getsize(acc_filename) == 0:
                f.write(header_line + "\n")
                f.write("-" * len(header_line) + "\n")
            f.write(data_line + "\n")
            
if __name__ == '__main__':
    main()
