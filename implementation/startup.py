"""Portable configuration loading and fail-fast single-node checks."""
from pathlib import Path
import math

import numpy as np
import yaml


def validate_router_aux_loss_coef(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("router_aux_loss_coef must be finite and nonnegative")


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    for section, key in [("data", "train_dataset_path"), ("general", "checkpoint_folder"),
                         ("train", "load_checkpoint_path")]:
        value = config[section].get(key)
        if value and not Path(value).is_absolute():
            config[section][key] = str((path.parent / value).resolve())
    return config


def validate_config(config, *, check_data=False, check_cuda=False):
    general, train, model, data = (config[key] for key in ("general", "train", "model", "data"))
    validate_router_aux_loss_coef(train.get("router_aux_loss_coef", 0.01))
    for section, keys in [(general, ("n_workers", "gpu_per_node")),
                          (train, ("data_parallel_num", "pipeline_parallel_stages", "microbatch_num",
                                   "batch_size", "save_interval", "print_loss_interval", "target_iteration")),
                          (model, ("num_head", "d_model", "d_ff", "context_len", "num_layers", "num_expert_per_node")),
                          (data, ("vocab_size",))]:
        for key in keys:
            if type(section[key]) is not int or section[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
    dp = train["data_parallel_num"]
    if general["device"] != "gpu":
        raise ValueError("Training requires CUDA/NCCL; general.device must be gpu")
    if general["n_workers"] != general["gpu_per_node"]:
        raise ValueError("The bundled launchers support one node only")
    if dp * train["pipeline_parallel_stages"] != general["n_workers"]:
        raise ValueError("data_parallel_num * pipeline_parallel_stages must equal n_workers")
    if train["batch_size"] % (dp * train["microbatch_num"]):
        raise ValueError("batch_size must be divisible by data_parallel_num * microbatch_num")
    if not 0 <= general["info_src"] < general["n_workers"]:
        raise ValueError("info_src must identify a worker")
    if model["d_model"] % model["num_head"]:
        raise ValueError("d_model must be divisible by num_head")
    head_dim = model["d_model"] // model["num_head"]
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError("Triton attention requires a power-of-two head dimension >= 16")
    if model["context_len"] % 16:
        raise ValueError("Use context_len divisible by the attention tile size (16)")
    if model["num_expert_per_node"] * dp < 2:
        raise ValueError("Top-2 MoE routing requires at least two experts")
    if any(value % dp for value in (data["vocab_size"], model["d_model"], model["d_ff"])):
        raise ValueError("vocab_size, d_model and d_ff must be divisible by data_parallel_num")
    scheduler = train["learning_rate_scheduler"]
    if not 0 <= scheduler["t_w"] < scheduler["t_c"]:
        raise ValueError("Scheduler requires 0 <= t_w < t_c")
    if check_data:
        from implementation.train_utils import load_dataset
        tokens = load_dataset(data["train_dataset_path"], vocab_size=data["vocab_size"], validate=True)
        if tokens.size <= model["context_len"]:
            raise ValueError("Dataset must contain more tokens than context_len")
    if check_cuda:
        import torch
        if not torch.distributed.is_nccl_available() or torch.cuda.device_count() < general["gpu_per_node"]:
            raise ValueError(f"Requires NCCL and {general['gpu_per_node']} visible CUDA GPUs")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("MoE computation requires a BF16-capable GPU")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate configuration or generate synthetic smoke-test tokens")
    parser.add_argument("--config", required=True)
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--check-cuda", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    if args.generate_data:
        output = Path(config["data"]["train_dataset_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(config["data"]["seed"])
        tokens = rng.integers(0, config["data"]["vocab_size"], size=max(4096, config["model"]["context_len"] * 4), dtype=np.int32)
        with output.open("xb") as target:
            np.save(target, tokens)
        print(f"Created synthetic test tokens: {output}")
    validate_config(config, check_data=True, check_cuda=args.check_cuda)
    print("Startup checks passed")