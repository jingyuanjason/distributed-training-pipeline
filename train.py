import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from implementation.startup import load_config, validate_config
from implementation.distributed.wrappers import FSDPWrapperPipelined
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import modal
import numpy as np

from implementation.layers import TransformerLMPipelined
from implementation.pipeline import pipelined_train_overlap
from implementation.nnfunctions import cross_entropy, learning_rate_schedule_wrapper
from implementation.optimizer import AdamW
from implementation.train_utils import get_batch, load_checkpoint, load_dataset, save_checkpoint
from datetime import datetime
import modal.experimental


app = modal.App("tiny-story-distributed-train")
BUNDLE_ROOT = Path(__file__).resolve().parent

CUDA_VERSION = "13.2.1"
PYTHON_VERSION = "3.12"

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-cudnn-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    # Required by Nsight's report importer.
    .apt_install("libdw1")
    .pip_install_from_pyproject(str(BUNDLE_ROOT / "pyproject.toml"))
    # Include the root entry point for spawned workers and Nsight subprocesses.
    .add_local_python_source("implementation", "train")
)


volume_dataset = modal.Volume.from_name("datasets")
volume_profile = modal.Volume.from_name("profile-data", create_if_missing=True)
volume_checkpoints = modal.Volume.from_name("checkpoints", create_if_missing=True)

#@modal.experimental.clustered(size=1)
@app.function(
    image=image,
    gpu="B300:8",
    volumes={"/mnt/dataset": volume_dataset, "/mnt/checkpoints": volume_checkpoints, "/mnt/profile-data": volume_profile},
    scaledown_window=10,
    timeout=60 * 60 * 12,
)
@modal.experimental.clustered(size=1)
def profile_wrapper(config):
    validate_config(config, check_data=True, check_cuda=True)
    if config["general"]["n_workers"] != 8:
        raise ValueError("This Modal deployment is configured for one node with eight GPUs")
    world_size = config.get("general").get("n_workers")
    gpu_per_node = config.get("general").get("gpu_per_node")

    cluster = modal.experimental.get_cluster_info()
    os.environ["MASTER_ADDR"] = cluster.container_ips[0]
    os.environ["MASTER_PORT"] = "29500"

    # One host ID per physical node, inherited by its eight children.
    os.environ["NCCL_HOSTID"] = f"{cluster.cluster_id}-node-{cluster.rank}"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET6"
    os.environ["NCCL_DEBUG"] = "WARN"

    profile_config = config.get("profile", {})
    if profile_config.get("enabled", False):
        nsys = shutil.which("nsys")
        if nsys is None:
            raise RuntimeError("Nsight Systems (`nsys`) is not installed in the training image. Use a CUDA image containing Nsight Systems or disable profile.enabled.")

        output_dir = profile_config.get("output_dir", "/mnt/profile-data")
        os.makedirs(output_dir, exist_ok=True)
        output = os.path.join(output_dir, f"pipeline-node-{cluster.rank}")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as config_file:
            json.dump(config, config_file)
            config_path = config_file.name

        command = _build_nsys_command(nsys, output, config_path, cluster.rank, profile_config)
        print(f"Launching Nsight Systems: {' '.join(command)}", flush=True)
        try:
            subprocess.run(command, check=True, env=os.environ.copy())
            report_path = f"{output}.nsys-rep"
            if not os.path.isfile(report_path):
                raise RuntimeError(f"Nsight Systems did not create {report_path}")
            volume_profile.commit()
            print(f"Committed Nsight report to the profile-data Volume: {report_path}", flush=True)
        finally:
            os.unlink(config_path)
    else:
        _spawn_training_workers(config, cluster.rank, world_size, gpu_per_node)


def _build_nsys_command(nsys, output, config_path, cluster_rank, profile_config):
    # Force software CUDA tracing because Modal runs under gVisor.
    trace = profile_config.get("nsight_trace", "cuda-sw,nvtx")
    return [
        nsys,
        "profile",
        f"--trace={trace}",
        "--trace-fork-before-exec=false",
        "--kill=none",
        "--cuda-event-trace=false",
        "--cuda-memory-usage=false",
        "--sample=none",
        "--cpuctxsw=none",
        "--wait=all",
        "--force-overwrite=true",
        f"--output={output}",
        sys.executable,
        os.path.abspath(__file__),
        "--profile-worker",
        config_path,
        str(cluster_rank),
    ]


def _spawn_training_workers(config, cluster_rank, world_size=None, gpu_per_node=None):
    general = config["general"]
    world_size = general["n_workers"] if world_size is None else world_size
    gpu_per_node = general["gpu_per_node"] if gpu_per_node is None else gpu_per_node
    mp.spawn(fn=_train_worker, args=(cluster_rank, world_size, config), nprocs=gpu_per_node, join=True)


def _train_worker(local_rank, cluster_rank, world_size, config):
    try:
        train(local_rank, cluster_rank, world_size, config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _nvtx_range(name):
    return torch.cuda.nvtx.range(name) if torch.cuda.is_available() else nullcontext()


def train(local_rank, cluster_rank, world_size, config):
    gpu_per_node = config.get("general").get("gpu_per_node")

    glob_rank = gpu_per_node * cluster_rank + local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://", rank=glob_rank, world_size=world_size, device_id=torch.device("cuda", local_rank))

    vocab_size = config.get("data").get("vocab_size")
    num_head = config.get("model").get("num_head")
    d_model = config.get("model").get("d_model")
    d_ff = config.get("model").get("d_ff")
    context_len = config.get("model").get("context_len")
    num_layers = config.get("model").get("num_layers")
    batch_size = config.get("train").get("batch_size")
    save_interval = config.get("train").get("save_interval")
    data_seed = config.get("data").get("seed", 42)
    validate_dataset = config.get("data").get("validate_dataset", False)
    validate_batches = config.get("data").get("validate_batches", False)
    profile_config = config.get("profile", {})
    profile_enabled = profile_config.get("enabled", False)
    profile_warmup = int(profile_config.get("nsight_warmup", 5))
    profile_iterations = int(profile_config.get("nsight_iterations", 3))
    if profile_enabled and profile_iterations < 1:
        raise ValueError("profile.nsight_iterations must be at least 1")

    train_dataset_path = config.get("data").get("train_dataset_path")
    learning_rate = config.get("train").get("learning_rate")
    device = config.get("general").get("device")
    print_loss_interval = config.get("train").get("print_loss_interval")
    optimizer_config = config.get("train").get("optimizer")
    learning_rate_scheduler_config = config.get("train").get("learning_rate_scheduler")
    pipeline_parallel_stages = config.get("train").get("pipeline_parallel_stages")
    data_parallel_num = config.get("train").get("data_parallel_num")
    microbatch_num = config.get("train").get("microbatch_num")
    num_expert_per_node = config.get("model").get("num_expert_per_node")
    fsdp_compute_dtype = torch.float16
    pipeline_dtype = torch.float16
    moe_compute_dtype = torch.bfloat16
    if device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    rank = glob_rank

    assert pipeline_parallel_stages * data_parallel_num == world_size
    pipeline_stage_this = rank // data_parallel_num
    dp_idx = rank % data_parallel_num

    dp_groups = [[start + i for i in range(data_parallel_num)] for start in range(0, world_size, data_parallel_num)]
    pp_groups = [[start + i for start in range(0, world_size, data_parallel_num)] for i in range(data_parallel_num)]
    dp_group = None
    pp_group = None
    for dp_group_ranks in dp_groups:
        dp_group_created = dist.new_group(ranks=dp_group_ranks)
        if rank in dp_group_ranks:
            dp_group = dp_group_created
    for pp_group_ranks in pp_groups:
        pp_group_created = dist.new_group(ranks=pp_group_ranks)
        if rank in pp_group_ranks:
            pp_group = pp_group_created

    checkpoint_folder = config.get("general").get("checkpoint_folder")
    info_src = config.get("general").get("info_src")
    load_checkpoint_path = config.get("train").get("load_checkpoint_path")
    target_iteration = config.get("train").get("target_iteration")

    model = TransformerLMPipelined(
        vocab_size,
        context_len,
        d_model,
        num_layers,
        num_head,
        pipeline_stage_this,
        pipeline_parallel_stages,
        num_expert_per_node=num_expert_per_node,
        d_ff=d_ff,
        dp_group=dp_group,
        moe_compute_dtype=moe_compute_dtype,
    )
    model.to(device)
    model = FSDPWrapperPipelined(model, FSDP_communication_group=dp_group, compute_dtype=fsdp_compute_dtype, moe_compute_dtype=moe_compute_dtype)
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(optimizer_config["beta1"], optimizer_config["beta2"]), weight_decay=optimizer_config["lambda"])

    now = datetime.now()
    train_start_time = now.strftime("%Y%m%d%H%M%S")
    iteration = 0
    if load_checkpoint_path:
        train_start_time, iteration = load_checkpoint_path.split("/")[-2:]
        iteration = load_checkpoint(load_checkpoint_path + f"/shaded_dp_{dp_idx}_pp_{pipeline_stage_this}.ckpt", model, optimizer)
        print(f"loaded checkpoint at {iteration} iterations, training start time is {train_start_time}")

    obj_list = [train_start_time]
    dist.broadcast_object_list(object_list=obj_list, src=info_src)
    train_start_time = obj_list[0]

    lr_scheduler = learning_rate_schedule_wrapper(
        learning_rate_scheduler_config["a_max"], learning_rate_scheduler_config["a_min"], learning_rate_scheduler_config["t_w"], learning_rate_scheduler_config["t_c"]
    )

    dist.barrier()
    train_dataset = load_dataset(train_dataset_path, vocab_size=vocab_size, validate=validate_dataset)
    data_rng = np.random.default_rng(data_seed + dp_idx)

    if rank == info_src:
        print(f"training start time {train_start_time}", flush=True)
    print(f"Rank {rank} Ready, batch size {batch_size}, in this rank, batch size is {batch_size // data_parallel_num}", flush=True)
    loss_acc = 0
    task_loss_acc = 0
    router_loss_acc = 0
    interval_duration = 0.0
    interval_tokens_processed = 0
    final_iteration = min(target_iteration, profile_warmup + profile_iterations) if profile_enabled else target_iteration
    while iteration < final_iteration:
        iteration_range = f"train_iteration_{iteration}|rank={rank}|pp={pipeline_stage_this}|dp={dp_idx}"
        with _nvtx_range(iteration_range):
            ranks = dist.get_process_group_ranks(pp_group)

            first_rank = ranks[0]
            last_rank = ranks[-1]
            train_data, train_target = None, None
            with _nvtx_range("batch_preparation"):
                if rank == first_rank or rank == last_rank:
                    train_data, train_target = get_batch(
                        train_dataset,
                        batch_size // data_parallel_num,
                        context_len,
                        device,
                        rng=data_rng,
                        vocab_size=vocab_size,
                        validate=validate_batches,
                    )

            x_spec = [batch_size // data_parallel_num, context_len, d_model]
            with _nvtx_range("learning_rate_update"):
                lr = lr_scheduler(iteration)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            # Align ranks so the measured pass represents the complete distributed step.
            with _nvtx_range("pre_step_synchronization"):
                torch.cuda.synchronize()
                dist.barrier()
            pass_start_time = time.perf_counter()
            with _nvtx_range("pipeline_step|forward_backward_p2p_dp_sync_optimizer"):
                loss_total = pipelined_train_overlap(
                    model,
                    optimizer,
                    train_data,
                    train_target,
                    x_spec,
                    microbatch_num,
                    pipeline_stage_this,
                    cross_entropy,
                    dtype=pipeline_dtype,
                    device=device,
                    pp_group=pp_group,
                    dp_group=dp_group,
                    router_aux_loss_coef=config["train"].get("router_aux_loss_coef", 0.01),
                    return_metrics=True,
                )
            with _nvtx_range("post_step_synchronization"):
                torch.cuda.synchronize()
                dist.barrier()
            pass_duration = time.perf_counter() - pass_start_time

            loss_acc += loss_total["total_loss"]
            task_loss_acc += loss_total["task_loss"]
            router_loss_acc += loss_total["router_aux_loss"]
            interval_duration += pass_duration
            interval_tokens_processed += batch_size * context_len
            iteration += 1
            print(f"Rank {rank} finishes iteration {iteration}", flush=True)
            if iteration % print_loss_interval == 0:
                if rank == info_src:
                    tokens_per_second = interval_tokens_processed / interval_duration
                    print(
                        f"Iterations {iteration - print_loss_interval + 1}-{iteration}: "
                        f"processed {interval_tokens_processed:,} tokens in "
                        f"{interval_duration:.6f} seconds "
                        f"({tokens_per_second:,.2f} tokens/s)",
                        flush=True,
                    )
                    print(f"Rank {rank} Loss from last {print_loss_interval} iterations is {loss_acc / print_loss_interval}", flush=True)
                    print(f"Task loss: {task_loss_acc / print_loss_interval}; router auxiliary loss (unweighted): {router_loss_acc / print_loss_interval}", flush=True)
                loss_acc = 0
                task_loss_acc = 0
                router_loss_acc = 0
                interval_duration = 0.0
                interval_tokens_processed = 0

            # save checkpoint
            if iteration % save_interval == 0:
                with _nvtx_range("checkpoint_io"):
                    checkpoint_dir = f"{checkpoint_folder}/checkpoint/{train_start_time}/{iteration}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_checkpoint_path = f"{checkpoint_dir}/shaded_dp_{dp_idx}_pp_{pipeline_stage_this}.ckpt"
                    save_checkpoint(model, optimizer, iteration, save_checkpoint_path)


@app.local_entrypoint()
def main(config_path: str = str(BUNDLE_ROOT / "configs/distributed_training_config.yaml")):
    config = load_config(config_path)
    validate_config(config)
    profile_wrapper.remote(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline-parallel training worker")
    parser.add_argument("--profile-worker", nargs=2, metavar=("CONFIG", "CLUSTER_RANK"))
    parser.add_argument("--config", default=str(BUNDLE_ROOT / "configs/smoke.yaml"))
    args = parser.parse_args()
    if args.profile_worker is None:
        config = load_config(args.config)
        validate_config(config, check_data=True, check_cuda=True)
        if config.get("profile", {}).get("enabled", False):
            parser.error("Local launch does not wrap Nsight; disable profile.enabled")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        _spawn_training_workers(config, 0)
    else:
        worker_config_path, worker_cluster_rank = args.profile_worker
        with open(worker_config_path, encoding="utf-8") as f:
            worker_config = json.load(f)
        _spawn_training_workers(worker_config, int(worker_cluster_rank))
