import argparse
import os
from pathlib import Path
import time
from concurrent.futures import Future
from contextlib import nullcontext
from implementation.distributed.ddp_modules import FSDPWrapperPipelined
import torch
import torch.distributed as dist
import numpy as np

import yaml
from implementation.layers import TransformerLMPipelined
from pipeline import pipelined_train_overlap
from implementation.nnfunctions import cross_entropy, learning_rate_schedule_wrapper
from implementation.optimizer import AdamW
from implementation.train_utils import get_batch, load_checkpoint_async_dist, load_dataset, save_checkpoint_async_dist
from datetime import datetime, timedelta



def _nvtx_range(name):
    return torch.cuda.nvtx.range(name) if torch.cuda.is_available() else nullcontext()


def _latest_checkpoint(root, dp_size, pp_size, run=None):
    candidates = []
    for marker in Path(root).glob(f"{run or '*'}/*/COMMITTED"):
        directory = marker.parent
        if directory.name.isdigit() and all(
            (directory / f"shaded_dp_{dp}_pp_{pp}.ckpt").is_file()
            for dp in range(dp_size) for pp in range(pp_size)
        ):
            candidates.append(directory)
    return max(candidates, key=lambda p: (p.parent.name, int(p.name)), default=None)


def _select_checkpoint(config, rank, info_src):
    selection = [None, None]
    if rank == info_src:
        try:
            settings = config["train"]
            load_checkpoint_path = settings.get("load_checkpoint_path")
            root = Path(config["general"]["checkpoint_folder"]) / "checkpoint"
            if load_checkpoint_path:
                path = Path(load_checkpoint_path)
                if int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")):
                    found = _latest_checkpoint(path.parent.parent, settings["data_parallel_num"], settings["pipeline_parallel_stages"], run=path.parent.name)
                    if found is not None and found.parent == path.parent:
                        path = found
                if not (path / "COMMITTED").is_file():
                    raise ValueError(f"Checkpoint is not committed: {path}")
            else:
                path = _latest_checkpoint(root, settings["data_parallel_num"], settings["pipeline_parallel_stages"])
            selection[0] = str(path) if path is not None else None
        except Exception as error:
            selection[1] = str(error)
    dist.broadcast_object_list(selection, src=info_src)
    if selection[1]:
        raise RuntimeError(selection[1])
    return selection[0]


def _verify_pending_checkpoints(pending, iteration, rank, info_src, device, drain=False):
    while pending and (drain or iteration - pending[0][0] >= 5):
        saved_iteration, checkpoint_dir, future = pending[0]
        success = 1
        try:
            future.result()
        except Exception as error:
            success = 0
            print(f"Rank {rank} checkpoint at iteration {saved_iteration} failed: {error}", flush=True)

        status = torch.tensor(success, dtype=torch.int32, device=device)
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        if rank == info_src:
            if status.item() == 1:
                message = f"Checkpoint at iteration {saved_iteration} committed: all ranks saved successfully."
                marker = Path(checkpoint_dir, "COMMITTED.tmp")
                marker.write_text(message + "\n", encoding="utf-8")
                marker.replace(Path(checkpoint_dir, "COMMITTED"))
                print(message, flush=True)
            else:
                print(f"Checkpoint at iteration {saved_iteration} not committed: at least one rank failed.", flush=True)
        pending.pop(0)


def train(local_rank, cluster_rank, world_size, config):
    glob_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    collective_timeout = timedelta(seconds=config.get("recovery", {}).get("collective_timeout_seconds", 180))
    dist.init_process_group("nccl", init_method="env://", rank=glob_rank, world_size=world_size, device_id=torch.device("cuda", local_rank), timeout=collective_timeout)

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
        dp_group_created = dist.new_group(ranks=dp_group_ranks, timeout=collective_timeout)
        if rank in dp_group_ranks:
            dp_group = dp_group_created
    for pp_group_ranks in pp_groups:
        pp_group_created = dist.new_group(ranks=pp_group_ranks, timeout=collective_timeout)
        if rank in pp_group_ranks:
            pp_group = pp_group_created

    checkpoint_folder = config.get("general").get("checkpoint_folder")
    info_src = config.get("general").get("info_src")
    load_checkpoint_path = _select_checkpoint(config, rank, info_src)
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
    train_start_time = now.strftime("%Y%m%d%H%M%S%f")
    iteration = 0
    data_rng = np.random.default_rng(data_seed + dp_idx)
    if load_checkpoint_path:
        train_start_time = Path(load_checkpoint_path).parent.name
        iteration = load_checkpoint_async_dist(load_checkpoint_path + f"/shaded_dp_{dp_idx}_pp_{pipeline_stage_this}.ckpt", model, optimizer, data_rng=data_rng)
        if iteration != int(Path(load_checkpoint_path).name):
            raise ValueError("Checkpoint shard iteration does not match committed directory")
        print(f"loaded checkpoint at {iteration} iterations, training start time is {train_start_time}")

    obj_list = [train_start_time]
    dist.broadcast_object_list(object_list=obj_list, src=info_src)
    train_start_time = obj_list[0]

    lr_scheduler = learning_rate_schedule_wrapper(
        learning_rate_scheduler_config["a_max"], learning_rate_scheduler_config["a_min"], learning_rate_scheduler_config["t_w"], learning_rate_scheduler_config["t_c"]
    )

    dist.barrier()
    train_dataset = load_dataset(train_dataset_path, vocab_size=vocab_size, validate=validate_dataset)

    if rank == info_src:
        print(f"training start time {train_start_time}", flush=True)
    print(f"Rank {rank} Ready, batch size {batch_size}, in this rank, batch size is {batch_size // data_parallel_num}", flush=True)
    loss_acc = 0
    interval_duration = 0.0
    interval_tokens_processed = 0
    pending_checkpoints = []
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
                )
            with _nvtx_range("post_step_synchronization"):
                torch.cuda.synchronize()
                dist.barrier()
            pass_duration = time.perf_counter() - pass_start_time

            loss_acc += loss_total
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
                    loss_acc = 0
                interval_duration = 0.0
                interval_tokens_processed = 0

            _verify_pending_checkpoints(pending_checkpoints, iteration, rank, info_src, device)

            # save checkpoint
            if iteration % save_interval == 0:
                with _nvtx_range("checkpoint_io"):
                    checkpoint_dir = f"{checkpoint_folder}/checkpoint/{train_start_time}/{iteration}"
                    try:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        save_checkpoint_path = f"{checkpoint_dir}/shaded_dp_{dp_idx}_pp_{pipeline_stage_this}.ckpt"
                        future = save_checkpoint_async_dist(model, optimizer, iteration, save_checkpoint_path, data_rng=data_rng)
                    except Exception as error:
                        future = Future()
                        future.set_exception(error)
                    pending_checkpoints.append((iteration, checkpoint_dir, future))

    _verify_pending_checkpoints(pending_checkpoints, iteration, rank, info_src, device, drain=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pipeline training/profile worker entrypoint (run under torchrun)")
    parser.add_argument("--config-path", default=str(Path(__file__).resolve().parent / "configs" / "run_config.yaml"))
    args = parser.parse_args(argv)
    with open(args.config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    required_env = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK")
    if any(key not in os.environ for key in required_env):
        parser.error("this worker entrypoint must be launched by torchrun")
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != config["general"]["n_workers"]:
        raise ValueError("torchrun WORLD_SIZE does not match general.n_workers")
    if int(os.environ["LOCAL_WORLD_SIZE"]) != config["general"]["gpu_per_node"]:
        raise ValueError("torchrun LOCAL_WORLD_SIZE does not match general.gpu_per_node")


    try:
        train(int(os.environ["LOCAL_RANK"]), int(os.environ["GROUP_RANK"]), world_size, config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
