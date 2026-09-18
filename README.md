# Distributed MoE Training Pipeline

A distributed training implementation for a **BF16 Mixtral 8×7B MoE model**, inspired by [Stanford CS336 LLM System Project](https://github.com/stanford-cs336/assignment2-systems) with substantial extensions.

## Implementation

- **Fully Sharded Data Parallel (FSDP):** custom parameter sharding and gradient synchronization in [distributed wrappers](implementation/distributed/ddp_modules.py).
- **Data Parallel (DP):** data distribution and multidimensional communication-group setup in [training orchestration](distributed_parallel_training_pipelined.py).
- **Expert Parallel (EP):** distributed experts and all-to-all token routing in [MoE layers](implementation/layers.py).
- **Pipeline Parallel (PP):** microbatch scheduling and communication overlap in [pipeline training](pipeline/pipelined_train_overlap.py).

## Running the Project

Use Python 3.12 or 3.13 on Linux with CUDA/NCCL and BF16-capable NVIDIA GPUs. Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Edit [the run configuration](configs/run_config.yaml): set `data.train_dataset_path` to an existing 1D integer NumPy `.npy` token array (IDs in `[0, vocab_size)`, longer than `model.context_len`) and `general.checkpoint_folder` to a writable directory. Use absolute paths. The default configuration uses **one node with eight GPUs** and a large model; adjust model and batch sizes to fit your GPU memory. Keep `data_parallel_num * pipeline_parallel_stages == n_workers` and `batch_size` divisible by `data_parallel_num * microbatch_num`.

Launch the default single-node topology:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=3 \
  --module distributed_parallel_training_pipelined \
  --config-path configs/run_config.yaml
```

The process count must match both `general.gpu_per_node` and `general.n_workers` for a single-node run.

**Modal alternative:** authenticate with `modal setup`, create/populate the `datasets` Modal Volume with tokens matching the configured path under `/mnt/dataset`, then run:

```bash
modal run modal_wrapper.py --config-path configs/run_config.yaml
```

The [Modal wrapper](modal_wrapper.py) currently requests one node with eight B300 GPUs. Keep its GPU request and cluster size consistent with the configuration. Leave `profile.enabled: false` for normal training; Modal profiling requires `nsys` in the training image.

## Recovery

On startup, one rank discovers the latest complete `COMMITTED` checkpoint under
`general.checkpoint_folder/checkpoint` and broadcasts its path to all ranks.
Incomplete saves are ignored. Model, optimizer, iteration (and hence LR schedule),
data-sampling RNG, and Torch/Python RNG states are restored. With no committed
checkpoint, training starts from scratch. Use a separate checkpoint folder per
experiment; automatic selection orders run directories by timestamp, then step.
`train.load_checkpoint_path` pins the initial checkpoint; elastic retries advance
to the latest committed step in that run. Older checkpoints without RNG state are
rejected for recovery rather than silently repeating data.

`torchrun --max-restarts=3` kills and recreates the worker group after a rank fails;
the Modal launcher sets this from `recovery.max_restarts` (default 3). For multi-node
launches, use a fixed node count, `--rdzv-backend=c10d`, and the same unique
`--rdzv-id` and reachable `--rdzv-endpoint=HOST:PORT` on every node. Do not run
independent jobs against the same checkpoint directory. Topology, model, dataset,
and training settings must remain unchanged. Checkpoint storage must provide
shared visibility and atomic rename across ranks. Modal Volume snapshots are not
a substitute for a coherent shared filesystem across multiple containers.

The collective timeout defaults to 180 seconds and is configurable through
`recovery.collective_timeout_seconds`. Restart budgets are finite; whole-node or
launcher failures need external infrastructure to relaunch/replace the node.
Publication protects against worker interruption, not storage loss or corruption.

## Topology

The current multi-node topology is designed to match the communication characteristics of each parallelism strategy with the available hardware interconnect:

- **Intra-node: Data Parallel (DP) and Expert Parallel (EP)** groups are placed **within the same node**, where high-speed NVLink is available. Both DP (gradient synchronization) and EP (all-to-all token routing) are bandwidth-intensive, so they benefit from NVLink's high intra-node bandwidth.
- **Inter-node: Pipeline Parallel (PP)** stages are placed **across different nodes**, since PP communication is limited to sending activations forward and gradients backward between adjacent stages—point-to-point transfers that are small enough to tolerate slower inter-node networking.

## Beyond the CS336 Project

The project adds PP and EP, detailed NVIDIA Nsight Systems profiling, training outside Modal, and large-scale experiment support with multidimensional communication groups—targeting up to 32 NVIDIA B200/B300 GPUs. Much larger models and context lengths require distributing training across GPUs.

## Ongoing Work / Future Directions

- **Training checkpoints:** improve distributed checkpoint saving and restoration for reliable training resumption.
- **Tensor Parallelism (TP):** shard computation within layers across GPUs, complementing the existing PP, DP, FSDP, and EP setup.

## AI-Generated Content

AI generated portions of the training startup scripts, profiling setup, MoE loss collection, and project-structure cleanup/refactoring and some local test cases (not included in the repo).
