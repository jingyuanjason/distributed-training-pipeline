# Distributed MoE Training Pipeline

A distributed training implementation for a **BF16 Mixtral 8×7B MoE model**, inspired by [Stanford CS336 LLM System Project](https://github.com/stanford-cs336/assignment2-systems) with substantial extensions.

## Implementation

- **Fully Sharded Data Parallel (FSDP):** custom parameter sharding and gradient synchronization in [distributed wrappers](implementation/distributed/ddp_modules.py).
- **Data Parallel (DP):** data distribution and multidimensional communication-group setup in [training orchestration](distributed_parallel_training_pipelined.py).
- **Expert Parallel (EP):** distributed experts and all-to-all token routing in [MoE layers](implementation/layers.py).
- **Pipeline Parallel (PP):** microbatch scheduling and communication overlap in [pipeline training](pipeline/pipelined_train_overlap.py).
- **Asynchronous training checkpoints:** improve distributed checkpoint saving (CPU Staging) and restoration for reliable training resumption.
- **Kubernetes Support:** Run training job in kubernetes gpu cluster, with Kubeflow TrainJob

## Running the Project

Use Python 3.12 or 3.13 on Linux with CUDA/NCCL and BF16-capable NVIDIA GPUs. Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Edit [the run configuration](/home/jingyuan_li/pipeline_training/configs/run_config.yaml): set `data.train_dataset_path` to an existing 1D integer NumPy `.npy` token array (IDs in `[0, vocab_size)`, longer than `model.context_len`) and `general.checkpoint_folder` to a writable directory. adjust model and batch sizes to fit your GPU memory. Keep `data_parallel_num * pipeline_parallel_stages == n_workers` and `batch_size` divisible by `data_parallel_num * microbatch_num`.

Launch the default single-node topology:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=3 \
  --module distributed_parallel_training_pipelined \
  --config-path /home/jingyuan_li/pipeline_training/configs/run_config.yaml
```

The process count must match both `general.gpu_per_node` and `general.n_workers` for a single-node run.

**Modal alternative:** authenticate with `modal setup`, create/populate the `datasets` Modal Volume with tokens matching the configured path under `/mnt/dataset`, then run:

```bash
modal run modal_wrapper.py --config-path /home/jingyuan_li/pipeline_training/configs/run_config.yaml
```

The [Modal wrapper](modal_wrapper.py) currently requests one node with eight B300 GPUs. Keep its GPU request and cluster size consistent with the configuration. Leave `profile.enabled: false` for normal training; Modal profiling requires `nsys` in the training image.


## Docker / Kubeflow

Requires Helm, Kubeflow Trainer v2, JobSet, the `torch-distributed`
ClusterTrainingRuntime, and NVIDIA GPU support.

### Build and push

Replace `YOUR_REGISTRY` with your registry. Rebuild the base image only when
its Dockerfile or dependencies change; for code changes, rebuild only the training image.

```bash
cd /home/jingyuan_li/pipeline_training
docker build --platform linux/amd64 \
  -f /home/jingyuan_li/pipeline_training/dockerfiles/baseimage/Dockerfile \
  -t YOUR_REGISTRY/pipeline-training-base:0.1.0 .
docker push YOUR_REGISTRY/pipeline-training-base:0.1.0

docker build --platform linux/amd64 \
  -f /home/jingyuan_li/pipeline_training/dockerfiles/train/Dockerfile \
  --build-arg BASE_IMAGE=YOUR_REGISTRY/pipeline-training-base:0.1.0 \
  -t YOUR_REGISTRY/pipeline-training:0.1.0 .
docker push YOUR_REGISTRY/pipeline-training:0.1.0
```

### Configure and deploy

Edit `/home/jingyuan_li/pipeline_training/kubernetes/values.yaml`:

- `numNodes` and `gpusPerNode`: defaults to **1 node × 8 GPUs**.
- `runConfig`: training settings, mounted into the pods through a ConfigMap.
  Worker counts are derived automatically; keep DP/PP and batch settings compatible.
- Disable the enabled chaos-engineering options for normal training.

```bash
helm install pipeline-training /home/jingyuan_li/pipeline_training/kubernetes \
  --set image=YOUR_REGISTRY/pipeline-training:0.1.0
```

Config changes require no image rebuild. To start a new run with changed settings,
run `helm uninstall pipeline-training`, then repeat the install command. This stops
existing workers; Helm upgrades do not automatically restart training.

**Storage:** the chart does not mount datasets or persistent checkpoints. Extend
the runtime with the required volumes before real-data training; multi-node
checkpointing needs shared storage supporting atomic rename. Container-local
checkpoints are lost when pods are replaced.

Local/Modal runs still use the separate
`/home/jingyuan_li/pipeline_training/configs/run_config.yaml`.

## Topology

The current multi-node topology is designed to match the communication characteristics of each parallelism strategy with the available hardware interconnect:

- **Intra-node: Data Parallel (DP) and Expert Parallel (EP)** groups are placed **within the same node**, where high-speed NVLink is available. Both DP (gradient synchronization) and EP (all-to-all token routing) are bandwidth-intensive, so they benefit from NVLink's high intra-node bandwidth.
- **Inter-node: Pipeline Parallel (PP)** stages are placed **across different nodes**, since PP communication is limited to sending activations forward and gradients backward between adjacent stages—point-to-point transfers that are small enough to tolerate slower inter-node networking.


## Recovery

1. **Discover:** On startup, one rank selects the latest committed checkpoint and
   broadcasts its path to all ranks, ignoring incomplete saves. Without a committed
   checkpoint, training starts from scratch.
2. **Restore:** All ranks reload model, optimizer, iteration/LR schedule, and
   sampling and Torch/Python RNG states. Checkpoints missing RNG state are rejected.
3. **Restart:** If a rank fails, the launcher recreates the entire worker group,
   which resumes from the latest committed step. An explicitly selected checkpoint
   sets the initial restore point; retries advance within that same run.

Recovery requires unchanged topology, model, dataset, and training settings, plus
shared checkpoint storage with atomic rename. Keep checkpoint directories separate
per experiment; multi-node workers must join the same rendezvous. Restart attempts
are bounded, and node or launcher failures require external relaunch. Checkpoint
publication does not protect against storage loss or corruption.

## Recorded Results

| GPU count | Model FLOPs utilization (MFU) | Throughput (tokens/s/GPU) |
| --- | --- | --- |
| 8 | 31.9% | 11,008 |
| 32 | 24.5% | 8,584 |

At a context length of **8,192 tokens**, the model FLOPs estimate is **65.2 GFLOPs**, calculated using only the **activated parameters** in the MoE model, rather than all expert parameters. MFU is calculated using the **NVIDIA B200 GPU's FP16 peak performance** as the hardware reference.

## Ongoing Work / Future Directions

- **RL Training Integration** Trace collection from RL rollout workers
- **Tensor Parallelism (TP):** shard computation within layers across GPUs, complementing the existing PP, DP, FSDP, and EP setup.

## AI-Generated Content

AI generated portions of the training startup scripts, profiling setup, MoE loss collection, and project-structure cleanup/refactoring and some local test cases (not included in the repo). Also used AI for debugging.
