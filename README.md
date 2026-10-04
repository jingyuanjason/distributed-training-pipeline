# Distributed MoE Training Pipeline

A distributed training implementation for a **BF16 Mixtral 8×7B MoE model**, inspired by [Stanford CS336 LLM System Project](https://github.com/stanford-cs336/assignment2-systems) with substantial extensions.

## Implementation
- **Mixtral 8×7B-like BF16 MoE model:** a 32-layer decoder with eight SwiGLU experts per layer, top-2 token routing, and BF16 expert computation, combining grouped-query causal attention, RoPE, and RMSNorm.
- **Fully Sharded Data Parallel (FSDP):** custom parameter sharding and gradient synchronization in [distributed wrappers](implementation/distributed/ddp_modules.py).
- **Data Parallel (DP):** data distribution and multidimensional communication-group setup in [training orchestration](distributed_parallel_training_pipelined.py).
- **Expert Parallel (EP):** distributed experts and all-to-all token routing in [MoE layers](implementation/layers.py).
- **Pipeline Parallel (PP):** microbatch scheduling and communication overlap in [pipeline training](pipeline/pipelined_train_overlap.py).
- **Asynchronous training checkpoints:** improve distributed checkpoint saving (CPU Staging) and restoration for reliable training resumption.
- **Kubernetes Support:** Run training job in kubernetes gpu cluster, with Kubeflow TrainJob

## Results

Training results for the 32-layer MoE model at a context length of **8,192 tokens** on NVIDIA B300 GPUs. Throughput is the total across all GPUs.

| Batch size | Microbatches | GPU | GPU count | GFLOPs/token | Total throughput (tokens/s) | MFU (%) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 32 | B300 | 8 | 84.01 | 75,076 | 35.0 |
| 256 | 32 | B300 | 16 | 84.01 | 132,536 | 30.9 |
| 512 | 32 | B300 | 32 | 84.01 | 241,060 | 28.1 |

## Model Architecture

The current Kubernetes configuration in [values.yaml](/home/jingyuan_li/pipeline_training/kubernetes/chart/values.yaml) defines a **32-layer, top-2 MoE decoder**. The effective architecture below reflects the [model implementation](/home/jingyuan_li/pipeline_training/implementation/layers.py), not just the configuration field names.

| Setting | Effective value |
| --- | --- |
| Transformer layers | **32** = 8 layers per pipeline stage × 4 stages |
| Hidden dimension (`d_model`) | 4,096 |
| Expert intermediate dimension (`d_ff`) | 14,336 |
| Query heads / KV heads | **32 / 8** — 4:1 grouped-query attention (GQA) |
| Head dimension | 128 |
| Attention | Causal scaled dot-product attention with RoPE |
| Experts per layer | **8** = 2 local experts per rank × 4 ranks in the DP/EP group |
| Active experts per token | **2**, selected by a learned router with renormalized top-2 weights |
| Expert architecture | SwiGLU with three projections: gate, up, and down |
| Normalization | Pre-RMSNorm in each transformer block; final RMSNorm before the LM head |
| Vocabulary size | 32,000 |
| Context length | 8,192 tokens |
| Token embeddings / LM head | Separate, untied weights |

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

## Ongoing Work / Future Directions

- **RL Training Integration** Trace collection from RL rollout workers
- **Tensor Parallelism (TP):** shard computation within layers across GPUs, complementing the existing PP, DP, FSDP, and EP setup.

## AI-Generated Content

AI generated portions of the training startup scripts, profiling setup, MoE loss collection, and project-structure cleanup/refactoring and some local test cases (not included in the repo). Also used AI for debugging.
