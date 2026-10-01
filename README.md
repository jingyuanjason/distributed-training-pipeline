# Distributed MoE Training Pipeline

A distributed training implementation for a **BF16 Mixtral 8×7B MoE model**, inspired by [Stanford CS336 LLM System Project](https://github.com/stanford-cs336/assignment2-systems) with substantial extensions.

## Implementation

- **Fully Sharded Data Parallel (FSDP):** custom parameter sharding and gradient synchronization in [distributed wrappers](implementation/distributed/ddp_modules.py).
- **Data Parallel (DP):** data distribution and multidimensional communication-group setup in [training orchestration](distributed_parallel_training_pipelined.py).
- **Expert Parallel (EP):** distributed experts and all-to-all token routing in [MoE layers](implementation/layers.py).
- **Pipeline Parallel (PP):** microbatch scheduling and communication overlap in [pipeline training](pipeline/pipelined_train_overlap.py).
- **Asynchronous training checkpoints:** improve distributed checkpoint saving (CPU Staging) and restoration for reliable training resumption.

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

Deploy the Helm chart (requires Kubeflow Trainer v2, JobSet, the
`torch-distributed` ClusterTrainingRuntime, and NVIDIA GPU support):

```bash
helm install pipeline-training /home/jingyuan_li/pipeline_training/kubernetes
```

Edit `runConfig` in `/home/jingyuan_li/pipeline_training/kubernetes/values.yaml`
for Kubernetes runs. The chart defaults to one node with eight GPUs and retains
the current image. Set `image`, `numNodes`, `gpusPerNode`, `runtimeRef`, and
optional CPU/memory `resourcesPerNode` using Helm values or `--set`.
Worker counts in the ConfigMap are derived from the node/GPU counts; DP/PP and
batch settings must match. Review the existing chaos-engineering settings before
production use: fault injection is enabled in the preserved defaults.

The separate `/home/jingyuan_li/pipeline_training/configs/run_config.yaml`
remains the local/Modal configuration; the two copies are not automatically
synchronized. No image rebuild is required for config changes. The TrainJob
mounts its `run_config.yaml` key read-only at `/app/configs/run_config.yaml`,
overriding the image's config without rebuilding it. Both resources must be in
the same namespace. This uses Kubeflow Trainer v2's `podSpecOverrides` API;
check that your installed TrainJob CRD supports it.

Resources are named `train-<release>` and `train-<release>-config`. For the release
above they are `train-pipeline-training` and `train-pipeline-training-config`.
The runtime must expose a replicated job and container named `node`.

TrainJob fields may be immutable, and Helm upgrades do not automatically restart
a training run. To change configuration, uninstall and reinstall the release
(this stops workers and deletes the ConfigMap). A `subPath` mount does not receive
live ConfigMap updates, and workers load configuration only at startup:

```bash
helm uninstall pipeline-training
helm install pipeline-training /home/jingyuan_li/pipeline_training/kubernetes
```

If the earlier plain manifests were deployed, delete their TrainJob
`pipeline-training` and ConfigMap `pipeline-training-config` before installing
the chart to avoid running duplicate jobs. This chart still does not provide
persistent checkpoint storage; pod replacement loses container-local files.

For eight nodes with eight GPUs each, one matching topology is:

```bash
helm install pipeline-training /home/jingyuan_li/pipeline_training/kubernetes \
  --set numNodes=8 \
  --set runConfig.train.data_parallel_num=8 \
  --set runConfig.train.pipeline_parallel_stages=8 \
  --set runConfig.train.batch_size=256
```

Multi-node training requires extending the runtime with shared checkpoint storage
supporting atomic rename. GPU memory fit depends on hardware and model settings.

The base Dockerfile installs Python 3.12 and production dependencies from `uv.lock`.
The training Dockerfile inherits that image, then copies and installs the project
under `/app`. The environment at `/opt/venv`
is on `PATH`, so `python` and `torchrun` work directly. There is no default command
or entry point: Kubeflow must supply the launcher. Development dependencies are
excluded; Modal remains installed because it is a declared project dependency.

Build and push from the repository root (replace the example registry). Build the
base once, then rebuild it only when dependencies or the base Dockerfile change:

```bash
docker build --platform linux/amd64 \
  -f dockerfiles/baseimage/Dockerfile \
  -t YOUR_REGISTRY/pipeline-training-base:0.1.0 .
docker push YOUR_REGISTRY/pipeline-training-base:0.1.0

docker build --platform linux/amd64 \
  -f dockerfiles/train/Dockerfile \
  --build-arg BASE_IMAGE=YOUR_REGISTRY/pipeline-training-base:0.1.0 \
  -t YOUR_REGISTRY/pipeline-training:0.1.0 .
docker push YOUR_REGISTRY/pipeline-training:0.1.0
```

For code-only releases, run only the training build/push with a new training tag
and the same `BASE_IMAGE`. Nodes that already have the base layers download only
the changed application layers; the first pull still downloads the full image.
Use immutable base tags (or digests), and publish a new base whenever
`pyproject.toml` or `uv.lock` changes. For local builds, the default `BASE_IMAGE`
is `pipeline-training-base:0.1.0`.

The locked Linux PyTorch wheels include CUDA 13 and NCCL user-space libraries;
the image does not need a separate CUDA base image. GPU nodes still require a
compatible NVIDIA driver, NVIDIA Container Toolkit, and Kubernetes GPU device
plugin (or GPU Operator). This image does not include the full CUDA development
toolkit or Nsight Systems. Keep profiling disabled unless you extend the image.

For a **single-node, eight-GPU job**, use these fields in the training container
specification (this is a container fragment, not a complete Kubeflow resource):

```yaml
image: YOUR_REGISTRY/pipeline-training:0.1.0
workingDir: /app
command: ["torchrun"]
args:
  - "--standalone"
  - "--nnodes=1"
  - "--nproc-per-node=8"
  - "--max-restarts=3"
  - "--module"
  - "distributed_parallel_training_pipelined"
  - "--config-path"
  - "/app/configs/run_config.yaml"
resources:
  limits:
    nvidia.com/gpu: 8
```

- Mount the dataset and writable checkpoint volumes at the paths in your run
  configuration. For multi-node recovery, checkpoints must be on shared storage
  supporting atomic rename. Local data and credentials are excluded by
  `.dockerignore`; supply credentials through Kubernetes Secrets.
- Override the run configuration with a ConfigMap if needed. Match worker count,
  GPUs per node, and parallelism settings to the allocated resources. Review the
  `chaos_engineering` flags before production runs; the current configuration
  enables some fault injection.
- Mount a memory-backed `emptyDir` at `/dev/shm` for NCCL/PyTorch shared memory,
  with a size limit appropriate for your job. Account for it in pod memory limits.
- For multi-node jobs, do **not** use `--standalone`. Configure `torchrun` with
  the node count, node rank, and a shared master/rendezvous endpoint supplied by
  your Kubeflow setup. If your Kubeflow runtime already launches `torchrun`, supply
  the training module to that runtime instead of launching a second `torchrun`.
  Modal-specific networking settings are intentionally not applied by this image.

After building, run a CPU-side import/launcher smoke test:

```bash
docker run --rm YOUR_REGISTRY/pipeline-training:0.1.0 \
  python -c 'import torch, triton, numpy, yaml, psycopg; import distributed_parallel_training_pipelined; print(torch.__version__, torch.version.cuda)'
docker run --rm YOUR_REGISTRY/pipeline-training:0.1.0 torchrun --help
```

On a GPU-enabled Docker host, also check driver access:

```bash
docker run --rm --gpus all YOUR_REGISTRY/pipeline-training:0.1.0 \
  python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

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
