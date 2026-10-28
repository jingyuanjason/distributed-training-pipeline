# Training chart

## Parallel topology

The defaults use two nodes with eight GPUs each: TP=1, FSDP=2, DP=2, PP=4.
Model parallel sizes live in `runConfig.model.shard_model`; data parallel
sizes live in `runConfig.train.shard_data`. Each entry has a `shard` count.
`EP.use_tp_topology=true` reuses TP groups instead of adding a rank dimension.
`runConfig.general.gpu_placement_priority` lists fastest-varying dimensions
first, favoring node-local placement when ranks are contiguous per node.

The chart validates:
- Independent TP × EP × FSDP × PP × DP equals `numNodes * gpusPerNode`.
  EP is excluded when sharing TP topology.
- `global_batch_size` is divisible by FSDP × DP × `microbatch_num`.
- `model.num_layers_model` is divisible by PP shards.

For FSDP=2 and DP=4 on four eight-GPU nodes:

```bash
helm template hybrid /home/jingyuan_li/pipeline_training/kubernetes/chart \
  --set numNodes=4 \
  --set runConfig.train.shard_data.FSDP.shard=2 \
  --set runConfig.train.shard_data.DP.shard=4
```

Use a training image built from the updated code; the chart does not build or
update the image automatically. Grouped MoE expert parameters still require
cross-DDP initialization and gradient synchronization support before hybrid
MoE training is fully synchronized. Do not resume checkpoints under a different
FSDP/DDP topology.

When migrating Helm overrides, replace `fsdp_parallel_num` and
`data_parallel_num` with `shard_data.FSDP.shard` and `shard_data.DP.shard`;
move `pipeline_parallel_stages` to `model.shard_model.PP.shard` and rename
`batch_size` to `global_batch_size`. Model fields are now `num_layers_model`
and `num_expert_model` (the latter is the total expert count), and
`tokenizer_path` now lives under `data` rather than `general`.

## Runtime and memlock

With `memlock.enabled` and `memlock.createRuntime` enabled (the defaults), a
pre-install/pre-upgrade hook creates the cluster-scoped runtime named by
`runtimeRef.name` before the TrainJob is submitted. Its training container has
`SYS_RESOURCE`, allowing the launcher to run `ulimit -l unlimited`.
Cluster security policies must permit this capability.

The chart uses Helm `lookup` to reuse an existing runtime without modifying or
replacing it. The runtime is shared across training releases and retained on
uninstall. Existing runtimes must already provide the required capability;
this chart validates that the `node` job's `node` container grants `SYS_RESOURCE`
(or is already privileged) and fails with an explanatory error otherwise. It does
not repair existing runtimes. Runtime changes must be managed separately.
Avoid simultaneous first installs competing to create the shared runtime.

The installing identity needs cluster-scoped runtime read/create permissions.
To use an administrator-provisioned runtime, set `memlock.createRuntime=false`.
Offline `helm template` cannot see existing runtimes and renders the hook;
do not apply that output wholesale with kubectl, which does not execute Helm
hook ordering. Deploy using Helm instead:

```bash
helm upgrade --install train-workload-v10 \
  /home/jingyuan_li/pipeline_training/kubernetes/chart \
  -n mock-train-ns
```

## Checking runtime creation

When the runtime already exists, the hook is deliberately omitted. An empty
`helm get hooks` result is therefore normal for a release that reused it. The
runtime is not an ordinary release-owned resource: Helm upgrades do not update
its image or security configuration. The TrainJob's `trainer.image` still selects
the training image for each run.

Validate both paths without launching training:

```bash
helm lint /home/jingyuan_li/pipeline_training/kubernetes/chart
helm upgrade --install train-workload-v10 \
  /home/jingyuan_li/pipeline_training/kubernetes/chart \
  -n mock-train-ns --dry-run=server
helm template runtime-check /home/jingyuan_li/pipeline_training/kubernetes/chart \
  --show-only templates/memlock-runtime.yaml
```

Use a different `runtimeRef.name` to create a separate runtime. Do not delete an
in-use shared runtime to force recreation: Kubeflow protects it with a finalizer,
and other TrainJobs may depend on it. `helm.sh/resource-policy: keep` is intentional;
do not add a `hook-succeeded` deletion policy to this hook.