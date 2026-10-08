# Training chart

## Parallel topology

The defaults use four nodes with eight GPUs each: FSDP=8, DDP=1, PP=4.
`runConfig.train.fsdp_parallel_num` sets the shard/MoE routing group size;
`runConfig.train.data_parallel_num` sets the DDP replica count. DDP groups
connect matching shards. Each combined FSDP × DDP stage block stays within
one node; pipeline groups may span nodes.

The chart validates:
- FSDP × DDP × PP equals `numNodes * gpusPerNode`.
- FSDP × DDP divides `gpusPerNode`.
- `batch_size` is divisible by FSDP × DDP × `microbatch_num`.

For FSDP=2 and DDP=4 on the same four eight-GPU nodes:

```bash
helm template hybrid /home/jingyuan_li/pipeline_training/kubernetes/chart \
  --set runConfig.train.fsdp_parallel_num=2 \
  --set runConfig.train.data_parallel_num=4
```

Use a training image built from the updated code; the chart does not build or
update the image automatically. Grouped MoE expert parameters still require
cross-DDP initialization and gradient synchronization support before hybrid
MoE training is fully synchronized. Do not resume checkpoints under a different
FSDP/DDP topology.

When migrating old Helm overrides, rename the old `data_parallel_num` value to
`fsdp_parallel_num` and set `data_parallel_num=1`. Helm merges chart defaults,
so simply omitting the new key from an override does not select legacy behavior.

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