"""Modal deployment and Nsight profiling wrapper for pipeline training."""

import json
from importlib.metadata import requires
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import modal
import modal.experimental
import yaml

app = modal.App("tiny-story-distributed-train")

CUDA_VERSION = "13.2.1"
PYTHON_VERSION = "3.12"
PROJECT_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-cudnn-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    # Required by Nsight's report importer.
    .apt_install("libdw1")
)
if (PROJECT_ROOT / "pyproject.toml").is_file():
    image = image.pip_install_from_pyproject(str(PROJECT_ROOT / "pyproject.toml"))
else:
    image = image.pip_install(*requires("pipeline-training"))
image = (
    image
    .add_local_dir(str(PROJECT_ROOT / "implementation"), "/root/implementation")
    .add_local_dir(str(PROJECT_ROOT / "pipeline"), "/root/pipeline")
    .add_local_file(str(PROJECT_ROOT / "distributed_parallel_training_pipelined.py"), "/root/distributed_parallel_training_pipelined.py")
    .add_local_dir(str(PROJECT_ROOT / "configs"), "/root/configs")
)


volume_dataset = modal.Volume.from_name("datasets")
volume_profile = modal.Volume.from_name("profile-data", create_if_missing=True)
volume_checkpoints = modal.Volume.from_name("checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="B300:8",
    volumes={"/mnt/dataset": volume_dataset, "/mnt/checkpoints": volume_checkpoints, "/mnt/profile-data": volume_profile},
    scaledown_window=10,
    timeout=60 * 60 * 12,
)
@modal.experimental.clustered(size=1)
def profile_wrapper(config):
    """Launch training on this node, optionally wrapping torchrun with Nsight."""
    cluster = modal.experimental.get_cluster_info()
    os.environ["MASTER_ADDR"] = cluster.container_ips[0]
    os.environ["MASTER_PORT"] = "29500"

    # One host ID per physical node, inherited by its eight children.
    os.environ["NCCL_HOSTID"] = f"{cluster.cluster_id}-node-{cluster.rank}"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET6"
    os.environ["NCCL_DEBUG"] = "WARN"

    profile_config = config.get("profile", {})
    profile_enabled = profile_config.get("enabled", False)
    general = config["general"]
    with tempfile.TemporaryDirectory(prefix="pipeline-training-") as directory:
        config_path = Path(directory) / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        command = _build_torchrun_command(config_path, cluster.rank, general["n_workers"], general["gpu_per_node"], config.get("recovery", {}).get("max_restarts", 3))
        if profile_enabled:
            nsys = shutil.which("nsys")
            if nsys is None:
                raise RuntimeError("Nsight Systems (`nsys`) is not installed in the training image. Use a CUDA image containing Nsight Systems or disable profile.enabled.")
            output_dir = profile_config.get("output_dir", "/mnt/profile-data")
            os.makedirs(output_dir, exist_ok=True)
            output = os.path.join(output_dir, f"pipeline-node-{cluster.rank}")
            command = _build_nsys_command(nsys, output, command, profile_config)
        print(f"Launching {'Nsight Systems' if profile_enabled else 'training'}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, env=os.environ.copy())
        if profile_enabled:
            report_path = f"{output}.nsys-rep"
            if not os.path.isfile(report_path):
                raise RuntimeError(f"Nsight Systems did not create {report_path}")
            volume_profile.commit()
            print(f"Committed Nsight report to the profile-data Volume: {report_path}", flush=True)


def _build_torchrun_command(config_path, cluster_rank, world_size, gpu_per_node, max_restarts=3):
    if not isinstance(max_restarts, int) or max_restarts < 0:
        raise ValueError("recovery.max_restarts must be a nonnegative integer")
    if world_size < 1 or gpu_per_node < 1 or world_size % gpu_per_node:
        raise ValueError("n_workers must be positive and divisible by gpu_per_node")
    nnodes = world_size // gpu_per_node
    if not 0 <= cluster_rank < nnodes:
        raise ValueError("node rank must be in [0, n_workers / gpu_per_node)")
    if nnodes > 1 and not os.environ.get("MASTER_ADDR"):
        raise ValueError("MASTER_ADDR must be set for multi-node training")
    return [
        sys.executable, "-m", "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc-per-node={gpu_per_node}",
        f"--node-rank={cluster_rank}",
        f"--master-addr={os.environ.get('MASTER_ADDR', '127.0.0.1')}",
        f"--master-port={os.environ.get('MASTER_PORT', '29500')}",
        f"--max-restarts={max_restarts}",
        "--monitor-interval=1",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=[{os.environ.get('MASTER_ADDR', '127.0.0.1')}]:{os.environ.get('MASTER_PORT', '29500')}",
        f"--rdzv-id={os.environ.get('NCCL_HOSTID', 'pipeline').rsplit('-node-', 1)[0]}",
        "--module", "distributed_parallel_training_pipelined",
        "--config-path", str(config_path),
    ]


def _build_nsys_command(nsys, output, training_command, profile_config):
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
    ] + training_command


@app.local_entrypoint()
def main(config_path: str = str(PROJECT_ROOT / "configs" / "run_config.yaml")):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    profile_wrapper.remote(config)
