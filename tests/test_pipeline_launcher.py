"""CPU-only tests for wrapper commands and the pipeline worker entrypoint."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import distributed_parallel_training_pipelined as training
import modal_wrapper as launcher


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("profile", [False, True])
def test_torchrun_command_and_cleanup(monkeypatch, tmp_path, fail, profile):
    config = {"general": {"n_workers": 16, "gpu_per_node": 8}, "profile": {"enabled": profile, "output_dir": str(tmp_path)}}
    monkeypatch.setattr(launcher.modal.experimental, "get_cluster_info", lambda: SimpleNamespace(container_ips=["::1"], rank=1, cluster_id="test"))
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "nsys")
    commits = []
    monkeypatch.setattr(launcher, "volume_profile", SimpleNamespace(commit=lambda: commits.append(True)))
    paths = []

    def run(command, check, env):
        if profile:
            assert command[:2] == ["nsys", "profile"]
        else:
            assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
        for argument in ["--nnodes=2", "--nproc-per-node=8", "--node-rank=1", "--master-addr=::1", "--master-port=29500"]:
            assert argument in command
        assert check
        assert command[command.index("--module") + 1] == "distributed_parallel_training_pipelined"
        assert env["MASTER_ADDR"] == "::1"
        path = Path(command[-1])
        paths.append(path)
        assert json.loads(path.read_text()) == config
        if fail:
            raise subprocess.CalledProcessError(1, command)
        if profile:
            (tmp_path / "pipeline-node-1.nsys-rep").touch()

    monkeypatch.setattr(launcher.subprocess, "run", run)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            launcher.profile_wrapper.local(config)
    else:
        launcher.profile_wrapper.local(config)
    assert paths and not paths[0].exists()
    assert commits == ([True] if profile and not fail else [])


@pytest.mark.parametrize("world_size,gpus,rank", [(0, 8, 0), (8, 0, 0), (9, 8, 0), (8, 8, 1), (8, 8, -1)])
def test_invalid_topology(world_size, gpus, rank):
    with pytest.raises(ValueError):
        launcher._build_torchrun_command("/tmp/config.json", rank, world_size, gpus)


def test_multi_node_requires_master(monkeypatch):
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    with pytest.raises(ValueError, match="MASTER_ADDR"):
        launcher._build_torchrun_command("/tmp/config.json", 0, 16, 8)


def test_worker_dispatch(monkeypatch, tmp_path):
    config = {"general": {"n_workers": 16, "gpu_per_node": 8}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    for key, value in {"LOCAL_RANK": "3", "RANK": "11", "WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8", "GROUP_RANK": "1"}.items():
        monkeypatch.setenv(key, value)
    calls = []
    monkeypatch.setattr(training, "_train_worker", lambda *args: calls.append(args))
    training.main(["--config-path", str(path)])
    assert calls == [(3, 1, 16, config)]


def test_worker_requires_torchrun(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"general": {"n_workers": 8, "gpu_per_node": 8}}))
    with pytest.raises(SystemExit, match="2"):
        training.main(["--config-path", str(path)])
    assert not hasattr(training, "_spawn_training_workers")
    assert not hasattr(training, "subprocess")


def test_profile_wraps_torchrun(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "::1")
    config = {"general": {"n_workers": 16, "gpu_per_node": 8}, "profile": {"nsight_trace": "cuda-sw,nvtx,nccl"}}
    torchrun = launcher._build_torchrun_command("/tmp/config.json", 1, 16, 8)
    command = launcher._build_nsys_command("nsys", "/tmp/report", torchrun, config["profile"])
    assert command[:2] == ["nsys", "profile"]
    assert "--trace=cuda-sw,nvtx,nccl" in command
    torchrun = launcher._build_torchrun_command("/tmp/config.json", 1, 16, 8)
    assert command[-len(torchrun):] == torchrun


def test_real_torchrun_cpu_dispatch(tmp_path):
    """Exercise the real launcher and entrypoint without starting GPU training."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"general": {"n_workers": 2, "gpu_per_node": 2}}))
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import distributed_parallel_training_pipelined as launcher\n"
        "def record(local_rank, node_rank, world_size, config):\n"
        f"    Path({str(tmp_path)!r}, 'rank-' + os.environ['RANK']).write_text(str((local_rank, node_rank, world_size)))\n"
        "launcher._train_worker = record\n"
        "launcher.main()\n"
    )
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", str(worker), "--config-path", str(config)],
        check=True, timeout=90,
    )
    assert (tmp_path / "rank-0").read_text() == "(0, 0, 2)"
    assert (tmp_path / "rank-1").read_text() == "(1, 0, 2)"