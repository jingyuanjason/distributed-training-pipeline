"""Ensure the migrated source resolves inside the standalone project."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", [
    "modal_wrapper",
    "distributed_parallel_training_pipelined",
    "implementation.chaos_engineer",
    "implementation.distributed.ddp_modules",
    "implementation.kernels.attention",
    "implementation.layers",
    "implementation.nnfunctions",
    "implementation.nn_utils",
    "implementation.optimizer",
    "implementation.train_utils",
])
def test_local_dependency(name):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(ROOT)


def test_modal_paths_do_not_depend_on_cwd(monkeypatch, tmp_path):
    import modal_wrapper

    monkeypatch.chdir(tmp_path)
    importlib.reload(modal_wrapper)
    assert modal_wrapper.PROJECT_ROOT == ROOT


def test_chaos_disabled():
    from implementation.chaos_engineer import ChaosEnginnerTrigger

    trigger = ChaosEnginnerTrigger({"random_kill": {"enabled": False}})
    assert trigger._triggers == []
    trigger.trigger()


def test_project_layout():
    assert not (ROOT / "cs336_systems").exists()
    assert not (ROOT / "cs336_basics").exists()
    assert (ROOT / "configs" / "run_config.yaml").is_file()
    for name in ("modal_wrapper", "distributed_parallel_training_pipelined"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().parent == ROOT


def test_default_worker_config(monkeypatch, tmp_path):
    import distributed_parallel_training_pipelined as training

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    # Reading the bundled default succeeds before the torchrun guard fires.
    with pytest.raises(SystemExit, match="2"):
        training.main([])