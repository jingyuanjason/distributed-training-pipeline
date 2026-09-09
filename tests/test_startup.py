from pathlib import Path

import numpy as np
import pytest
from implementation.startup import load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_import_training():
    import train as training
    from implementation.pipeline import pipelined_train_overlap
    assert training.pipelined_train_overlap is pipelined_train_overlap
    assert Path(training.__file__).resolve().is_relative_to(ROOT)
    assert training.BUNDLE_ROOT == ROOT
    command = training._build_nsys_command("nsys", "report", "config.json", 0, {})
    assert command[-4:] == [str(ROOT / "train.py"), "--profile-worker", "config.json", "0"]


def test_schedule_modules():
    from implementation import layers
    from implementation.pipeline import pipelined_train, pipelined_train_overlap

    assert pipelined_train.__module__ == "implementation.pipeline.pipelined_train"
    assert pipelined_train_overlap.__module__ == "implementation.pipeline.pipelined_train_overlap"
    assert not hasattr(layers, "pipelined_train")
    assert not hasattr(layers, "pipelined_train_overlap")


def test_configs():
    for name in ("smoke", "modal"):
        config = load_config(ROOT / "configs" / f"{name}.yaml")
        validate_config(config)
        assert Path(config["data"]["train_dataset_path"]).is_absolute()


@pytest.mark.parametrize("section,key,value", [
    ("train", "batch_size", 3), ("train", "pipeline_parallel_stages", 2),
    ("general", "device", "cpu"), ("general", "info_src", 1),
    ("model", "num_head", 3), ("train", "save_interval", 0),
    ("model", "num_expert_per_node", 1),
])
def test_invalid_config(section, key, value):
    config = load_config(ROOT / "configs/smoke.yaml")
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_dataset(tmp_path):
    config = load_config(ROOT / "configs/smoke.yaml")
    path = tmp_path / "tokens.npy"
    config["data"]["train_dataset_path"] = str(path)
    np.save(path, np.arange(128, dtype=np.int32))
    validate_config(config, check_data=True)
    np.save(path, np.array([999] * 128))
    with pytest.raises(ValueError, match="token range"):
        validate_config(config, check_data=True)