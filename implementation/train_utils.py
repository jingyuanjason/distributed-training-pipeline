import os

import numpy as np
import torch
import torch.nn as nn
from typing import IO, Any, BinaryIO

def load_dataset(path: str, vocab_size: int | None = None, validate: bool = False):
    dataset = np.load(path, mmap_mode="r")

    if dataset.ndim != 1:
        raise ValueError(f"Expected a 1D token dataset, got shape {dataset.shape}")
    if not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError(f"Expected an integer token dataset, got dtype {dataset.dtype}")

    if validate:
        if vocab_size is None:
            raise ValueError("vocab_size must be provided when validate=True")
        dataset_min = int(dataset.min())
        dataset_max = int(dataset.max())
        if dataset_min < 0 or dataset_max >= vocab_size:
            raise ValueError(
                f"Dataset token range [{dataset_min}, {dataset_max}] is incompatible "
                f"with vocab_size={vocab_size}"
            )

    return dataset


def get_batch(
    tokens,
    batch_size,
    context_length,
    device,
    serial_sampling: bool = False,
    start_idx: int = 0,
    rng: np.random.Generator | None = None,
    vocab_size: int | None = None,
    validate: bool = False,
):
    if rng is None:
        rng = np.random.default_rng()

    if tokens.ndim != 1:
        raise ValueError(f"Expected a 1D token dataset, got shape {tokens.shape}")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise TypeError(f"Expected an integer token dataset, got dtype {tokens.dtype}")
    if tokens.size <= context_length:
        raise ValueError(
            f"Dataset length {tokens.size} must be greater than context_length {context_length}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if serial_sampling:
        start_indices = np.arange(
            start_idx,
            min(start_idx + context_length * batch_size, tokens.size - context_length),
            context_length,
            dtype=np.int64,
        )
        batch_size = len(start_indices)
    else:
        start_indices = rng.integers(
            low=start_idx,
            high=tokens.size - context_length,
            size=batch_size,
            dtype=np.int64,
        )

    train_data_np = np.empty((batch_size, context_length), dtype=np.int64)
    train_target_np = np.empty((batch_size, context_length), dtype=np.int64)
    for i, start in enumerate(start_indices):
        train_data_np[i] = tokens[start : start + context_length]
        train_target_np[i] = tokens[start + 1 : start + context_length + 1]

    if validate:
        if vocab_size is None:
            raise ValueError("vocab_size must be provided when validate=True")
        data_min = int(train_data_np.min())
        data_max = int(train_data_np.max())
        target_min = int(train_target_np.min())
        target_max = int(train_target_np.max())
        if data_min < 0 or data_max >= vocab_size:
            raise ValueError(
                f"Invalid input token range [{data_min}, {data_max}] for vocab_size={vocab_size}"
            )
        if target_min < 0 or target_max >= vocab_size:
            raise ValueError(
                f"Invalid target token range [{target_min}, {target_max}] for vocab_size={vocab_size}"
            )

    train_data = torch.from_numpy(train_data_np)
    train_target = torch.from_numpy(train_target_np)

    device = torch.device(device)
    if device.type == "cuda":
        train_data = train_data.pin_memory()
        train_target = train_target.pin_memory()

    return (
        train_data.to(device, non_blocking=True),
        train_target.to(device, non_blocking=True),
    )

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    
    checkpoint = {"model.state_dict": model.state_dict(),
                  "optimizer.state_dict": optimizer.state_dict(),
                  "iteration": iteration}
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer = None):
    
    checkpoint = dict(torch.load(src))
    model.load_state_dict(checkpoint["model.state_dict"])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer.state_dict"])
    return checkpoint["iteration"]