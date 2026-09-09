"""Numerical regression tests for the paired 1F1B schedule."""

from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from implementation.pipeline import pipelined_train, pipelined_train_overlap
from torch import nn


class PipelineStage(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.accumulated = {}

    def forward(self, x):
        return self.layer(x)

    def clear_grad_accumulators(self):
        self.accumulated.clear()

    def accumulate_full_gradients(self):
        for p in self.parameters():
            if p not in self.accumulated:
                self.accumulated[p] = p.grad.clone()
            else:
                self.accumulated[p].add_(p.grad)
            p.grad = None

    def reduce_accumulated_gradients(self):
        for p, grad in self.accumulated.items():
            p.grad = grad


def _check_pipeline(rank, stages, microbatches, rendezvous, schedule="paired", backend="gloo"):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    if stages > 1:
        dist.init_process_group(
            backend, init_method=rendezvous, rank=rank, world_size=stages,
            timeout=timedelta(seconds=30),
        )
    try:
        torch.manual_seed(42)
        reference = nn.Sequential(*[nn.Sequential(nn.Linear(4, 4), nn.Tanh()) for _ in range(stages)]).to(device)
        stage = PipelineStage(deepcopy(reference[rank]))
        train = pipelined_train_overlap
        optimizer = torch.optim.SGD(stage.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        spec = [2 * microbatches, 3, 4]
        for _ in range(2):
            x = torch.randn(*spec).to(device)
            labels = torch.randn(*spec).to(device)
            reference_optimizer.zero_grad()
            expected_loss = nn.functional.mse_loss(reference(x), labels)
            expected_loss.backward()
            reference_optimizer.step()
            loss = train(
                stage, optimizer, x if rank == 0 else None,
                labels if rank == stages - 1 else None, spec, microbatches,
                rank, nn.functional.mse_loss, dtype=torch.float32, device=device,
                pp_group=dist.group.WORLD if stages > 1 else None,
            )
            assert spec == [2 * microbatches, 3, 4]
            if rank == stages - 1:
                torch.testing.assert_close(loss, expected_loss)
            for actual, expected in zip(stage.layer.parameters(), reference[rank].parameters()):
                torch.testing.assert_close(actual, expected)
    finally:
        if stages > 1:
            dist.destroy_process_group()


@pytest.mark.parametrize("stages,microbatches", [(1, 1), (1, 4), (2, 1), (2, 4), (3, 1), (3, 2), (3, 5)])
@pytest.mark.parametrize("schedule", ["paired"])
def test_pipeline_matches_reference(stages, microbatches, schedule, tmp_path):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    if stages == 1:
        _check_pipeline(0, stages, microbatches, rendezvous, schedule)
    else:
        mp.spawn(_check_pipeline, args=(stages, microbatches, rendezvous, schedule), nprocs=stages, join=True)


@pytest.mark.parametrize("microbatches", [1, 4])
def test_legacy_schedule_matches_reference(microbatches, tmp_path):
    # The legacy schedule requires an initialized default group even for one
    # stage and mutates x_spec; preserve those semantics during this refactor.
    dist.init_process_group("gloo", init_method=(tmp_path / "legacy").as_uri(), rank=0, world_size=1)
    try:
        torch.manual_seed(42)
        reference = nn.Linear(4, 4)
        stage = PipelineStage(deepcopy(reference))
        optimizer = torch.optim.SGD(stage.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        for _ in range(2):
            x = torch.randn(2 * microbatches, 3, 4)
            labels = torch.randn_like(x)
            reference_optimizer.zero_grad()
            expected_loss = nn.functional.mse_loss(reference(x), labels)
            expected_loss.backward()
            reference_optimizer.step()
            spec = list(x.shape)
            loss = pipelined_train(
                stage, optimizer, x, labels, spec, microbatches, 0,
                nn.functional.mse_loss, dtype=torch.float32, device="cpu",
                pp_group=dist.group.WORLD,
            )
            torch.testing.assert_close(loss, expected_loss)
            assert spec == [2, 3, 4]
            for actual, expected in zip(stage.layer.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()


