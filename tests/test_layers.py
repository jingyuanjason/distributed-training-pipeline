"""Regression checks for the consolidated layer module."""

from pathlib import Path

import torch

from implementation import layers
from implementation.distributed import ddp_modules


def test_consolidated_dependencies():
    root = Path(layers.__file__).parent
    assert not (root / "model.py").exists()
    assert not (root / "pipelined_layers.py").exists()
    assert not (root / "distributed" / "wrappers.py").exists()
    for name in ("LinearLayer", "EmbeddingLayer", "RMSNormLayer"):
        assert getattr(ddp_modules, name) is getattr(layers, name)
    for name in ("Linear", "Embedding", "RMSNorm"):
        assert not hasattr(layers, name)
        assert not hasattr(ddp_modules, name)
    assert not hasattr(ddp_modules, "FSDPWrapper")
    assert not hasattr(ddp_modules, "sync_grad_reduce_scatter")


def test_pipeline_layers_forward_backward():
    embedding = layers.EmbeddingLayer(8, 4)
    linear = layers.LinearLayer(4, 3)
    norm = layers.RMSNormLayer(3)
    tokens = torch.tensor([[0, 2, 5]])
    x = embedding(tokens)
    torch.testing.assert_close(x, embedding.weight[tokens])
    projected = linear(x)
    torch.testing.assert_close(projected, x @ linear.weight.T)
    expected = projected * torch.rsqrt(projected.square().mean(-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(norm(projected), expected * norm.weight)
    norm(projected).square().sum().backward()
    for module in (embedding, linear, norm):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()


def test_grouped_moe_optional_auxiliary_loss(monkeypatch):
    # CPU stand-in for the GPU grouped-mm kernel; exercise actual routing.
    def grouped_forward(self, x, offsets):
        return x

    monkeypatch.setattr(layers.GroupedPositionWiseFFLayer, "forward", grouped_forward)
    model = layers.MoELayerMulti(4, 8, num_expert_per_node=2, dtype=torch.bfloat16)
    x = torch.randn(2, 3, 4)
    plain = model(x)
    output, auxiliary = model(x, return_aux_loss=True)
    torch.testing.assert_close(output, plain)
    assert auxiliary.ndim == 0 and torch.isfinite(auxiliary)
    auxiliary.backward()
    assert model.router.weight.grad is not None