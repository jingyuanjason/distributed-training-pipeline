"""CPU checks for the consolidated FSDP wrapper's compatibility options."""

import pytest
import torch
from torch import nn

from implementation.distributed import ddp_modules
from implementation.layers import LinearLayer


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("compile_experts", [False, True])
def test_wrapper_options_and_keyword_forwarding(monkeypatch, prefetch, compile_experts):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = LinearLayer(4, 4)
            self.second = LinearLayer(4, 4)
            self.expert = nn.Identity()

        def forward(self, x, return_aux_loss=False):
            # Hook registration is checked below; GPU collectives are not run.
            return (x, x.square().mean()) if return_aux_loss else x

    class Group:
        def size(self):
            return 1

    monkeypatch.setattr(ddp_modules.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(ddp_modules.dist, "get_global_rank", lambda group, rank: rank)
    monkeypatch.setattr(ddp_modules.dist, "broadcast", lambda *args, **kwargs: None)
    compiled = []
    monkeypatch.setattr(nn.Module, "compile", lambda self: compiled.append(self))
    model = Model()
    wrapper = ddp_modules.FSDPWrapperPipelined(
        model, FSDP_communication_group=Group(),
        prefetch_weights=prefetch, compile_experts=compile_experts,
    )
    assert compiled == ([model.expert] if compile_experts else [])
    assert len(model.first._forward_pre_hooks) == (2 if prefetch else 1)
    assert len(model.second._backward_pre_hooks) == (2 if prefetch else 1)
    assert len(wrapper.sharded_params) == 2
    x = torch.randn(2, 4, requires_grad=True)
    assert wrapper(x) is x
    output, auxiliary = wrapper(x, return_aux_loss=True)
    assert output is x
    torch.testing.assert_close(auxiliary, x.square().mean())
    auxiliary.backward()
    assert x.grad is not None


def test_trainers_share_wrapper():
    import distributed_parallel_training_pipelined as training
    import train

    assert train.FSDPWrapperPipelined is ddp_modules.FSDPWrapperPipelined
    assert training.FSDPWrapperPipelined is ddp_modules.FSDPWrapperPipelined