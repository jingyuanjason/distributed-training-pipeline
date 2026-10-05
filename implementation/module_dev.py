from implementation.layers import *
from torch import nn


class ModuleStack(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
    #    self.attn1 = MultiHeadLayerLL(d_in, 32, device=device, dtype=dtype, use_rope=False)
        self.ffn1 = PositionWiseFFLayer(d_in, 14336, device=device, dtype=dtype)
    #    self.attn2 = MultiHeadLayerLL(d_in, 32, device=device, dtype=dtype, use_rope=False)
        self.ffn2 = PositionWiseFFLayer(d_in, 14336, device=device, dtype=dtype)
    #    self.attn3 = MultiHeadLayerLL(d_in, 32, device=device, dtype=dtype, use_rope=False)
        self.ffn3 = PositionWiseFFLayer(d_in, 14336, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expert weights are not FSDP-sharded, so cast them transiently to the
        # activation dtype. FSDP-managed weights already match x.dtype.
    #    x = self.attn1(x)
        x = self.ffn1(x)
    #    x = self.attn2(x)
        x = self.ffn2(x)
    #    x = self.attn3(x)
        x = self.ffn3(x)
        return x