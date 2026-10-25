
from einops import rearrange
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from implementation.nnfunctions import silu, softmax
import torch.distributed as dist
from implementation.kernels.attention import FlashAttentionTritonFunc

class Attention_Simple(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, k, q, v, mask):
        res = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return res
    
class Attention_Flash(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, k, q, v, mask):
        res = FlashAttentionTritonFunc().apply(k, q, v)
        return res

class PositionWiseFFLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.dff = d_ff
        self.device = device
        self.dtype = dtype
     #   self.w1 = nn.Parameter(torch.nn.init.trunc_normal_(torch.Tensor(size=(self.dff, self.d_model), device=device), 0, 1, -3, 3))
     #   self.w2 = nn.Parameter(torch.nn.init.trunc_normal_(torch.Tensor(size=(self.d_model, self.dff), device=device), 0, 1, -3, 3))
     #   self.w3 = nn.Parameter(torch.nn.init.trunc_normal_(torch.Tensor(size=(self.dff, self.d_model), device=device), 0, 1, -3, 3))
        self.w1 = LinearLayer(self.d_model, self.dff, device=device, dtype=dtype)
        self.w2 = LinearLayer(self.dff, self.d_model, device=device, dtype=dtype)
        self.w3 = LinearLayer(self.d_model, self.dff, device=device, dtype=dtype)
      #  self.silu = SiLuLayer()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))