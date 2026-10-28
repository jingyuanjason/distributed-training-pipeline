
from collections import defaultdict, deque

from einops import rearrange
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from implementation.distributed.parallelism_utils import TensorExchange
from implementation.layers import GroupedPositionWiseFFLayer, LinearLayer
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
    
class MoELayerMulti(nn.Module):
    def __init__(self, d_model: int, d_ff: int, k = 2, num_expert_per_node = 4, device=None, dtype=torch.bfloat16, loss_global_count: bool= False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.communication_groups = defaultdict(lambda: None)
        if self.ep_group is None:
            self.world_size = 1
        else:
            self.world_size = self.ep_group.size()
        if num_expert_per_node < 1 or not 1 <= k <= self.world_size * num_expert_per_node:
            raise ValueError("Require positive expert count and 1 <= k <= total experts")
        if dtype != torch.bfloat16:
            raise ValueError("Grouped MoE requires torch.bfloat16 compute dtype")
        self.k = k
        self.num_expert_per_node = num_expert_per_node
        self.experts = GroupedPositionWiseFFLayer(num_expert_per_node, d_model, d_ff, device=device, dtype=dtype)
        self.router = LinearLayer(d_model, self.world_size * num_expert_per_node, device=device, dtype=dtype)
        self.loss_global_count = loss_global_count
        self.loss_material_queue = deque() # record the material for calculating loss, use queue for pipeline parallel microbatch handling

    
    def forward(self, x: torch.Tensor):
        # x B, S, D
        input_dtype = x.dtype
        x = x.to(self.dtype)
        batch_size, context_len = x.shape[:2]
        x = rearrange(x, "b c ... -> (b c) ...")
        logits = self.router(x)
        routing_res: torch.Tensor = softmax(logits, dim_sum=-1)
        probs, expert_idx = routing_res.topk(self.k, dim=-1)

        (expert_input, expert_inputs_offset, expert_split_sizes,
         recv_splits, assignment_order, probs) = self._dispatch_tokens(
            x, logits, probs, expert_idx,
        )
        async_op = self.record_moe_metadata(routing_res, expert_idx)
        expert_outputs = self.experts(expert_input, expert_inputs_offset)
        if async_op is not None:
            async_op.wait()
        out = self._combine_tokens(
            expert_outputs, expert_split_sizes, recv_splits, assignment_order,
            probs, batch_size, context_len, input_dtype,
        )
        return out

    def get_moe_loss(self):
        if len(self.loss_material_queue) == 0:
            return 0.0
        prob_sum, counts = self.loss_material_queue.popleft()
        experts = self.world_size * self.num_expert_per_node
        assignments = counts.sum().clamp_min(1)
        count_world_size = self.world_size if self.loss_global_count else 1
        mean_probs = prob_sum * (count_world_size * self.k / assignments)
        aux_loss = experts * (counts / assignments * mean_probs).sum()
        return aux_loss

    def record_moe_metadata(self, probs, expert_idx):
        total_experts = self.world_size * self.num_expert_per_node
        counts = torch.bincount(expert_idx.flatten(), minlength=total_experts).float()
        prob_sum = probs.sum(dim=0) # num_experts
        self.loss_material_queue.append((prob_sum, counts))
        if self.loss_global_count:
            async_op = dist.all_reduce(counts, group=self.ep_group, async_op=True)
            return async_op
        return None


    @torch.compiler.disable
    def _dispatch_tokens(self, x, probs, expert_idx):
        probs = probs/probs.sum(dim=-1, keepdim=True)
        expert_idx = rearrange(expert_idx, " ... c d -> ... (c d)")

        expert_indices = []
        split_sizes = []
        for j in range(self.world_size * self.num_expert_per_node):

            indices = (expert_idx == j).nonzero(as_tuple=True)[0]
            split_sizes.append(len(indices))
            expert_indices.append(indices)

        assignment_order = torch.cat(expert_indices, dim=0)
        sent_tensor = x[assignment_order // self.k, :]

        if self.world_size == 1:
            recv_tensors, recv_splits = sent_tensor, split_sizes
        else:
            recv_tensors, recv_splits = TensorExchange.apply(sent_tensor, split_sizes, self.ep_group)

        chunks = torch.split(recv_tensors, recv_splits, dim=0)
        expert_inputs = []
        expert_inputs_size = []
        expert_split_sizes = []
        for i in range(self.num_expert_per_node):
            expert_inputs.extend(chunks[i::self.num_expert_per_node])
            expert_inputs_size.append(sum(x.shape[0] for x in chunks[i::self.num_expert_per_node]))
            expert_split_sizes.extend(x.shape[0] for x in chunks[i::self.num_expert_per_node])

        expert_input = torch.cat(expert_inputs, dim=0)
        expert_inputs_offset = torch.tensor(
            expert_inputs_size, device=expert_input.device, dtype=torch.int32,
        ).cumsum(dim=0, dtype=torch.int32)

        return (expert_input, expert_inputs_offset, expert_split_sizes,
                recv_splits, assignment_order, probs)

    @torch.compiler.disable
    def _combine_tokens(self, expert_outputs, expert_split_sizes, recv_splits,
                        assignment_order, probs, batch_size, context_len, input_dtype):
        expert_outputs = torch.split(expert_outputs, expert_split_sizes)
        expert_outputs_rearranged = []
        for rank_idx in range(self.world_size):
            expert_outputs_rearranged.extend([expert_outputs[expert_idx * self.world_size + rank_idx] for expert_idx in range(self.num_expert_per_node)])


        sent_tensors = torch.cat(expert_outputs_rearranged, dim=0)
        if self.world_size == 1:
            recv_tensors = sent_tensors
        else:
            recv_tensors, _ = TensorExchange.apply(sent_tensors, recv_splits, self.dp_group)
                
        out = torch.empty(
            assignment_order.shape[0], *recv_tensors.shape[1:],
            dtype=recv_tensors.dtype, device=recv_tensors.device,
        )
        out[assignment_order, :] = recv_tensors

        out = rearrange(out, " ... (c d) e -> ... c d e", d=self.k)
        out = (out * probs.unsqueeze(dim=-1)).sum(dim=-2)
        out = rearrange(out, "(b c) ... -> b c ...", b = batch_size, c = context_len).to(input_dtype)
        return out
    
    
class TransformerLM(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff=None, theta=10000):
        super().__init__()
        self.communication_groups = defaultdict(lambda: None)
        if d_ff == None:
            d_ff = math.ceil(d_model * (8/3)/64)*64
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        
        self.token_embeddings = EmbeddingLayer(vocab_size, d_model)
        self.layers = nn.Sequential(*[TransformerLayer(d_model, num_heads, d_ff, theta) for i in range(num_layers)])
        self.ln_final = RMSNormLayer(d_model)
        self.lm_head = LinearLayer(d_model, vocab_size)
    
    def forward(self, x:torch.Tensor):
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x