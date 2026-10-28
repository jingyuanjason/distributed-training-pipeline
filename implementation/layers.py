"""Pipeline model and shared layers used by distributed wrappers."""

from einops import rearrange
from itertools import accumulate
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from implementation.distributed.parallelism_utils import AttnCombineGate, AttnDispatchGate, MoETokenCombineGate, MoETokenDispatchGate
from implementation.nnfunctions import silu, softmax
import torch.distributed as dist
from collections import defaultdict, deque
from threading import RLock
from weakref import WeakValueDictionary



class LinearLayer(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.communication_groups = defaultdict(lambda: None)
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype
        std_v = math.sqrt(2/(d_out+ d_in))
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((d_out, d_in), device=device), 0, std_v, -3 * std_v, 3 * std_v))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expert weights are not FSDP-sharded, so cast them transiently to the
        # activation dtype. FSDP-managed weights already match x.dtype.
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return einsum("ab,...b -> ...a", weight, x)


class EmbeddingLayer(nn.Module):
    def __init__(self,  num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.communication_groups = defaultdict(lambda: None)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(torch.Tensor(size=(num_embeddings, embedding_dim), device=device), 0, 1, -3, 3))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x,:]
    
class RMSNormLayer(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.communication_groups = defaultdict(lambda: None)
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(size=[d_model], device=device, requires_grad=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(torch.mean(x_float**2, dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)
    
    
class _ROPELayerMeta(type(nn.Module)):
    _instances = WeakValueDictionary()
    _lock = RLock()

    def __call__(cls, theta: float, d_k: int, max_seq_len: int, device=None):
        device = torch.get_default_device() if device is None else torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (cls, theta, d_k, max_seq_len, device)
        with cls._lock:
            instance = cls._instances.get(key)
            # A caller may have moved the shared module since construction.
            if instance is None or instance.ropemat.device != device:
                instance = super().__call__(theta, d_k, max_seq_len, device=device)
                cls._instances[key] = instance
            return instance


class ROPELayer(nn.Module, metaclass=_ROPELayerMeta):
    """Weakly cached singleton per configuration and construction device.

    All references share module mutations, including .to() calls and hooks.
    A new construction replaces a cached module moved to a different device.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.communication_groups = defaultdict(lambda: None)
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        seq_idx = torch.arange(0, max_seq_len, device=device)
        dim = torch.arange(1, d_k//2+1, device=device)
        dim = theta ** ((2 - 2 * dim) / d_k)
        seq_all = einsum("i,j->ij", seq_idx, dim)
        t_a = torch.stack([torch.cos(seq_all), -torch.sin(seq_all)], dim=-1) # seq_len, d_k/2, 2
        t_b = torch.stack([torch.sin(seq_all), torch.cos(seq_all)], dim=-1) # seq_len, d_k/2, 2
        self.register_buffer("ropemat", torch.stack([t_a, t_b], dim=-1), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # batch, seq_len, d_k
        mat_this = self.ropemat[token_positions, :,:,:] # ...seq_len, d_k/2, 2, 2
        x = rearrange(x, "...(a b)->... a b", b=2)
        return einsum("...k,...kl->...l", x.float(), mat_this.float()).flatten(start_dim=-2).to(x.dtype)


class GroupedPositionWiseFFLayer(nn.Module):
    def __init__(self, num_experts:int, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.dff = d_ff
        self.device = device
        self.dtype = dtype
        self.num_experts = num_experts
        std_v = math.sqrt(2 / (d_model + d_ff))
        self.w1 = nn.Parameter(nn.init.trunc_normal_(torch.empty((num_experts, d_model, d_ff), device=device, dtype=dtype), 0, std_v, -3 * std_v, 3 * std_v))
        self.w2 = nn.Parameter(nn.init.trunc_normal_(torch.empty((num_experts, d_model, d_ff), device=device, dtype=dtype), 0, std_v, -3 * std_v, 3 * std_v))
        self.w3 = nn.Parameter(nn.init.trunc_normal_(torch.empty((num_experts, d_ff, d_model), device=device, dtype=dtype), 0, std_v, -3 * std_v, 3 * std_v))
        self.communication_groups = defaultdict(lambda: None)
        
    
    def forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x1 = F.grouped_mm(
            x,
            self.w1.to(x.dtype),
            offs=offsets,
        )
        x1 = silu(x1)
        x2 = F.grouped_mm(
            x,
            self.w2.to(x.dtype),
            offs=offsets,
        )
        x1 = x1 * x2
        x_out = F.grouped_mm(
            x1,
            self.w3.to(x.dtype),
            offs=offsets,
        )
        return x_out
    

class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, k = 2, num_experts = 4, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.communication_groups = defaultdict(lambda: None)
        self.k = k
        self.ep_group = self.communication_groups["EP"]
        self.world_size = 1
        self._ep_registered = False
        if num_experts < 1 or not 1 <= k <= num_experts:
            raise ValueError("Require positive expert count and 1 <= k <= total experts")
        if dtype != torch.bfloat16:
            raise ValueError("Grouped MoE requires torch.bfloat16 compute dtype")
        # Keep master parameters (and their accumulated gradients) in FP32;
        # GroupedPositionWiseFFLayer casts weights to the activation dtype.
        self.experts = GroupedPositionWiseFFLayer(num_experts, d_model, d_ff, device=device, dtype=torch.float32)
        self.router = LinearLayer(d_model, num_experts, device=device, dtype=dtype)
        self.num_experts = num_experts
        self.loss_material_queue = deque() # record the material for calculating loss, use queue for pipeline parallel microbatch handling

    
    def forward(self, x: torch.Tensor):
        # x B, S, D
        
        if self.ep_group is None:
            ep_rank_this = 0
        else:
            ep_rank_this = dist.get_group_rank(self.communication_groups["EP"], dist.get_rank())
        
        input_dtype = x.dtype
        batch_size, context_len = x.shape[:2]
        x = rearrange(x, "b c ... -> (b c) ...")
        # Routing decisions and auxiliary statistics are sensitive to rounding.
        logits = self.router(x.float())
        x = x.to(self.dtype)
        
        routing_res: torch.Tensor = softmax(logits, dim_sum=-1)
        probs, expert_idx = routing_res.topk(self.k, dim=-1)
        
        # ------------------------------------------------------------
        probs = probs/probs.sum(dim=-1, keepdim=True)
        expert_idx = rearrange(expert_idx, " ... c d -> ... (c d)")
        async_op = self.record_moe_metadata(routing_res, expert_idx)

        expert_indices_all_experts = []
        split_sizes_all_experts = []

        for i in range(self.world_size):
            size_this = []
            expert_indices_this = []
            for j in range(self.experts.num_experts):
                expert_this = i * self.experts.num_experts + j
                indices = (expert_idx == expert_this).nonzero(as_tuple=True)[0]
                size_this.append(len(indices))
                expert_indices_this.append(indices)  
            expert_indices_all_experts.append(expert_indices_this)
            split_sizes_all_experts.append(size_this)

        assignment_order = torch.cat([item for group in expert_indices_all_experts for item in group], dim=0)
        split_sizes_all_experts = torch.tensor(split_sizes_all_experts, device=x.device, dtype=torch.int32)
        expert_input_this = MoETokenDispatchGate.apply(x, expert_indices_all_experts, split_sizes_all_experts, self.k, self.communication_groups["EP"])

        # --------------------------------------------------------------
        expert_inputs_offset = torch.cumsum(split_sizes_all_experts[ep_rank_this], dim=0, dtype=torch.int32)
        expert_outputs = self.experts(expert_input_this, expert_inputs_offset)
        if async_op is not None:
            async_op.wait()
        # --------------------------------------------------------------

        recv_tensors = MoETokenCombineGate.apply(expert_outputs, split_sizes_all_experts, self.ep_group)

        out = torch.empty(
            assignment_order.shape[0], *recv_tensors.shape[1:],
            dtype=recv_tensors.dtype, device=recv_tensors.device,
        )
        out[assignment_order, :] = recv_tensors

        out = rearrange(out, " ... (c d) e -> ... c d e", d=self.k)
        out = (out * probs.to(out.dtype).unsqueeze(dim=-1)).sum(dim=-2)
        out = rearrange(out, "(b c) ... -> b c ...", b = batch_size, c = context_len).to(input_dtype)
        return out

    def get_moe_loss(self, loss_global_count=False):
        if len(self.loss_material_queue) == 0:
            return 0.0
        prob_sum, counts = self.loss_material_queue.popleft()
        
        if loss_global_count and self.ep_group is not None:
            dist.all_reduce(counts, group=self.ep_group)
        
        experts = self.num_experts
        assignments = counts.sum().clamp_min(1)
        count_world_size = self.world_size if loss_global_count else 1
        mean_probs = prob_sum * (count_world_size * self.k / assignments)
        aux_loss = experts * (counts / assignments * mean_probs).sum()
        return aux_loss

    def record_moe_metadata(self, probs, expert_idx):
        total_experts = self.num_experts
        counts = torch.bincount(expert_idx.flatten(), minlength=total_experts).float()
        prob_sum = probs.sum(dim=0) # num_experts
        self.loss_material_queue.append((prob_sum, counts))

    
    def register_ep(self, ep_communication_group: dist.ProcessGroup):
        if self._ep_registered:
            raise RuntimeError("Expert parallelism has already been registered")
        world_size = ep_communication_group.size()
        if self.num_experts % world_size != 0:
            raise ValueError("Expert count must be divisible by EP group size")
        self.communication_groups["EP"] = ep_communication_group
        ep_rank = dist.get_rank(ep_communication_group)
        num_local_experts = self.num_experts // world_size

        with torch.no_grad():
            for weight in (self.experts.w1, self.experts.w2, self.experts.w3):
                weight.data = weight.detach()[ep_rank * num_local_experts:(ep_rank+1) * num_local_experts].clone()

        self.ep_group = ep_communication_group
        self.world_size = world_size
        self.experts.num_experts = num_local_experts
        self._ep_registered = True

    
    
class MultiHeadLayerLL(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta=10000, device=None, dtype=None, use_rope=True):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_heads % 4 != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by 4 for 4:1 grouped-query attention")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_heads // 4
        self.head_dim = d_model // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_sizes = (self.d_model, kv_dim, kv_dim)
        self.qkv_proj = LinearLayer(self.d_model, sum(self.qkv_sizes), device=device, dtype=dtype)
        self.output_proj = LinearLayer(self.d_model, self.d_model, device=device, dtype=dtype)
        self.RoPE = None
        self.use_rope=use_rope
        if use_rope:
            self.RoPE = ROPELayer(theta, d_model//num_heads, 100000, device=device)

        self.communication_groups = defaultdict(lambda: None)
        self._tp_registered = False

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        
        x = AttnDispatchGate.apply(x, self.communication_groups["TP"])
        # batch, seq_len, d_k
        q, k, v = self.qkv_proj(x).split(self.qkv_sizes, dim=-1)
        q = rearrange(q, "... i (j k) -> ... j i k", k=self.head_dim)
        k = rearrange(k, "... i (j k) -> ... j i k", k=self.head_dim)
        v = rearrange(v, "... i (j k) -> ... j i k", k=self.head_dim)

        if self.use_rope:
            if token_positions is None:
                len_seq = x.shape[1]
                token_positions = torch.arange(0, len_seq, device=x.device)
            k = self.RoPE(k, token_positions)
            q = self.RoPE(q, token_positions)
        
        # Use the causal fast path by default; explicit masks retain their semantics.
        res = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=mask is None, enable_gqa=True,
        )
        res = rearrange(res, "... i j k ->... j (i k)")
        x_out = self.output_proj(res)
        x_out = AttnCombineGate.apply(x_out, self.communication_groups["TP"])
        return x_out

    def register_tp(self, tp_communication_group: dist.ProcessGroup):
        
        if self._tp_registered:
            raise RuntimeError("Tensor parallelism has already been registered")
        
        
        tp_rank = dist.get_rank(tp_communication_group)
        tp_group_rank_list = dist.get_process_group_ranks(tp_communication_group)
        tp_group_size = len(tp_group_rank_list)
        assert self.num_kv_heads % tp_group_size == 0, "attention head cannot be splited evenly"
        head_dim = self.head_dim
        num_head = self.num_heads
        num_head_per_tp_rank = num_head // tp_group_size
        num_kv_head = self.num_kv_heads
        num_kv_head_per_tp_rank = num_kv_head // tp_group_size
        local_sizes = (head_dim * num_head_per_tp_rank,
                       head_dim * num_kv_head_per_tp_rank,
                       head_dim * num_kv_head_per_tp_rank)
        qkv_weights = self.qkv_proj.weight.detach().split(self.qkv_sizes, dim=0)
        self.qkv_proj.weight.data = torch.cat([
            weight[tp_rank * size:(tp_rank + 1) * size]
            for weight, size in zip(qkv_weights, local_sizes)
        ], dim=0)
        self.qkv_sizes = local_sizes
        self.qkv_proj.d_out = sum(local_sizes)

        output_proj_weight = self.output_proj.weight.detach()
        self.output_proj.weight.data = output_proj_weight[:, tp_rank * (head_dim * num_head_per_tp_rank):(tp_rank + 1) * (head_dim * num_head_per_tp_rank)].clone()
        self.output_proj.d_in = local_sizes[0]
        self.communication_groups["TP"] = tp_communication_group
        self._tp_registered = True

class TransformerLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, theta = 10000, moe_compute_dtype=torch.bfloat16, num_experts=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNormLayer(d_model)
        self.attn = MultiHeadLayerLL(d_model, num_heads, theta=theta)
        self.ln2 = RMSNormLayer(d_model)
        self.ffn = MoELayer(d_model, d_ff, num_experts=num_experts, dtype=moe_compute_dtype)
        self.communication_groups = defaultdict(lambda: None)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x




class TransformerLMPipelined(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers_model, num_heads, pipeline_stage_this, pipeline_stage_total, d_ff=None, theta=10000, moe_compute_dtype=torch.bfloat16, num_expert_model=4, compute_dtype=torch.float32):
        super().__init__()
        if compute_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("compute_dtype must be torch.float32 or torch.bfloat16")
        self.compute_dtype = compute_dtype
        self.communication_groups = defaultdict(lambda: None)
        if d_ff == None:
            d_ff = math.ceil(d_model * (8/3)/64)*64
            
        assert num_layers_model % pipeline_stage_total == 0, "layer number must be divided by stage number"
        self.pipeline_stage_total = pipeline_stage_total
        self.pipeline_stage_this = pipeline_stage_this
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers_model = num_layers_model
        self.num_layers_stage = num_layers_model // pipeline_stage_total
        if self.pipeline_stage_this == 0:
            self.token_embeddings = EmbeddingLayer(vocab_size, d_model)
        else:
            self.token_embeddings = None
        self.num_expert_model = num_expert_model
        self.layers = nn.Sequential(*[TransformerLayer(d_model, num_heads, d_ff, theta, moe_compute_dtype=moe_compute_dtype, num_experts=num_expert_model) for _ in range(self.num_layers_stage)])
        if self.pipeline_stage_this == self.pipeline_stage_total - 1:
            self.ln_final = RMSNormLayer(d_model)
            self.lm_head = LinearLayer(d_model, vocab_size)
        else:
            self.ln_final = None
            self.lm_head = None
    
    def forward(self, x:torch.Tensor):
        if self.pipeline_stage_this == 0:
            x = self.token_embeddings(x)
        # Cast activations, never model parameters: linear layers cast their
        # FP32 weights transiently, preserving FP32 gradients and Adam states.
        x = x.to(self.compute_dtype)
        x = self.layers(x)
        if self.pipeline_stage_this == self.pipeline_stage_total - 1:
            x = self.ln_final(x)
            x = self.lm_head(x)
        return x


