"""Pipeline model and shared layers used by distributed wrappers."""

from einops import rearrange
from itertools import accumulate
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from implementation.nnfunctions import silu, softmax
import torch.distributed as dist
from collections import defaultdict, deque


class TensorExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sent_tensors, sent_splits, dp_group: dist.ProcessGroup = None):

        dtype = sent_tensors.dtype
        device = sent_tensors.device
        
        send_size = torch.tensor(sent_splits, dtype=torch.int64, device=device) # size for each rank
        recv_size = torch.empty_like(send_size)

        total_expert = len(sent_splits)
        world_size = dp_group.size()
        num_expert_per_node = total_expert//world_size
        dist.all_to_all_single(recv_size, send_size, group=dp_group)
        recv_tensors = torch.empty(int(recv_size.sum().item()), *tuple(sent_tensors.shape[1:]), dtype=dtype, device=device)
        output_splits = recv_size.cpu().tolist()
        sent_splits_size = [int(sum(sent_splits[i:i+num_expert_per_node])) for i in range(0, len(sent_splits), num_expert_per_node)]
        output_splits_size = [int(sum(output_splits[i:i+num_expert_per_node])) for i in range(0, len(output_splits), num_expert_per_node)]

        dist.all_to_all_single(recv_tensors, sent_tensors, input_split_sizes=sent_splits_size, output_split_sizes=output_splits_size, group=dp_group)
        ctx.sent_split = [int(d) for d in sent_splits]
        ctx.recv_split = [int(d) for d in output_splits]
        ctx.dp_group = dp_group
        return recv_tensors, [int(d) for d in output_splits]

    def backward(ctx, d_recv, *_):
        dtype = d_recv.dtype
        device = d_recv.device
        sent_split, recv_split = ctx.sent_split, ctx.recv_split
        dp_group = ctx.dp_group
        total_expert = len(sent_split)
        world_size = dp_group.size()
        num_expert_per_node = total_expert//world_size
        sent_splits_size = [int(sum(sent_split[i:i+num_expert_per_node])) for i in range(0, len(sent_split), num_expert_per_node)]
        output_splits_size = [int(sum(recv_split[i:i+num_expert_per_node])) for i in range(0, len(recv_split), num_expert_per_node)]

        d_sent = torch.empty(sum(sent_split), *tuple(d_recv.shape[1:]), dtype=dtype, device=device)
        dist.all_to_all_single(d_sent, d_recv, input_split_sizes=output_splits_size, output_split_sizes=sent_splits_size, group=dp_group)
        return d_sent, None, None

class MoETokenCollect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sent_tensors, split_sizes: torch.Tensor, ep_group: dist.ProcessGroup = None):
        dtype = sent_tensors.dtype
        device = sent_tensors.device

        # limitation: nccl only behavior, for gool the input shape has to be the same for different ranks
        if ep_group is not None:
            recv_buffer = [torch.empty(split_size, *tuple(sent_tensors.shape[1:]), dtype=dtype, device=device) for split_size in split_sizes]
            print([buf.size() for buf in recv_buffer])
            print(sent_tensors.size())
            dist.all_gather(recv_buffer, sent_tensors, group=ep_group)
        else:
            recv_buffer = [sent_tensors]
        ctx.split_sizes = split_sizes
        ctx.ep_group = ep_group
        return torch.cat(recv_buffer, dim=0)

    def backward(ctx, d_recv, *_):
        size_csum = torch.cumsum(ctx.split_sizes, dim=0)
        start_idx = 0
        if ctx.ep_group is not None:
            ep_rank_this = dist.get_group_rank(ctx.ep_group, dist.get_rank())
        else:
            ep_rank_this = 0
        if ep_rank_this != 0:
            start_idx = size_csum[ep_rank_this-1]
        end_idx = size_csum[ep_rank_this]
        return d_recv[start_idx:end_idx], None, None


class ActivationSync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, communication_group: dist.ProcessGroup = None):

        dist.all_reduce(x, group=communication_group)
        ctx.communication_group = communication_group
        return x

    def backward(ctx, d_recv, *_):
        dist.all_reduce(d_recv, group=ctx.communication_group)
        return d_recv


class LinearLayer(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None, communication_groups: defaultdict = defaultdict(lambda: None)):
        super().__init__()
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
    def __init__(self,  num_embeddings, embedding_dim, device=None, dtype=None, communication_groups: defaultdict = defaultdict(lambda: None)):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(torch.Tensor(size=(num_embeddings, embedding_dim), device=device), 0, 1, -3, 3))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x,:]
    
class RMSNormLayer(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None, communication_groups: defaultdict = defaultdict(lambda: None)):
        super().__init__()
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


class MoELayerMultiSharedInput(nn.Module):
    # Assume EP rank shares input, e.g. in the case of TP
    def __init__(self, d_model: int, d_ff: int, k = 2, num_expert_per_node = 4, device=None, dtype=torch.bfloat16, loss_global_count: bool= False, communication_groups: defaultdict = defaultdict(lambda: None)):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.communication_groups = communication_groups
        self.ep_group = communication_groups.get("EP")
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
        
        # ------------------------------------------------------------
        probs = probs/probs.sum(dim=-1, keepdim=True)
        expert_idx = rearrange(expert_idx, " ... c d -> ... (c d)")
        async_op = self.record_moe_metadata(routing_res, expert_idx)

        expert_indices = []
        split_sizes = []

        for i in range(self.world_size):
            for j in range(self.num_expert_per_node):
                expert_this = i * self.num_expert_per_node + j
                indices = (expert_idx == expert_this).nonzero(as_tuple=True)[0]
                split_sizes.append(len(indices))
                expert_indices.append(indices)    
        
        assignment_order = torch.cat(expert_indices, dim=0)

        split_sizes = torch.tensor(split_sizes, device=x.device, dtype=torch.int32)
        split_sizes_by_rank = rearrange(split_sizes, "(a b) -> a b", b=self.num_expert_per_node).sum(dim=-1)
        if self.ep_group is None:
            ep_rank_this = 0
        else:
            ep_rank_this = dist.get_group_rank(self.communication_groups["EP"], dist.get_rank())
        token_sizes_this_rank = split_sizes[ep_rank_this * self.num_expert_per_node:(ep_rank_this + 1) * self.num_expert_per_node]
        token_indices_this_rank = expert_indices[ep_rank_this * self.num_expert_per_node:(ep_rank_this + 1) * self.num_expert_per_node]
        assignment_order_this = torch.cat(token_indices_this_rank, dim=0)
        expert_input_this = x[assignment_order_this // self.k, :]
        
        
        expert_inputs_offset = torch.cumsum(token_sizes_this_rank, dim=0, dtype=torch.int32)

        # --------------------------------------------------------------
        
        expert_outputs = self.experts(expert_input_this, expert_inputs_offset)



        if async_op is not None:
            async_op.wait()
        # --------------------------------------------------------------

        recv_tensors = MoETokenCollect.apply(expert_outputs, split_sizes_by_rank, self.ep_group)

        out = torch.empty(
            assignment_order.shape[0], *recv_tensors.shape[1:],
            dtype=recv_tensors.dtype, device=recv_tensors.device,
        )
        out[assignment_order, :] = recv_tensors

        out = rearrange(out, " ... (c d) e -> ... c d e", d=self.k)
        out = (out * probs.unsqueeze(dim=-1)).sum(dim=-2)
        out = rearrange(out, "(b c) ... -> b c ...", b = batch_size, c = context_len).to(input_dtype)
        return out

    def get_moe_loss(self):
        if len(self.loss_material_queue) == 0:
            return 0.0
        prob_sum, counts = self.loss_material_queue.popleft()
        experts = self.world_size * self.num_expert_per_node
        assignments = counts.sum().clamp_min(1)
        mean_probs = prob_sum * (self.world_size * self.k / assignments)
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



class MoELayerMulti(nn.Module):
    def __init__(self, d_model: int, d_ff: int, k = 2, num_expert_per_node = 4, device=None, dtype=torch.bfloat16, loss_global_count: bool= False, communication_groups: defaultdict = defaultdict(lambda: None)):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.communication_groups = communication_groups
        self.ep_group = communication_groups.get("EP")
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
        mean_probs = prob_sum * (self.world_size * self.k / assignments)
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
        """Route tokens and prepare grouped expert inputs outside compilation."""
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
        """Redistribute expert outputs and combine tokens outside compilation."""
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

class GroupedPositionWiseFFLayer(nn.Module):
    def __init__(self, num_experts:int, d_model: int, d_ff: int, device=None, dtype=None, communication_groups: defaultdict = defaultdict(lambda: None)):
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
        self.communication_groups = communication_groups
        
    
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
    
    
class ROPELayer(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, communication_groups: defaultdict = defaultdict(lambda: None)):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        seq_idx = torch.arange(0, max_seq_len)
        dim = torch.arange(1, d_k//2+1)
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
    
    
class MultiHeadLayerLL(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, use_rope=True, RoPE: nn.Module = None, communication_groups: defaultdict = defaultdict(lambda: None)):
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
        self.k_proj = LinearLayer(self.d_model, kv_dim, device=device, dtype=dtype)
        self.q_proj = LinearLayer(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = LinearLayer(self.d_model, kv_dim, device=device, dtype=dtype)
        self.output_proj = LinearLayer(self.d_model, self.d_model, device=device, dtype=dtype)
        self.RoPE = None
        self.use_rope=use_rope
        if use_rope:
            if RoPE is None:
                self.RoPE = ROPELayer(10000, d_model//num_heads, 100000)
            else:
                self.RoPE = RoPE
        self.communication_groups = communication_groups

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        # batch, seq_len, d_k
        k = rearrange(self.k_proj(x), "... i (j k) -> ... j i k", k=self.head_dim)
        q = rearrange(self.q_proj(x), "... i (j k) -> ... j i k", k=self.head_dim)
        v = rearrange(self.v_proj(x), "... i (j k) -> ... j i k", k=self.head_dim)

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
        return x_out


class TransformerLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, num_expert_per_node=1, RoPELayer = None, communication_groups: defaultdict = defaultdict(lambda: None), moe_compute_dtype=torch.bfloat16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNormLayer(d_model)
        self.attn = MultiHeadLayerLL(d_model, num_heads, RoPE=RoPELayer)
        self.ln2 = RMSNormLayer(d_model)
        self.ffn = MoELayerMulti(d_model, d_ff, num_expert_per_node = num_expert_per_node, communication_groups=communication_groups, dtype=moe_compute_dtype)
        self.communication_groups = communication_groups
    
    def forward(self, x):
        x = x + ActivationSync.apply(self.attn(self.ln1(x)), self.communication_groups["TP"])
        x = x + ActivationSync.apply(self.ffn(self.ln2(x)), self.communication_groups["TP"])
        return x


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, num_expert_per_node=1, d_ff=None, theta=10000, communication_groups: defaultdict = defaultdict(lambda: None)):
        super().__init__()
        if d_ff == None:
            d_ff = math.ceil(d_model * (8/3)/64)*64
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embeddings = EmbeddingLayer(vocab_size, d_model)
        rope_layer = ROPELayer(theta, d_model//num_heads, context_length)
        self.layers = nn.Sequential(*[TransformerLayer(d_model, num_heads, d_ff, num_expert_per_node=num_expert_per_node, RoPELayer=rope_layer, communication_groups=communication_groups) for i in range(num_layers)])
        self.ln_final = RMSNormLayer(d_model)
        self.lm_head = LinearLayer(d_model, vocab_size)
    
    def forward(self, x:torch.Tensor):
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

class TransformerLMPipelined(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, stage_this, stage_total, num_expert_per_node=1, d_ff=None, theta=10000, communication_groups: defaultdict = defaultdict(lambda: None), moe_compute_dtype=torch.bfloat16):
        super().__init__()
        if d_ff == None:
            d_ff = math.ceil(d_model * (8/3)/64)*64
        self.stage_this = stage_this
        self.stage_total = stage_total
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        if self.stage_this == 0:
            self.token_embeddings = EmbeddingLayer(vocab_size, d_model)
        else:
            self.token_embeddings = None
        rope_layer = ROPELayer(theta, d_model//num_heads, context_length)
        self.layers = nn.Sequential(*[TransformerLayer(d_model, num_heads, d_ff, num_expert_per_node=num_expert_per_node, RoPELayer=rope_layer, communication_groups=communication_groups, moe_compute_dtype=moe_compute_dtype) for i in range(num_layers)])
        if self.stage_this == self.stage_total - 1:
            self.ln_final = RMSNormLayer(d_model)
            self.lm_head = LinearLayer(d_model, vocab_size)
        else:
            self.ln_final = None
            self.lm_head = None
    
    def forward(self, x:torch.Tensor, return_aux_loss=False):
        if self.stage_this == 0:
            x = self.token_embeddings(x)
        if return_aux_loss:
            aux_loss = x.new_zeros((), dtype=torch.float32)
            for layer in self.layers:
                x, layer_loss = layer(x, return_aux_loss=True)
                aux_loss = aux_loss + layer_loss
        else:
            x = self.layers(x)
        if self.stage_this == self.stage_total - 1:
            x = self.ln_final(x)
            x = self.lm_head(x)
        return (x, aux_loss) if return_aux_loss else x


