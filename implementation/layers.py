from einops import rearrange
import torch
import math
import torch.nn as nn
from torch import einsum
from implementation.nnfunctions import silu, softmax
import torch.distributed as dist
from implementation.kernels.attention import FlashAttentionTritonFunc


def exchange_tensors(sent_tensors: list[torch.Tensor], sent_splits: list[int]):

    dtype = sent_tensors.dtype
    device = sent_tensors.device
    
    send_size = torch.tensor(sent_splits, dtype=torch.int64, device=device) # size for each rank
    recv_size = torch.empty_like(send_size)

    dist.all_to_all_single(recv_size, send_size)
    
    recv_tensors = torch.empty(int(recv_size.sum().item()), *tuple(sent_tensors.shape[1:]), dtype=dtype, device=device)
    output_splits = recv_size.cpu().tolist()
    dist.all_to_all_single(recv_tensors, sent_tensors, input_split_sizes=[int(d) for d in sent_splits], output_split_sizes=[int(d) for d in output_splits])
    return recv_tensors, output_splits



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



class LinearLayer(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype
        std_v = math.sqrt(2/(d_out+ d_in))
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(torch.Tensor(size=(d_out, d_in), device=device), 0, std_v, -3 * std_v, 3 * std_v))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expert weights are not FSDP-sharded, so cast them transiently to the
        # activation dtype. FSDP-managed weights already match x.dtype.
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return einsum("ab,...b -> ...a", weight, x)


class EmbeddingLayer(nn.Module):
    def __init__(self,  num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
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



class MoELayerMulti(nn.Module):
    def __init__(self, d_model: int, d_ff: int, k = 2, num_expert_per_node = 4, device=None, dtype=torch.bfloat16, dp_group: dist.ProcessGroup = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dp_group = dp_group
        if dp_group is None:
            self.world_size = 1
        else:
            self.world_size = dp_group.size()
        self.k = k
        self.num_expert_per_node = num_expert_per_node
        self.experts = GroupedPositionWiseFFLayer(num_expert_per_node, d_model, d_ff, device=device, dtype=dtype)
        self.router = LinearLayer(d_model, self.world_size * num_expert_per_node, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x B, S, D
        input_dtype = x.dtype
        x = x.to(self.dtype)
        batch_size, context_len = x.shape[:2]
        x = rearrange(x, "b c ... -> (b c) ...")
        logits = self.router(x)
        routing_res: torch.Tensor = softmax(logits, dim_sum=-1)
        probs, expert_idx = routing_res.topk(self.k, dim=-1)
        probs = probs/probs.sum(dim=-1, keepdim=True)
        len_each = expert_idx.shape[0]
        expert_idx = rearrange(expert_idx, " ... c d -> ... (c d)")
        out = torch.empty(expert_idx.shape[0], *x.shape[1:], dtype=x.dtype, device=x.device)

   
        expert_indices = []
        split_sizes = []
        for j in range(self.world_size * self.num_expert_per_node):

            indices = (expert_idx == j).nonzero(as_tuple=True)[0]
            split_sizes.append(len(indices))
            expert_indices.append(indices)
        
        sent_tensor = x[torch.cat(expert_indices, dim=0) % len_each, :]


        recv_tensors, recv_splits = TensorExchange().apply(sent_tensor, split_sizes, self.dp_group)

        chunks = torch.split(recv_tensors, recv_splits, dim=0)
        expert_inputs = []
        expert_inputs_size = []
        expert_split_sizes = []
        for i in range(self.num_expert_per_node):
            expert_inputs.extend(chunks[i::self.num_expert_per_node])
            expert_inputs_size.append(sum(x.shape[0] for x in chunks[i::self.num_expert_per_node]))
            expert_split_sizes.extend(x.shape[0] for x in chunks[i::self.num_expert_per_node])

        expert_input = torch.cat(expert_inputs, dim=0)
        expert_inputs_size = torch.tensor(expert_inputs_size)
        expert_inputs_offset = torch.cumsum(torch.tensor(expert_inputs_size, device=expert_input.device))

        expert_outputs = self.experts.forward(expert_input, expert_inputs_offset)
        expert_outputs = torch.split(expert_outputs, expert_split_sizes)
        expert_outputs_rearranged = []
        for rank_idx in self.dp_group.size():
            expert_outputs_rearranged.extend([expert_outputs[expert_idx * self.dp_group.size() + rank_idx ] for expert_idx in range(self.num_expert_per_node)])
        

        sent_tensors = torch.cat(expert_outputs_rearranged, dim=0)
        recv_tensors, _ = TensorExchange().apply(sent_tensors, recv_splits, self.dp_group)
                
        out[torch.cat(expert_indices, dim=0), :] = recv_tensors

        out = rearrange(out, " ... (c d) e -> ... c d e", d=self.k)
        out = (out * probs.unsqueeze(dim=-1)).sum(dim=-2)
        return rearrange(out, "(b c) ... -> b c ...", b = batch_size, c = context_len).to(input_dtype)

class GroupedPositionWiseFFLayer(nn.Module):
    def __init__(self, num_experts:int, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.dff = d_ff
        self.device = device
        self.dtype = dtype
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.nn.init.trunc_normal_(torch.Tensor(size=(num_experts, self.dff, self.d_model), device=device, dtype=dtype), 0, 1, -3, 3))
        self.w2 = nn.Parameter(torch.nn.init.trunc_normal_(torch.Tensor(size=(num_experts, self.d_model, self.dff), device=device, dtype=dtype), 0, 1, -3, 3))
        self.w3 = nn.Parameter(torch.nn.init.trunc_normal_(torch.Tensor(size=(num_experts, self.dff, self.d_model), device=device, dtype=dtype), 0, 1, -3, 3))
     #   self.w1 = LinearLayer(self.d_model, self.dff, device=device, dtype=dtype)
     #   self.w2 = LinearLayer(self.dff, self.d_model, device=device, dtype=dtype)
     #   self.w3 = LinearLayer(self.d_model, self.dff, device=device, dtype=dtype)
      #  self.silu = SiLuLayer()
    
    def forward(self, x: torch.Tensor, offsets: list) -> torch.Tensor:
        x1 = F.grouped_mm(
            x,
            self.w1,
            offs=offsets,
        )
        x1 = silu(x1)
        x2 = F.grouped_mm(
            x,
            self.w3,
            offs=offsets,
        )
        x1 = x1 * x2
        x_out = F.grouped_mm(
            x1,
            self.w2,
            offs=offsets,
        )
        return x_out
    
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
    
    
class ROPELayer(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
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
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, use_rope=True, RoPE: nn.Module = None):
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

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        # batch, seq_len, d_k
        k = rearrange(self.k_proj(x), "... i (j k) -> ... j i k", k=self.head_dim)
        q = rearrange(self.q_proj(x), "... i (j k) -> ... j i k", k=self.head_dim)
        v = rearrange(self.v_proj(x), "... i (j k) -> ... j i k", k=self.head_dim)

        if self.use_rope:
            if token_positions is None:
                len_seq = x.shape[1]
                token_positions = torch.arange(0, len_seq, device=x.device)
            k = self.RoPE.forward(k, token_positions)
            q = self.RoPE.forward(q, token_positions)
        
        if mask is None:
            mask = (1 - torch.triu(torch.ones(size=(q.shape[-2], k.shape[-2]), device=x.device), diagonal=1)).bool().reshape([1] * (q.ndim - 2) + [q.shape[-2], k.shape[-2]])
        res = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        res = rearrange(res, "... i j k ->... j (i k)")
        return self.output_proj(res)


class TransformerLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, num_expert_per_node=1, RoPELayer = None, dp_group: dist.ProcessGroup = None, moe_compute_dtype=torch.bfloat16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNormLayer(d_model)
        self.attn = MultiHeadLayerLL(d_model, num_heads, RoPE=RoPELayer)
        self.ln2 = RMSNormLayer(d_model)
        # self.ffn = PositionWiseFFLayer(d_model, d_ff)
        self.ffn = MoELayerMulti(d_model, d_ff, num_expert_per_node = num_expert_per_node, dp_group=dp_group, dtype=moe_compute_dtype)
    
    def forward(self, x, return_aux_loss=False):
        x = x + self.attn(self.ln1(x))
        if return_aux_loss:
            out, aux_loss = self.ffn(self.ln2(x), return_aux_loss=True)
            return x + out, aux_loss
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, num_expert_per_node=1, d_ff=None, theta=10000, dp_group: dist.ProcessGroup = None):
        super().__init__()
        if d_ff == None:
            d_ff = math.ceil(d_model * (8/3)/64)*64
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embeddings = EmbeddingLayer(vocab_size, d_model)
        rope_layer = ROPELayer(theta, d_model//num_heads, context_length)
        self.layers = nn.Sequential(*[TransformerLayer(d_model, num_heads, d_ff, num_expert_per_node=num_expert_per_node, RoPELayer=rope_layer, dp_group=dp_group) for i in range(num_layers)])
        self.ln_final = RMSNormLayer(d_model)
        self.lm_head = LinearLayer(d_model, vocab_size)
    
    def forward(self, x:torch.Tensor):
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

class TransformerLMPipelined(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, stage_this, stage_total, num_expert_per_node=1, d_ff=None, theta=10000, dp_group: dist.ProcessGroup = None, moe_compute_dtype=torch.bfloat16):
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
        self.layers = nn.Sequential(*[TransformerLayer(d_model, num_heads, d_ff, num_expert_per_node=num_expert_per_node, RoPELayer=rope_layer, dp_group=dp_group, moe_compute_dtype=moe_compute_dtype) for i in range(num_layers)])
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
    