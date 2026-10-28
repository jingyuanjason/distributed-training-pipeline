import torch
import torch.distributed as dist
from einops import rearrange

class MoETokenDispatchGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, expert_indices_all_experts, split_sizes_all_experts, k, ep_group: dist.ProcessGroup = None):
        if ep_group is not None:
            ep_rank_this = dist.get_group_rank(ep_group, dist.get_rank())
        else:
            ep_rank_this = 0
        indices_this = torch.cat(expert_indices_all_experts[ep_rank_this], dim=0)

        expert_input_this = x[indices_this//k, :]
        
        ctx.communication_group = ep_group
        ctx.expert_indices_all_experts = expert_indices_all_experts
        ctx.split_sizes_all_experts = split_sizes_all_experts
        ctx.x_size = x.size()
        ctx.k = k
        return expert_input_this

    def backward(ctx, d_recv: torch.Tensor, *_):
        device = d_recv.device
        dtype = d_recv.dtype
        _, d_model = d_recv.size()
        communication_group = ctx.communication_group
        expert_indices_all_experts = ctx.expert_indices_all_experts 
        split_sizes_all_experts = ctx.split_sizes_all_experts
        
        split_sizes_all_ranks = torch.sum(split_sizes_all_experts, dim=-1)
        if ctx.communication_group is not None:
            grad_recv_buffer = [torch.empty(split_size_rank, *tuple(d_recv.shape[1:]), dtype=dtype, device=device) for split_size_rank in split_sizes_all_ranks]
            dist.all_gather(grad_recv_buffer, d_recv, group=communication_group)
            
        else:
            grad_recv_buffer = [d_recv]
            
        d_x = torch.zeros( torch.sum(split_sizes_all_ranks), d_model, dtype=dtype, device=device)
        for grad_recv, indices_expert_per_rank in zip(grad_recv_buffer, expert_indices_all_experts):
            indices_rank = torch.cat(indices_expert_per_rank, dim=0)
            d_x[indices_rank,:] = grad_recv
        d_x = rearrange(d_x, "(a b) ...->a b ...", b=ctx.k).sum(dim=1)
        return d_x, None, None, None, None
    
class MoETokenCombineGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sent_tensors, split_sizes_all_experts: torch.Tensor, ep_group: dist.ProcessGroup = None):
        split_sizes_all_rank = split_sizes_all_experts.sum(dim=-1)
        dtype = sent_tensors.dtype
        device = sent_tensors.device

        # limitation: nccl only behavior, for gool the input shape has to be the same for different ranks
        if ep_group is not None:
            recv_buffer = [torch.empty(split_size, *tuple(sent_tensors.shape[1:]), dtype=dtype, device=device) for split_size in split_sizes_all_rank]
            dist.all_gather(recv_buffer, sent_tensors, group=ep_group)
        else:
            recv_buffer = [sent_tensors]
        ctx.split_sizes_all_rank = split_sizes_all_rank
        ctx.ep_group = ep_group
        return torch.cat(recv_buffer, dim=0)

    def backward(ctx, d_recv, *_):
        
        if ctx.ep_group is not None:
            ep_rank_this = dist.get_group_rank(ctx.ep_group, dist.get_rank())
        else:
            ep_rank_this = 0
    
        
        size_csum = torch.cumsum(ctx.split_sizes_all_rank, dim=0)
        start_idx = 0

        if ep_rank_this != 0:
            start_idx = size_csum[ep_rank_this-1]
        end_idx = size_csum[ep_rank_this]
        return d_recv[start_idx:end_idx], None, None
    
class AttnDispatchGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp_group):
        ctx.tp_group = tp_group
        return x
    
    def backward(ctx, d_recv, *_):
        if ctx.tp_group is not None:
            d_recv = d_recv.contiguous().clone()
            dist.all_reduce(d_recv, group=ctx.tp_group)
        return d_recv, None
    
class AttnCombineGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp_group):
        if tp_group is not None:
            x = x.contiguous().clone()
            dist.all_reduce(x, group=tp_group)
        return x
    
    def backward(ctx, d_recv, *_):
        return d_recv, None
    
    
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