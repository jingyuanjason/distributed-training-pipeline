from typing import Any, Type

from implementation.layers import LinearLayer, RMSNormLayer, EmbeddingLayer
from implementation.model import Linear, RMSNorm, Embedding
from einops import einsum, rearrange
import torch
import math
import torch.nn as nn
import torch.distributed as dist

from implementation.nnfunctions import dot_product_att, silu, softmax
from torch.optim import Optimizer
import collections


def sync_grad(works, FSDP_communication_group: dist.ProcessGroup = None):
    def dist_func(params):
        work = dist.all_reduce(params.grad.data, op=dist.ReduceOp.AVG, async_op=True, group=FSDP_communication_group)
        works.append(work)
        return None
    return dist_func

def sync_grad_reduce_scatter(works, FSDP_communication_group: dist.ProcessGroup = None):
    def dist_func(params):
        if FSDP_communication_group is not None:
            group_size = FSDP_communication_group.size()
        else:
            group_size = dist.get_world_size()
        grad_data = params.grad.data.to(dtype=torch.float32)
        params.grad.data = torch.empty(grad_data.shape[0]//group_size, *grad_data.shape[1:], device=grad_data.device, dtype=torch.float32)
        work = dist.reduce_scatter_tensor(params.grad.data, grad_data, op=dist.ReduceOp.AVG, async_op=True, group=FSDP_communication_group)
        works.append(work)
        return None
    return dist_func

def sync_weight_wrapper(dtype=torch.float32, FSDP_communication_group: dist.ProcessGroup = None):
    def sync_weight(module: nn.Module, args):
        if FSDP_communication_group is not None:
            group_size = FSDP_communication_group.size()
        else:
            group_size = dist.get_world_size()
        weight_data = module.weight.data
        module.weight.data_original = weight_data
        module.weight.data = torch.empty(group_size * weight_data.shape[0], *weight_data.shape[1:], device=weight_data.device, dtype=dtype)
        dist.all_gather_into_tensor(module.weight.data, weight_data.to(dtype=dtype), group=FSDP_communication_group)

    return sync_weight

def free_weight(module: nn.Module, args, output):
    module.weight.data = module.weight.data_original
    module.weight.data_original = None

def sync_weight_wrapper_backward(dtype=torch.float32, FSDP_communication_group: dist.ProcessGroup = None):
    def sync_weight(module: nn.Module, grad_output=None):
        if FSDP_communication_group is not None:
            group_size = FSDP_communication_group.size()
        else:
            group_size = dist.get_world_size()
        weight_data = module.weight.data
        module.weight.data_original = weight_data
        module.weight.data = torch.empty(group_size * weight_data.shape[0], *weight_data.shape[1:], device=weight_data.device, dtype=dtype)
        dist.all_gather_into_tensor(module.weight.data, weight_data.to(dtype=dtype), group=FSDP_communication_group)

    return sync_weight

def free_weight_backward(module: nn.Module, grad_input, grad_output):
      #  print(module._get_name(), module.weight.grad)
    module.weight.data = module.weight.data_original
    module.weight.data_original = None


class FSDPWrapperPipelined(nn.Module):
    def __init__(self, module: nn.Module, FSDP_communication_group: dist.ProcessGroup=None, compute_dtype: torch.dtype = torch.float32, moe_compute_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.compute_dtype = compute_dtype

        if FSDP_communication_group is not None:
            group_size = FSDP_communication_group.size()
        else:
            group_size = dist.get_world_size()

        dp_idx = dist.get_rank(FSDP_communication_group)
        self.works = []
        self.FSDP_communication_group = FSDP_communication_group
        self.group_size = group_size
        self.sharded_params = []
        self.replicated_params = []
        self.full_grad_accumulators = {}
        
        for layer_name, submodule in dict(module.named_modules()).items():
            
            if "expert" in layer_name:
                submodule.compile()
            if isinstance(submodule, (LinearLayer, EmbeddingLayer)):
                dist.broadcast(submodule.weight.data, dist.get_global_rank(FSDP_communication_group, 0), group=FSDP_communication_group)
                assert submodule.weight.data.shape[0] % group_size == 0, (
                    f"Cannot shard parameter with shape {tuple(submodule.weight.shape)} "
                    f"over {group_size} ranks"
                )
                shard_size = submodule.weight.data.shape[0] // group_size
                full_weight = submodule.weight.detach()
                submodule.weight.data = full_weight[dp_idx * shard_size: dp_idx * shard_size + shard_size, :].clone()
                layer_compute_dtype = moe_compute_dtype if ".ffn.router" in f".{layer_name}" else compute_dtype
                submodule.register_forward_pre_hook(sync_weight_wrapper(layer_compute_dtype, FSDP_communication_group))
                submodule.register_forward_hook(free_weight)
                submodule.register_full_backward_pre_hook(sync_weight_wrapper_backward(layer_compute_dtype, FSDP_communication_group))
                submodule.register_full_backward_hook(free_weight_backward)
                for params in submodule.parameters():
                    self.sharded_params.append(params)
            elif isinstance(submodule, (RMSNormLayer)):
                dist.broadcast(submodule.weight.data, dist.get_global_rank(FSDP_communication_group, 0), group=FSDP_communication_group)
                for params in submodule.parameters():
                    self.replicated_params.append(params)
        self.module = module
    
    def forward(self, x, **kwargs):
        return self.module.forward(x, **kwargs)

    @torch.no_grad()
    def clear_grad_accumulators(self):
        self.full_grad_accumulators.clear()
        for params in self.sharded_params + self.replicated_params:
            params.grad = None

    @torch.no_grad()
    def accumulate_full_gradients(self):
        for params in self.sharded_params + self.replicated_params:
            if params.grad is None:
                continue

            grad = params.grad.detach().to(dtype=torch.float32)
            accumulated = self.full_grad_accumulators.get(params)
            if accumulated is None:
                self.full_grad_accumulators[params] = grad.clone()
            else:
                if accumulated.shape != grad.shape:
                    raise RuntimeError(
                        f"Gradient shape changed between microbatches: "
                        f"{tuple(accumulated.shape)} vs {tuple(grad.shape)}"
                    )
                accumulated.add_(grad)

            params.grad = None

    @torch.no_grad()
    def reduce_accumulated_gradients(self):
        pending = []
        works = []

        for params in self.sharded_params:
            full_grad = self.full_grad_accumulators.get(params)
            if full_grad is None:
                continue
            if full_grad.shape[0] % self.group_size != 0:
                raise RuntimeError(
                    f"Cannot reduce-scatter gradient with shape {tuple(full_grad.shape)} "
                    f"over {self.group_size} ranks"
                )

            shard_grad = torch.empty_like(params.data, dtype=torch.float32)
            expected_shape = (
                full_grad.shape[0] // self.group_size,
                *full_grad.shape[1:],
            )
            if tuple(shard_grad.shape) != expected_shape:
                raise RuntimeError(
                    f"Gradient shard shape {tuple(shard_grad.shape)} does not "
                    f"match expected shape {expected_shape}"
                )
            reduce_input = full_grad.contiguous()
            work = dist.reduce_scatter_tensor(
                shard_grad,
                reduce_input,
                op=dist.ReduceOp.AVG,
                async_op=True,
                group=self.FSDP_communication_group,
            )
            works.append(work)
            pending.append((params, shard_grad, reduce_input))

        for params in self.replicated_params:
            full_grad = self.full_grad_accumulators.get(params)
            if full_grad is None:
                continue
            work = dist.all_reduce(
                full_grad,
                op=dist.ReduceOp.AVG,
                async_op=True,
                group=self.FSDP_communication_group,
            )
            works.append(work)
            pending.append((params, full_grad, full_grad))

        for work in works:
            work.wait()
        for params, reduced_grad, _collective_input in pending:
            params.grad = reduced_grad

        self.full_grad_accumulators.clear()

    def finish_gradient_synchronization(self):
        for work in self.works:
            work.wait()
        self.works.clear()


    


class FSDPWrapper(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.compute_dtype = compute_dtype
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        self.works = []
        
        for layer_name, submodule in dict(module.named_modules()).items():
            
            if "expert" in layer_name:
                continue
            if isinstance(submodule, (Linear, Embedding)):
                dist.broadcast(submodule.weight.data, 0)
                shard_size = submodule.weight.data.shape[0] // world_size
                full_weight = submodule.weight.detach()
                submodule.weight.data = full_weight[rank * shard_size: rank * shard_size + shard_size, :].clone()
                submodule.register_forward_pre_hook(sync_weight_wrapper(compute_dtype))
                submodule.register_forward_hook(free_weight)
                submodule.register_full_backward_pre_hook(sync_weight_wrapper_backward(compute_dtype))
                submodule.register_full_backward_hook(free_weight_backward)
                for params in submodule.parameters():
                    params.register_post_accumulate_grad_hook(sync_grad_reduce_scatter(self.works))
            elif isinstance(submodule, (RMSNorm)):
                dist.broadcast(submodule.weight.data, 0)
                for params in submodule.parameters():
                    params.register_post_accumulate_grad_hook(sync_grad(self.works))
        self.module = module
    
    def forward(self, x):

        return self.module.forward(x)

    def finish_gradient_synchronization(self):
        for work in self.works:
            work.wait()
        self.works.clear()

    def gather_full_state_dict(self):
        state_dict = self.module.state_dict()
        modules = dict(self.module.named_modules())
        for layer_name in state_dict:
            layer_name_defined = layer_name.rsplit(".", 1)[0]
            if modules[layer_name_defined].__class__.__name__ in ("Linear", "Embedding"):
                weight_data = state_dict[layer_name]
                world_size = dist.get_world_size()
                state_dict[layer_name] = torch.empty(world_size * weight_data.shape[0], *weight_data.shape[1:], device=weight_data.device, dtype=weight_data.dtype)
                dist.all_gather_into_tensor(state_dict[layer_name], weight_data)
        return state_dict

class DDPWraper(nn.Module):
    def __init__(self, module: nn.Module, src_rank: int = 0):
        super().__init__()
        self.works = []
        
        for params in module.parameters():
            dist.broadcast(params.data, src_rank)
            if params.requires_grad:
                params.register_post_accumulate_grad_hook(sync_grad(self.works))
        self.module = module
    
    def forward(self, x):
        return self.module.forward(x)

    def finish_gradient_synchronization(self):
        for work in self.works:
            work.wait()
        self.works.clear()


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer],  **kwargs: Any):
        #super().__init__(params, **kwargs)
        rank = 0
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        self.rank = rank
        self.world_size = world_size
        self.param_count = 0
        self.param_list = []
        param_group_this = []
        for param in params:
            if self.param_count % self.world_size == self.rank:
                param_group_this.append(param)
            self.param_list.append((self.param_count % self.world_size, param))
            self.param_count += 1

        self.optimizer_this = optimizer_cls(param_group_this, **kwargs)
    
    def step(self, closure = None, **kwargs):
        self.optimizer_this.step(closure=closure, **kwargs)
        workers = list(map(lambda b_args: dist.broadcast(b_args[1].data, src=b_args[0], async_op=True), self.param_list))
        for worker in workers:
            worker.wait()
    
    def add_param_group(self, param_group: dict[str, Any]):
        new_params = []
        for param in param_group["params"]:
            if self.param_count % self.world_size == self.rank:
                new_params.append(param)
            self.param_list.append((self.param_count % self.world_size, param))
            self.param_count += 1
        self.optimizer_this.add_param_group({"params": new_params})

    def __getattr__(self, name):
        # delegate unknown attributes/methods to the wrapped object
        return getattr(self.optimizer_this, name)
    
    

