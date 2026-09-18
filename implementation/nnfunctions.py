import torch
import math
import torch.nn as nn
from torch import einsum
import torch.cuda.nvtx as nvtx



def gradient_clippling(params, max_l2):
    l2_norm = torch.sqrt(
        sum( torch.sum(p.grad.data ** 2) if p.grad is not None else 0 for p in params)
    )

    if l2_norm > max_l2:
        for param in params:
            if param.grad is not None:
                param.grad.data = param.grad.data * (max_l2/(l2_norm + 1e-6))

def learning_rate_schedule(t, a_max, a_min, t_w, t_c):
    if t < t_w:
        return t/t_w * a_max
    elif t >= t_w and t <= t_c:
        return a_min + 1/2 * (1 + math.cos((t-t_w)/(t_c - t_w) * math.pi)) * (a_max - a_min)
    else:
        return a_min

def learning_rate_schedule_wrapper(a_max, a_min, t_w, t_c):
    def scheduler(t):
        if t < t_w:
            return t/t_w * a_max
        elif t >= t_w and t <= t_c:
            return a_min + 1/2 * (1 + math.cos((t-t_w)/(t_c - t_w) * math.pi)) * (a_max - a_min)
        else:
            return a_min
    return scheduler

def softmax(x: torch.Tensor, dim_sum: int, mask: torch.Tensor=None):
    x = x - torch.max(x, dim=dim_sum, keepdim=True).values
    
    x = torch.exp(x)
    if mask is not None:
        x = x * mask.to(x.dtype)
    return x/torch.sum(x, dim=dim_sum, keepdim=True)


def dot_product_att(k:torch.Tensor, q:torch.Tensor, v:torch.Tensor, mask: torch.Tensor=None):
    d_k = k.shape[-1]
    att_val = einsum("...ik,...jk->...ij", q, k) / math.sqrt(d_k)

    att_val = softmax(att_val, -1, mask)
    return einsum("...ki,...ij->...kj", att_val, v)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    logits = logits.float()
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values

    logexp_logits = torch.log(torch.exp(logits).sum(-1))

    logprob = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) - logexp_logits

    return -logprob.mean()

def sample_output(logits, t=1, p=0):
    prob = softmax(logits/t, dim_sum=-1)
    values, indices = torch.sort(prob, dim = -1, descending=True)
    current_prob_sum = 0
    mask_all = []
    for i in range(values.size(1)):
        current_prob_sum += values[:,i]
        if i == 0:
            mask_all.append(torch.ones_like(values[:,0], device=prob.device))
        else:
            mask_all.append((current_prob_sum < p).to(torch.float32))
    mask = torch.stack(mask_all, dim=-1)

    prob_masked = prob * mask
    sampled_pos = torch.multinomial(prob_masked, num_samples=1)
    sampled_index = torch.gather(indices, dim=-1, index=sampled_pos)
    return sampled_index