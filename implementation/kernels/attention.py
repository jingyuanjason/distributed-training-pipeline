from einops import rearrange
import torch
import math
import torch.nn as nn
from torch import einsum
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl


class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):

        bq = 16
        bk = 16
        d = k.shape[-1]
        L = torch.zeros(size=v.shape[:-1])
        O = torch.zeros_like(v)
        for i in range(q.shape[-2]//bq):
            mi_last = torch.tensor(-1e10, device=q.device, dtype=q.dtype)
            si = torch.tensor(0, device=q.device, dtype=q.dtype)
            for j in range(k.shape[-2]//bk):
                sij = einsum( "...ij, ...kj -> ...ik", q[...,i*bq:i*bq+bq, :], k[...,j*bk:j*bk+bk, :])/math.sqrt(d)
                mi = torch.maximum(mi_last, sij.max(dim=-1, keepdim=True).values)
                m_update = torch.exp(mi_last - mi)
                ej = torch.exp(sij - mi)
                mi_last = mi
                
                si = si * m_update + ej.sum(dim=-1, keepdim=True)
                O[...,i*bq:i*bq+bq, :] = O[...,i*bq:i*bq+bq, :] * m_update + einsum("...ij,...jd->...id", ej, v[...,j*bk:j*bk+bk, :])
            L[...,i*bq:i*bq+bq] = mi.squeeze(dim=-1) + torch.log(si).squeeze(dim=-1)
            O[...,i*bq:i*bq+bq, :] = O[...,i*bq:i*bq+bq, :]/si


        ctx.save_for_backward(q, k, v, O, L)
        return O
        
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        # k  b, lk, dv
        # q  b, lq, dv
        # v  b, lk, dv
        # o  b, lq, dv
        # l  b, lq
        # do b, lq, dv
        D = (o * do).sum(dim=-1, keepdim=True) # b, lq, dv
        s = (einsum("...qd,...kd-> ...qk", q, k) / math.sqrt(k.shape[-1])) # b, lq, dk
        P = torch.exp(s - l.unsqueeze(-1)) # b, lq, lk
        dv = einsum("...qv,...qk-> ...kv", do, P)
        dp = einsum("...qv,...kv->...qk", do, v)
        ds = P * (dp - D)
        dQ = einsum("...qk,...kv->...qv", ds, k) / math.sqrt(k.shape[-1])
        dk = einsum("...qk,...qv->...kv", ds, q) / math.sqrt(k.shape[-1])
        return dQ, dk, dv, None
    

class FlashAttentionTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal=False):

        bq = 16
        bk = 16
        o = torch.empty_like(v, dtype=v.dtype, device=v.device)
        l = torch.empty(v.shape[:-1], dtype=v.dtype, device=v.device)
        grid = (triton.cdiv(q.shape[1], bq), q.shape[0],)


        if not is_causal:
            mask = torch.ones((1, q.shape[-2], k.shape[-2]), dtype=v.dtype, device=v.device)
        else:
            k_idx = torch.arange(0, k.shape[-2], dtype=v.dtype, device=v.device)
            q_idx = torch.arange(0, q.shape[-2], dtype=v.dtype, device=v.device)
            mask = (q_idx[:, None] >= k_idx[None, :]).to(torch.float32).unsqueeze(0)

        attention_forward[grid](q, k, v, o, l,
                                q.stride(0), q.stride(1), q.stride(2),
                                k.stride(0), k.stride(1), k.stride(2),
                                v.stride(0), v.stride(1), v.stride(2),
                                o.stride(0), o.stride(1), o.stride(2),
                                l.stride(0), l.stride(1),
                                q.shape[1], k.shape[1],
                                1/math.sqrt(k.shape[-1]),
                                k.shape[-1],
                                bq,
                                bk,
                                is_causal)
        
        ctx.save_for_backward(q, k, v, o, l, mask)
        return o
        
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l, mask = ctx.saved_tensors
        attention_pytorch_compiled = torch.compile(attention_pytorch)
        return attention_pytorch_compiled(q, k, v, o, do, l, mask)
    
def attention_pytorch(q, k, v, o, do, l, mask):
    D = (o * do).sum(dim=-1, keepdim=True) # b, lq, dv
    s = (einsum("...qd,...kd-> ...qk", q, k) / math.sqrt(k.shape[-1])) * mask - 1e6 * (1 - mask) # b, lq, dk
    P = torch.exp(s - l.unsqueeze(-1)) # b, lq, lk
    dv = einsum("...qv,...qk-> ...kv", do, P)
    dp = einsum("...qv,...kv->...qk", do, v)
    ds = P * (dp - D)
    dQ = einsum("...qk,...kv->...qv", ds, k) / math.sqrt(k.shape[-1])
    dk = einsum("...qk,...qv->...kv", ds, q) / math.sqrt(k.shape[-1])
    return dQ, dk, dv, None
    
@triton.jit
def attention_forward(Q_ptr, K_ptr, V_ptr,
                        O_ptr, L_ptr,
                        stride_qb, stride_qq, stride_qd,
                        stride_kb, stride_kk, stride_kd,
                        stride_vb, stride_vk, stride_vd,
                        stride_ob, stride_oq, stride_od,
                        stride_lb, stride_lq,
                        N_QUERIES, N_KEYS,
                        scale,
                        D: tl.constexpr,
                        Q_TILE_SIZE: tl.constexpr,
                        K_TILE_SIZE: tl.constexpr,
                        is_causal: tl.constexpr = False):
    q_tile_idx = tl.program_id(0)
    batch_index = tl.program_id(1)
    q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb,
                                    shape=(N_QUERIES, D),
                                    strides=(stride_qq, stride_qd),
                                    offsets=(q_tile_idx * Q_TILE_SIZE, 0),
                                    block_shape=(Q_TILE_SIZE, D),
                                    order=(1, 0))
    k_block_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb,
                                    shape=(N_KEYS, D),
                                    strides=(stride_kk, stride_kd),
                                    offsets=(0, 0),
                                    block_shape=(K_TILE_SIZE, D),
                                    order=(1, 0)
                                    )
    v_block_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb,
                                    shape=(N_KEYS, D),
                                    strides=(stride_vk, stride_vd),
                                    offsets=(0, 0),
                                    block_shape=(K_TILE_SIZE, D),
                                    order=(1, 0))
    output_block_ptr = tl.make_block_ptr(O_ptr + batch_index * stride_ob,
                                         shape=(Q_TILE_SIZE, D),
                                         strides=(stride_oq, stride_od),
                                         offsets=(q_tile_idx * Q_TILE_SIZE, 0),
                                         block_shape=(Q_TILE_SIZE, D),
                                         order=(1, 0))
    l_block_ptr = tl.make_block_ptr(L_ptr + stride_lb * batch_index,
                                    shape=(Q_TILE_SIZE,),
                                    strides=(stride_lq,),
                                    offsets=(q_tile_idx * Q_TILE_SIZE,),
                                    block_shape=(Q_TILE_SIZE,),
                                    order=(0,))
    
    max_kq_prev = tl.full((Q_TILE_SIZE,1), -1e10, dtype=tl.float32)
    output_temp = tl.full((Q_TILE_SIZE, D), 0, dtype=tl.float32)
    l_temp = tl.full((Q_TILE_SIZE, 1), 0, dtype=tl.float32)
    q_this = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    idx_q = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    mask_this = tl.full((Q_TILE_SIZE, K_TILE_SIZE), 1, dtype=tl.float32)
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        if is_causal.value:
            idx_k = i * K_TILE_SIZE +  tl.arange(0, K_TILE_SIZE)
            mask_this = (idx_q[:,None] >= idx_k[None, :]).to(tl.float32)
        k_this = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_this = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        kq = tl.dot(q_this, k_this.T) * scale * mask_this - 1e6 * (1 - mask_this)
        max_kq = tl.maximum(max_kq_prev, tl.max(kq, axis=1, keep_dims=True))
        max_kq_diff = tl.exp(max_kq_prev - max_kq)
        kq_exp = tl.exp(kq - max_kq) 
        output_temp = output_temp * max_kq_diff + tl.dot(kq_exp.to(v_this.dtype), v_this)
        l_temp = l_temp * max_kq_diff + kq_exp.sum(axis=1, keep_dims=True)
        max_kq_prev = max_kq
        k_block_ptr = k_block_ptr.advance((K_TILE_SIZE, 0))
        v_block_ptr = v_block_ptr.advance((K_TILE_SIZE, 0))
    output_temp = output_temp / l_temp
    l_temp = max_kq_prev + tl.log(l_temp)
    tl.store(output_block_ptr, output_temp)
    tl.store(l_block_ptr, tl.max(l_temp, axis=1))