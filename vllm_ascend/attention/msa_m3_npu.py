# SPDX-License-Identifier: Apache-2.0
"""NPU sparse attention ops for MiniMax-M3 on Ascend."""

from __future__ import annotations

import torch

_SPARSE_ATTN_INNER_PRECISE = 4
FP8_E4M3_MAX = 448.0


def _split_main_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) < 2:
            raise ValueError("Main kv cache tuple must contain K and V tensors")
        k_cache, v_cache = kv_cache[0], kv_cache[1]
    else:
        if kv_cache.ndim != 5:
            raise ValueError(f"Unexpected main kv cache ndim: {kv_cache.ndim}")
        if kv_cache.shape[0] == 2:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        elif kv_cache.shape[1] == 2:
            k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
        else:
            raise ValueError(f"Unexpected main kv cache shape: {tuple(kv_cache.shape)}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(f"Unexpected split main kv cache shapes: {tuple(k_cache.shape)}, {tuple(v_cache.shape)}")
    return k_cache, v_cache


def _select_num_idx_from_topk(topk_idx: torch.Tensor) -> torch.Tensor:
    return (topk_idx >= 0).sum(dim=-1).to(dtype=torch.int32)


def _to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    block_size: int = 128,
) -> None:
    del prefix_lens, max_query_len
    key, value = _split_main_kv_cache(kv_cache)
    q_lens_t = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    q_fp8 = _to_fp8(q)
    key_fp8 = key if key.dtype == torch.float8_e4m3fn else _to_fp8(key)
    value_fp8 = value if value.dtype == torch.float8_e4m3fn else _to_fp8(value)
    q_scale = torch.ones(1, dtype=torch.float32, device=q.device)
    k_scale = torch.ones(1, dtype=torch.float32, device=key.device)
    v_scale = torch.ones(1, dtype=torch.float32, device=value.device)
    out = torch.ops._C_ascend.npu_sparse_attention_score(
        q_fp8,
        key_fp8,
        value_fp8,
        topk_idx,
        block_table,
        select_num_idx=_select_num_idx_from_topk(topk_idx),
        actual_seq_lengths=q_lens_t,
        actual_seq_lengths_kv=seq_lens,
        q_dequant_scale=q_scale,
        k_dequant_scale=k_scale,
        v_dequant_scale=v_scale,
        num_key_value_heads=num_kv_heads,
        scale_value=sm_scale,
        block_size=block_size,
        top_k=topk_idx.shape[-1],
        inner_precise=_SPARSE_ATTN_INNER_PRECISE,
        attention_out_dtype=torch.bfloat16,
    )
    output.copy_(out)


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    block_size: int = 128,
) -> None:
    num_reqs = seq_lens.shape[0]
    active_tokens = num_reqs * decode_query_len
    q_active = q[:active_tokens]
    topk_active = topk_idx[:, :active_tokens]
    key, value = _split_main_kv_cache(kv_cache)
    q_lens_t = torch.full(
        (num_reqs,),
        decode_query_len,
        device=q.device,
        dtype=torch.int32,
    )
    q_fp8 = _to_fp8(q_active)
    key_fp8 = key if key.dtype == torch.float8_e4m3fn else _to_fp8(key)
    value_fp8 = value if value.dtype == torch.float8_e4m3fn else _to_fp8(value)
    q_scale = torch.ones(1, dtype=torch.float32, device=q_active.device)
    k_scale = torch.ones(1, dtype=torch.float32, device=key.device)
    v_scale = torch.ones(1, dtype=torch.float32, device=value.device)
    out = torch.ops._C_ascend.npu_sparse_attention_score(
        q_fp8,
        key_fp8,
        value_fp8,
        topk_active,
        block_table,
        select_num_idx=_select_num_idx_from_topk(topk_active),
        actual_seq_lengths=q_lens_t,
        actual_seq_lengths_kv=seq_lens,
        q_dequant_scale=q_scale,
        k_dequant_scale=k_scale,
        v_dequant_scale=v_scale,
        num_key_value_heads=num_kv_heads,
        scale_value=sm_scale,
        block_size=block_size,
        top_k=topk_active.shape[-1],
        inner_precise=_SPARSE_ATTN_INNER_PRECISE,
        attention_out_dtype=torch.bfloat16,
    )
    output[:active_tokens].copy_(out)
