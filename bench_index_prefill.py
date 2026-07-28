"""Benchmark _prefill_index_score_prepare_for_topk_kernel: old grid vs new grid.

Old grid: (max_query_len, batch, num_idx_heads) -- one program per (token, batch,
head), each looping internally over all max_block (chunked by 2048). Total
programs = max_query_len * batch * num_idx_heads (very large).

New grid: (num_idx_heads, prep_target) with num_idx_heads * prep_target ~ 64
(PREP_TARGET_GRID). Each program owns one head + one max_block split-K slice and
streams over every (batch, token) row.

Reports per-iteration latency and throughput for several shapes.
"""
import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.attention.msa_m3_triton import (
    PREP_BLOCK_TILE_SIZE,
    SPARSE_BLOCK_SIZE,
    _choose_prep_query_tile_size,
    _prefill_index_score_prepare_for_topk_kernel,
    minimax_m3_index_score,
)

BLOCK_SIZE = SPARSE_BLOCK_SIZE  # 128


def _synchronize() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


# --- OLD-grid kernel (original implementation), copied verbatim -------------
@triton.jit(do_not_specialize=["max_block", "chunk_blocks", "num_prep_chunks"])
def _prefill_prepare_oldgrid(
    score_ptr,
    cu_seqlens,
    prefix_lens,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    max_block,
    chunk_blocks,
    num_prep_chunks,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    block_size: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    if pid_q >= q_len:
        return
    token_idx = seq_start + pid_q
    prefix_len = tl.load(prefix_lens + pid_b)
    valid_blocks = (prefix_len + pid_q + block_size) // block_size
    local_start = tl.maximum(0, valid_blocks - local_blocks)

    for pid_chunk in tl.range(0, num_prep_chunks):
        chunk_start = pid_chunk * chunk_blocks
        chunk_end = tl.minimum(chunk_start + chunk_blocks, max_block)
        if chunk_start < chunk_end:
            num_blks = chunk_end - chunk_start
            off_k = tl.arange(0, BLOCK_SIZE_K)
            for i in tl.range(0, num_blks, BLOCK_SIZE_K):
                blk = chunk_start + i + off_k
                mask = (i + off_k) < num_blks
                s_ptrs = (
                    score_ptr
                    + pid_h * stride_s_h
                    + token_idx * stride_s_n
                    + blk * stride_s_k
                )
                score = tl.load(s_ptrs, mask=mask, other=float("-inf"))
                blk_valid = blk < valid_blocks
                score = tl.where(blk_valid, score, float("-inf"))
                is_init = (blk < init_blocks) & blk_valid
                is_local = (blk >= local_start) & blk_valid
                score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
                tl.store(s_ptrs, score, mask=mask)


def run_old(score, cu_seqlens, prefix_lens, max_query_len, batch, num_idx_heads,
            max_block, init_blocks, local_blocks):
    prep_chunk_blocks = 2048
    num_prep_chunks = (max_block + prep_chunk_blocks - 1) // prep_chunk_blocks
    grid = (max_query_len, batch, num_idx_heads)
    _prefill_prepare_oldgrid[grid](
        score, cu_seqlens, prefix_lens,
        init_blocks, local_blocks,
        max_block, prep_chunk_blocks, num_prep_chunks,
        score.stride(0), score.stride(1), score.stride(2),
        block_size=BLOCK_SIZE, BLOCK_SIZE_K=2048,
    )


def run_new(score, cu_seqlens, prefix_lens, max_query_len, batch, num_idx_heads,
            max_block, init_blocks, local_blocks):
    query_tile_size = _choose_prep_query_tile_size(max_query_len, batch, num_idx_heads)
    grid = (triton.cdiv(max_query_len, query_tile_size), batch * num_idx_heads)
    _prefill_index_score_prepare_for_topk_kernel[grid](
        score, cu_seqlens, prefix_lens,
        num_idx_heads, init_blocks, local_blocks, max_block,
        score.stride(0), score.stride(1), score.stride(2),
        sparse_block_size=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_Q=query_tile_size,
        BLOCK_SIZE_K=PREP_BLOCK_TILE_SIZE,
    )


def build_score(q_lens, prefix_lens, num_idx_heads, head_dim, dtype):
    cpu = "cpu"
    q_lens = torch.tensor(q_lens, device=cpu, dtype=torch.int32)
    prefix_lens = torch.tensor(prefix_lens, device=cpu, dtype=torch.int32)
    seq_lens = prefix_lens + q_lens
    batch = q_lens.numel()
    max_query_len = int(q_lens.max().item())
    max_seq_len = int(seq_lens.max().item())
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = batch * max_blocks
    cu_seqlens = torch.zeros(batch + 1, device=cpu, dtype=torch.int32)
    cu_seqlens[1:] = q_lens.cumsum(0)
    block_table = torch.randperm(num_pages, device=cpu, dtype=torch.int32).reshape(
        batch, max_blocks
    )
    total_q = int(q_lens.sum().item())
    idx_q = torch.randn(total_q, num_idx_heads, head_dim, device=cpu, dtype=dtype)
    index_kv_cache = torch.randn(num_pages, BLOCK_SIZE, head_dim, device=cpu, dtype=dtype)
    sm_scale = head_dim ** -0.5

    idx_q = idx_q.npu()
    index_kv_cache = index_kv_cache.npu()
    block_table = block_table.npu()
    cu_seqlens = cu_seqlens.npu()
    seq_lens = seq_lens.npu()
    prefix_lens = prefix_lens.npu()

    score = minimax_m3_index_score(
        idx_q, index_kv_cache, block_table, cu_seqlens, seq_lens, prefix_lens,
        max_query_len=max_query_len, max_seq_len=max_seq_len,
        num_kv_heads=num_idx_heads, sm_scale=sm_scale,
    )
    _synchronize()
    meta = dict(
        cu_seqlens=cu_seqlens, prefix_lens=prefix_lens,
        max_query_len=max_query_len, batch=batch, num_idx_heads=num_idx_heads,
        max_block=score.shape[2], init_blocks=1, local_blocks=1,
    )
    return score, meta


def bench(fn, score, meta, iters, warmup):
    # Per-iteration device-event timing: record around each single kernel launch
    # and synchronize, so the measured elapsed_time is pure device execution with
    # no host-enqueue gap. Report the median over `iters` runs.
    for _ in range(warmup):
        fn(score, **meta)
    _synchronize()
    times = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn(score, **meta)
        end.record()
        _synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


SHAPES = [
    dict(name="q8k  h4",  q_lens=(8192,),   prefix_lens=(8 * 1024,),  num_idx_heads=4),
    dict(name="q16k h4",  q_lens=(16384,),  prefix_lens=(16 * 1024,), num_idx_heads=4),
    dict(name="q32k h4",  q_lens=(32768,),  prefix_lens=(32 * 1024,), num_idx_heads=4),
    dict(name="q16k h8",  q_lens=(16384,),  prefix_lens=(16 * 1024,), num_idx_heads=8),
    dict(name="2xq8k h4", q_lens=(8192, 8192), prefix_lens=(4 * 1024, 4 * 1024), num_idx_heads=4),
]


def main():
    assert hasattr(torch, "npu") and torch.npu.is_available(), "NPU is not available"
    torch.manual_seed(0)
    head_dim = 128
    dtype = torch.bfloat16
    iters, warmup = 30, 10

    print(
        f"{'shape':<12} | {'max_block':>8} {'total_q':>7} {'heads':>5} | "
        f"{'old grid':>22} | {'new grid':>16} | "
        f"{'old ms':>8} {'new ms':>8} {'speedup':>7} | {'match':>5}"
    )
    print("(device-event timed, median over {} iters)".format(iters))
    print("-" * 110)
    for s in SHAPES:
        score, meta = build_score(s["q_lens"], s["prefix_lens"], s["num_idx_heads"],
                                  head_dim, dtype)

        # Correctness: both must produce identical scores on the same input.
        ref = score.clone()
        run_old(ref, **meta)
        out_new = score.clone()
        run_new(out_new, **meta)
        _synchronize()
        match = torch.equal(ref, out_new)

        old_ms = bench(run_old, score, meta, iters, warmup)
        new_ms = bench(run_new, score, meta, iters, warmup)
        speedup = old_ms / new_ms if new_ms > 0 else float("inf")

        max_block = meta["max_block"]
        total_q = score.shape[1]
        nh = meta["num_idx_heads"]
        old_total = meta["max_query_len"] * meta["batch"] * nh
        qtile = _choose_prep_query_tile_size(meta["max_query_len"], meta["batch"], nh)
        new_total = triton.cdiv(meta["max_query_len"], qtile) * meta["batch"] * nh
        print(
            f"{s['name']:<12} | {max_block:>8} {total_q:>7} {nh:>5} | "
            f"{old_total:>22} | {new_total:>16} | "
            f"{old_ms:>8.3f} {new_ms:>8.3f} {speedup:>6.2f}x | {str(match):>5}"
        )

        del score, ref, out_new, meta
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
