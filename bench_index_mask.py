"""Benchmark _topk_index_mask_invalid_prefill_kernel: old grid vs new grid.

Old grid: (max_query_len, batch, num_idx_heads) -- one program per (token, batch,
head), scalar per-token body.

New grid: (cdiv(max_query_len, BLOCK_SIZE_Q), batch * num_idx_heads) -- tiled
query (BLOCK_SIZE_Q chosen to target ~64 programs, bounded by a max 2-D tile
size), vectorized [BLOCK_SIZE_Q, topk] body. Mirrors
_prefill_topk_invalid_index_mask_kernel.

Per-iteration device-event timed (median over N iters).
"""
import torch

from vllm.triton_utils import tl, triton
from vllm_ascend.attention.msa_m3_triton import (
    SPARSE_BLOCK_SIZE,
    _choose_invalid_mask_query_tile_size,
    _topk_index_mask_invalid_prefill_kernel,
)

BLOCK_SIZE = SPARSE_BLOCK_SIZE  # 128


def _synchronize() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


# --- OLD-grid kernel (original implementation), copied verbatim -------------
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _mask_oldgrid(
    ti_ptr,
    cu_seqlens,
    prefix_lens,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    BLOCK_SIZE_T: tl.constexpr,
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

    off_t = tl.arange(0, BLOCK_SIZE_T)
    ti_ptrs = (
        ti_ptr + pid_h * stride_ti_h + token_idx * stride_ti_n + off_t * stride_ti_t
    )
    store_mask = off_t < topk
    idx = tl.load(ti_ptrs, mask=store_mask, other=0)
    valid_slot = off_t < tl.minimum(topk, valid_blocks)
    valid_idx = (idx >= 0) & (idx < valid_blocks)
    masked_idx = tl.where(valid_slot & valid_idx, idx, -1)
    tl.store(ti_ptrs, masked_idx.to(ti_ptr.dtype.element_ty), mask=store_mask)


def run_old(ti, cu_seqlens, prefix_lens, max_query_len, batch, num_idx_heads, topk):
    grid = (max_query_len, batch, num_idx_heads)
    _mask_oldgrid[grid](
        ti, cu_seqlens, prefix_lens,
        BLOCK_SIZE, topk,
        ti.stride(0), ti.stride(1), ti.stride(2),
    )


def run_new(ti, cu_seqlens, prefix_lens, max_query_len, batch, num_idx_heads, topk):
    tile = _choose_invalid_mask_query_tile_size(max_query_len, batch, num_idx_heads, topk)
    grid = (triton.cdiv(max_query_len, tile), batch * num_idx_heads)
    _topk_index_mask_invalid_prefill_kernel[grid](
        ti, cu_seqlens, prefix_lens,
        num_idx_heads, BLOCK_SIZE, topk,
        ti.stride(0), ti.stride(1), ti.stride(2),
        BLOCK_SIZE_Q=tile,
    )


def build_inputs(q_lens, prefix_lens, num_idx_heads, topk):
    cpu = "cpu"
    q_lens = torch.tensor(q_lens, device=cpu, dtype=torch.int32)
    prefix_lens = torch.tensor(prefix_lens, device=cpu, dtype=torch.int32)
    seq_lens = prefix_lens + q_lens
    batch = q_lens.numel()
    max_query_len = int(q_lens.max().item())
    max_seq_len = int(seq_lens.max().item())
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_q = int(q_lens.sum().item())

    cu_seqlens = torch.zeros(batch + 1, device=cpu, dtype=torch.int32)
    cu_seqlens[1:] = q_lens.cumsum(0)

    # Random raw top-k indices in a range that mixes valid and out-of-range
    # block ids, so the invalid-mask path is actually exercised.
    ti = torch.randint(0, 2 * max_blocks, (num_idx_heads, total_q, topk),
                       device=cpu, dtype=torch.int32).npu()
    cu_seqlens = cu_seqlens.npu()
    prefix_lens = prefix_lens.npu()
    meta = dict(
        cu_seqlens=cu_seqlens, prefix_lens=prefix_lens,
        max_query_len=max_query_len, batch=batch, num_idx_heads=num_idx_heads,
        topk=topk,
    )
    return ti, meta


def bench(fn, ti, meta, iters, warmup):
    # Per-iteration device-event timing; median over `iters`.
    for _ in range(warmup):
        fn(ti, **meta)
    _synchronize()
    times = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn(ti, **meta)
        end.record()
        _synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


SHAPES = [
    dict(name="q8k  h4 k16",  q_lens=(8192,),   prefix_lens=(8 * 1024,),  num_idx_heads=4, topk=16),
    dict(name="q16k h4 k16",  q_lens=(16384,),  prefix_lens=(16 * 1024,), num_idx_heads=4, topk=16),
    dict(name="q32k h4 k16",  q_lens=(32768,),  prefix_lens=(32 * 1024,), num_idx_heads=4, topk=16),
    dict(name="q16k h4 k64",  q_lens=(16384,),  prefix_lens=(16 * 1024,), num_idx_heads=4, topk=64),
    dict(name="q16k h8 k16",  q_lens=(16384,),  prefix_lens=(16 * 1024,), num_idx_heads=8, topk=16),
    dict(name="2xq8k h4 k16", q_lens=(8192, 8192), prefix_lens=(4 * 1024, 4 * 1024), num_idx_heads=4, topk=16),
]


def main():
    assert hasattr(torch, "npu") and torch.npu.is_available(), "NPU is not available"
    torch.manual_seed(0)
    iters, warmup = 30, 10

    print(
        f"{'shape':<16} | {'topk':>4} {'heads':>5} | "
        f"{'old grid':>10} {'new grid':>10} | "
        f"{'old ms':>8} {'new ms':>8} {'speedup':>7} | {'match':>5}"
    )
    print("(device-event timed, median over {} iters)".format(iters))
    print("-" * 90)
    for s in SHAPES:
        ti, meta = build_inputs(s["q_lens"], s["prefix_lens"], s["num_idx_heads"], s["topk"])

        # Correctness: both must produce identical masked output on same input.
        ref = ti.clone()
        run_old(ref, **meta)
        out_new = ti.clone()
        run_new(out_new, **meta)
        _synchronize()
        match = torch.equal(ref, out_new)

        old_ms = bench(run_old, ti, meta, iters, warmup)
        new_ms = bench(run_new, ti, meta, iters, warmup)
        speedup = old_ms / new_ms if new_ms > 0 else float("inf")

        nh = meta["num_idx_heads"]
        topk = meta["topk"]
        old_total = meta["max_query_len"] * meta["batch"] * nh
        tile = _choose_invalid_mask_query_tile_size(
            meta["max_query_len"], meta["batch"], nh, topk
        )
        new_total = triton.cdiv(meta["max_query_len"], tile) * meta["batch"] * nh
        print(
            f"{s['name']:<16} | {topk:>4} {nh:>5} | "
            f"{old_total:>10} {new_total:>10} | "
            f"{old_ms:>8.3f} {new_ms:>8.3f} {speedup:>6.2f}x | {str(match):>5}"
        )

        del ti, ref, out_new, meta
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
