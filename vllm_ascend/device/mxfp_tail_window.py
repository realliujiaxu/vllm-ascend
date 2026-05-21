# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""C8_MXFP tail-group KV cache write logic.

V scale is shared across up to 64 consecutive tokens (one V-scale group). When a
group is incomplete (tail), bf16 K/V must be accumulated and re-quantized as a
whole before each cache write so ``npu_dynamic_mx_quant(..., axis=0)`` sees the
full window.  Per-layer state lives in fixed ``[max_num_seqs, 64, ...]`` tensors
indexed by batch slot (same row semantics as ``block_table``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import torch

from vllm_ascend.device.mxfp_compat import MXFP_KV_SCALE_GROUP_SIZE

WriteMxfp8Fn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
        int,
    ],
    None,
]
QuantizeKvFn = Callable[
    [torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]


def mxfp_slot_v_group(slot: int | torch.Tensor) -> int | torch.Tensor:
    """Map a paged slot to its V-scale cache group (``slot // 64``)."""
    return slot // MXFP_KV_SCALE_GROUP_SIZE


def mxfp_v_group_start(global_pos: int) -> int:
    """Global token index of the first token in the V-scale group containing ``global_pos``."""
    return (global_pos // MXFP_KV_SCALE_GROUP_SIZE) * MXFP_KV_SCALE_GROUP_SIZE


def split_tokens_into_v_group_segments(
    global_positions: torch.Tensor,
) -> list[tuple[int, int, bool]]:
    """Split token indices into contiguous V-scale groups.

    Returns ``(start, end, is_64_aligned_full_group)`` slices.
    ``is_64_aligned_full_group`` is True only when the segment has exactly 64
    tokens and starts at a 64-aligned global position (g0).
    """
    n = int(global_positions.numel())
    if n == 0:
        return []
    segments: list[tuple[int, int, bool]] = []
    i = 0
    while i < n:
        g = int(global_positions[i].item()) // MXFP_KV_SCALE_GROUP_SIZE
        g0 = g * MXFP_KV_SCALE_GROUP_SIZE
        j = i + 1
        while j < n and int(global_positions[j].item()) // MXFP_KV_SCALE_GROUP_SIZE == g:
            j += 1
        length = j - i
        is_64_aligned_full_group = (
            length == MXFP_KV_SCALE_GROUP_SIZE and int(global_positions[i].item()) == g0
        )
        segments.append((i, j, is_64_aligned_full_group))
        i = j
    return segments


def iter_req_token_groups(
    *,
    num_tokens: int,
    query_start_loc: torch.Tensor | None,
    num_computed_tokens_cpu: torch.Tensor | None,
) -> Iterator[tuple[int, slice, torch.Tensor]]:
    """Yield ``(req_idx, token_slice, global_positions)`` for each request in batch.

    ``req_idx`` is the batch row (0 .. num_reqs-1), aligned with
    ``input_batch.req_ids[req_idx]`` and ``block_table[req_idx]``.
    ``global_positions[i] = num_computed_tokens_cpu[req_idx] + local_i``.
    """
    if num_tokens <= 0:
        return
    if query_start_loc is None:
        positions = torch.arange(num_tokens, dtype=torch.long)
        if num_computed_tokens_cpu is not None and num_computed_tokens_cpu.numel() == 1:
            positions = positions + int(num_computed_tokens_cpu[0].item())
        yield 0, slice(0, num_tokens), positions
        return

    num_reqs = int(query_start_loc.numel()) - 1
    for req_idx in range(num_reqs):
        start = int(query_start_loc[req_idx].item())
        end = int(query_start_loc[req_idx + 1].item())
        if end <= start or start >= num_tokens:
            continue
        end = min(end, num_tokens)
        base = 0
        if num_computed_tokens_cpu is not None and req_idx < num_computed_tokens_cpu.numel():
            base = int(num_computed_tokens_cpu[req_idx].item())
        local_len = end - start
        positions = torch.arange(local_len, dtype=torch.long) + base
        yield req_idx, slice(start, end), positions


@dataclass
class MxfpTailWindowWriter:
    """Per-layer tail-window state and KV write orchestration.

    Tail buffers are fixed-size tensors indexed by batch slot (like ``block_table``):
    row ``i`` corresponds to ``input_batch.req_ids[i]``.
    """

    max_num_seqs: int
    # Fixed buffers; row i == batch slot i (see block_table).
    win_k: torch.Tensor | None = None  # [max_num_seqs, 64, num_kv_heads, head_dim]
    win_v: torch.Tensor | None = None  # [max_num_seqs, 64, num_kv_heads, head_dim]
    win_slots: torch.Tensor | None = None  # [max_num_seqs, 64], paged slot per token
    win_lens: torch.Tensor | None = None  # [max_num_seqs], active length w in [0, 64]
    win_g0: torch.Tensor | None = None  # [max_num_seqs], global start of current tail group
    win_slot_group: torch.Tensor | None = None  # [max_num_seqs], slots[0] // 64 for group switch
    _buffers_ready: bool = field(default=False, init=False)

    def ensure_buffers(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if self._buffers_ready:
            return
        group = MXFP_KV_SCALE_GROUP_SIZE
        kv_shape = (self.max_num_seqs, group, num_kv_heads, head_dim)
        self.win_k = torch.zeros(kv_shape, dtype=dtype, device=device)
        self.win_v = torch.zeros(kv_shape, dtype=dtype, device=device)
        self.win_slots = torch.zeros((self.max_num_seqs, group), dtype=torch.long, device=device)
        self.win_lens = torch.zeros(self.max_num_seqs, dtype=torch.long, device=device)
        self.win_g0 = torch.zeros(self.max_num_seqs, dtype=torch.long, device=device)
        self.win_slot_group = torch.zeros(self.max_num_seqs, dtype=torch.long, device=device)
        self._buffers_ready = True

    def _row_w(self, req_idx: int) -> int:
        assert self.win_lens is not None
        return int(self.win_lens[req_idx].item())

    def _reset_row(self, req_idx: int) -> None:
        assert self.win_lens is not None
        self.win_lens[req_idx] = 0
        self.win_g0[req_idx] = 0
        self.win_slot_group[req_idx] = 0

    def prune_tail_windows(self, num_active_reqs: int) -> None:
        """Clear tail windows for batch slots ``[num_active_reqs, max_num_seqs)``.

        Called by ``NPUModelRunner._prune_c8_mxfp_tail_windows`` when requests
        leave the batch.  Only metadata rows are zeroed; ``win_k/v/slots``
        contents in inactive rows are stale and ignored.
        """
        if not self._buffers_ready or self.win_lens is None:
            return
        if num_active_reqs >= self.max_num_seqs:
            return
        if num_active_reqs <= 0:
            self.win_lens.zero_()
            self.win_g0.zero_()
            self.win_slot_group.zero_()
            return
        self.win_lens[num_active_reqs:].zero_()
        self.win_g0[num_active_reqs:].zero_()
        self.win_slot_group[num_active_reqs:].zero_()

    def write_batch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        num_tokens: int,
        query_start_loc: torch.Tensor | None,
        num_computed_tokens_cpu: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        quantize_kv: QuantizeKvFn,
        write_mxfp8: WriteMxfp8Fn,
        block_size: int,
    ) -> None:
        self.ensure_buffers(
            num_kv_heads=key.shape[-2],
            head_dim=key.shape[-1],
            dtype=key.dtype,
            device=key.device,
        )

        # Flattened batch may contain multiple requests; split by query_start_loc
        # so each req is quantized/written independently (V axis=0 must not
        # cross req boundaries).  req_idx selects the tail-window row (block_table slot).
        for req_idx, token_slice, global_positions in iter_req_token_groups(
            num_tokens=num_tokens,
            query_start_loc=query_start_loc,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
        ):
            # token_slice: this req's tokens within the step-local K/V batch
            k_req = key[token_slice]
            v_req = value[token_slice]
            slots_req = slot_mapping[token_slice]
            # global_positions: sequence index of each token (for 64-align / tail split)
            self._write_request_tokens(
                req_idx,
                k_req,
                v_req,
                slots_req,
                global_positions,
                kv_cache=kv_cache,
                quantize_kv=quantize_kv,
                write_mxfp8=write_mxfp8,
                block_size=block_size,
            )

    def _write_request_tokens(
        self,
        req_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        slots: torch.Tensor,
        global_positions: torch.Tensor,
        *,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        quantize_kv: QuantizeKvFn,
        write_mxfp8: WriteMxfp8Fn,
        block_size: int,
    ) -> None:
        # Split this request's tokens into 64-aligned full groups vs tail segments.
        for start, end, is_64_aligned_full_group in split_tokens_into_v_group_segments(
            global_positions
        ):
            seg_k = key[start:end]
            seg_v = value[start:end]
            seg_slots = slots[start:end]
            # Complete 64-token V-scale group: one-shot quant+write, no tail buffer.
            if is_64_aligned_full_group:
                self._write_full_group(seg_k, seg_v, seg_slots, kv_cache, quantize_kv, write_mxfp8)
                continue
            # Incomplete tail group: accumulate in win_*[req_idx] and rewrite on append.
            g0 = mxfp_v_group_start(int(global_positions[start].item()))
            self._merge_tail_segment(
                req_idx,
                g0,
                seg_k,
                seg_v,
                seg_slots,
                kv_cache=kv_cache,
                quantize_kv=quantize_kv,
                write_mxfp8=write_mxfp8,
                block_size=block_size,
            )

    def _write_full_group(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slots: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        quantize_kv: QuantizeKvFn,
        write_mxfp8: WriteMxfp8Fn,
    ) -> None:
        # Full 64-token V-scale group: one-shot quant + write, no tail buffer.
        quant_key, quant_value, key_scale, value_scale = quantize_kv(key, value)
        write_mxfp8(
            quant_key,
            quant_value,
            key_scale,
            value_scale,
            kv_cache,
            slots,
            int(key.shape[0]),
        )

    def _merge_tail_segment(
        self,
        req_idx: int,
        g0: int,
        key: torch.Tensor,
        value: torch.Tensor,
        slots: torch.Tensor,
        *,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        quantize_kv: QuantizeKvFn,
        write_mxfp8: WriteMxfp8Fn,
        block_size: int,
    ) -> None:
        assert self.win_k is not None and self.win_v is not None and self.win_slots is not None
        w = self._row_w(req_idx)
        new_slot_group = int(mxfp_slot_v_group(slots[0]).item())
        # Start a fresh window when the tail group or V-scale slot group changes.
        need_new_window = (
            w == 0
            or int(self.win_g0[req_idx].item()) != g0
            or int(self.win_slot_group[req_idx].item()) != new_slot_group
        )
        if need_new_window:
            if w > 0:
                # Flush the previous incomplete group before switching.
                self._rewrite_tail_window_row_to_cache(
                    req_idx,
                    kv_cache=kv_cache,
                    quantize_kv=quantize_kv,
                    write_mxfp8=write_mxfp8,
                    block_size=block_size,
                )
            self._reset_row(req_idx)
            self.win_g0[req_idx] = g0
            self.win_slot_group[req_idx] = new_slot_group

        n = int(key.shape[0])
        w = self._row_w(req_idx)
        self.win_k[req_idx, w : w + n] = key
        self.win_v[req_idx, w : w + n] = value
        self.win_slots[req_idx, w : w + n] = slots.to(self.win_slots.device)
        self.win_lens[req_idx] = w + n

        # Re-quantize the entire window and overwrite all w slots in cache.
        self._rewrite_tail_window_row_to_cache(
            req_idx,
            kv_cache=kv_cache,
            quantize_kv=quantize_kv,
            write_mxfp8=write_mxfp8,
            block_size=block_size,
        )
        # Group complete: release the row so the next token opens a new window.
        if self._row_w(req_idx) == MXFP_KV_SCALE_GROUP_SIZE:
            self._reset_row(req_idx)

    def _rewrite_tail_window_row_to_cache(
        self,
        req_idx: int,
        *,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        quantize_kv: QuantizeKvFn,
        write_mxfp8: WriteMxfp8Fn,
        block_size: int,
    ) -> None:
        """Quantize ``win_*[req_idx, :w]`` and scatter into paged KV cache.

        V uses ``axis=0`` quant over the w accumulated rows, so every append
        must rewrite all w token slots (not just the latest token).
        """
        assert self.win_k is not None and self.win_v is not None and self.win_slots is not None
        w = self._row_w(req_idx)
        if w == 0:
            return
        row_k = self.win_k[req_idx, :w]
        row_v = self.win_v[req_idx, :w]
        row_slots = self.win_slots[req_idx, :w]
        quant_key, quant_value, key_scale, value_scale = quantize_kv(row_k, row_v)
        write_mxfp8(
            quant_key,
            quant_value,
            key_scale,
            value_scale,
            kv_cache,
            row_slots,
            w,
        )
