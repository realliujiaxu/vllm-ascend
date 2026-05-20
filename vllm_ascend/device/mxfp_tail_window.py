# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""C8_MXFP tail-group KV cache write logic (CPU-testable, NPU-agnostic core)."""

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
    return slot // MXFP_KV_SCALE_GROUP_SIZE


def mxfp_v_group_start(global_pos: int) -> int:
    return (global_pos // MXFP_KV_SCALE_GROUP_SIZE) * MXFP_KV_SCALE_GROUP_SIZE


@dataclass
class MxfpTailWindow:
    """Per-request bf16 buffer for the current incomplete 64-token V scale group."""

    g0: int
    win_k: torch.Tensor
    win_v: torch.Tensor
    win_slots: torch.Tensor

    @property
    def w(self) -> int:
        return int(self.win_k.shape[0])

    def append(self, k: torch.Tensor, v: torch.Tensor, slot: int | torch.Tensor) -> None:
        k_row = k.unsqueeze(0) if k.dim() == 2 else k
        v_row = v.unsqueeze(0) if v.dim() == 2 else v
        slot_t = slot if isinstance(slot, torch.Tensor) else torch.tensor([slot], dtype=torch.long)
        if self.w == 0:
            self.win_k = k_row
            self.win_v = v_row
            self.win_slots = slot_t.reshape(-1)
            return
        self.win_k = torch.cat([self.win_k, k_row], dim=0)
        self.win_v = torch.cat([self.win_v, v_row], dim=0)
        self.win_slots = torch.cat(
            [self.win_slots, slot_t.reshape(-1).to(self.win_slots.device)],
            dim=0,
        )

    def clear_buffers(self) -> None:
        self.win_k = self.win_k[:0]
        self.win_v = self.win_v[:0]
        self.win_slots = self.win_slots[:0]


def split_tokens_into_v_group_segments(
    global_positions: torch.Tensor,
) -> list[tuple[int, int, bool]]:
    """Split token indices into contiguous V-scale groups.

    Returns ``(start, end, is_full_group)`` slices. A full group has exactly 64
    tokens and starts at a 64-aligned global position.
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
        is_full = length == MXFP_KV_SCALE_GROUP_SIZE and int(global_positions[i].item()) == g0
        segments.append((i, j, is_full))
        i = j
    return segments


def iter_req_token_groups(
    *,
    num_tokens: int,
    query_start_loc: torch.Tensor | None,
    req_ids: Sequence[str] | None,
    num_computed_tokens_cpu: torch.Tensor | None,
) -> Iterator[tuple[str, slice, torch.Tensor]]:
    """Yield ``(req_id, token_slice, global_positions)`` for each request in batch."""
    if num_tokens <= 0:
        return
    if req_ids is None or query_start_loc is None:
        positions = torch.arange(num_tokens, dtype=torch.long)
        if num_computed_tokens_cpu is not None and num_computed_tokens_cpu.numel() == 1:
            positions = positions + int(num_computed_tokens_cpu[0].item())
        yield "default", slice(0, num_tokens), positions
        return

    num_reqs = min(len(req_ids), int(query_start_loc.numel()) - 1)
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
        yield req_ids[req_idx], slice(start, end), positions


def _encode_scale_as_uint8(scale: torch.Tensor) -> torch.Tensor:
    """Pack float scales into uint8 for CPU cache mock (tests only)."""
    return (scale.to(torch.float32) * 1000).round().clamp(0, 255).to(torch.uint8)


def simplified_mx_dynamic_quant(
    x: torch.Tensor,
    *,
    axis: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference quant for unit tests (deterministic, group-wise).

    Mimics ``npu_dynamic_mx_quant`` enough for tail-window equivalence tests:
    - default / K path: per-token scale along head_dim groups
    - ``axis=0``: one scale per up-to-64 rows (V tail groups)
    """
    if axis == 0:
        w = x.shape[0]
        num_groups = (w + MXFP_KV_SCALE_GROUP_SIZE - 1) // MXFP_KV_SCALE_GROUP_SIZE
        tail_shape = x.shape[1:]
        scale_rows = []
        quant_chunks = []
        for g in range(num_groups):
            chunk = x[g * MXFP_KV_SCALE_GROUP_SIZE : min((g + 1) * MXFP_KV_SCALE_GROUP_SIZE, w)]
            scale_val = chunk.abs().amax().clamp(min=1e-6)
            scale_rows.append(scale_val)
            quant_chunks.append(chunk / scale_val)
        quant = torch.cat(quant_chunks, dim=0)
        scale_stacked = torch.stack(scale_rows, dim=0)
        value_scale = scale_stacked.view(num_groups, *([1] * len(tail_shape))).expand(
            num_groups, *tail_shape
        )
        value_scale = value_scale.unsqueeze(-1).expand(*value_scale.shape, 2).contiguous()
        return quant, _encode_scale_as_uint8(value_scale)

    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"simplified_mx_dynamic_quant expects rank-3 K tensor, got shape {x.shape}")
    w, num_kv_heads, head_dim = x.shape
    groups = head_dim // MXFP_KV_SCALE_GROUP_SIZE
    grouped = x.view(w, num_kv_heads, groups, MXFP_KV_SCALE_GROUP_SIZE)
    scale_per_group = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    quant = (grouped / scale_per_group).view(w, num_kv_heads, head_dim)
    key_scale = (
        scale_per_group.squeeze(-1).unsqueeze(-1).expand(w, num_kv_heads, groups, 2).contiguous()
    )
    return quant, _encode_scale_as_uint8(key_scale)


@dataclass
class MxfpTailWindowWriter:
    """Per-layer tail-window state and KV write orchestration."""

    _tail_windows: dict[str, MxfpTailWindow] = field(default_factory=dict)

    def prune_tail_windows(self, active_req_ids: set[str]) -> None:
        for req_id in list(self._tail_windows):
            if req_id not in active_req_ids:
                del self._tail_windows[req_id]

    def prune_stale_except(self, req_ids: Sequence[str]) -> None:
        self.prune_tail_windows(set(req_ids))

    def write_batch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        num_tokens: int,
        query_start_loc: torch.Tensor | None,
        req_ids: Sequence[str] | None,
        num_computed_tokens_cpu: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        quantize_kv: QuantizeKvFn,
        write_mxfp8: WriteMxfp8Fn,
        block_size: int,
    ) -> None:
        if req_ids is not None:
            self.prune_stale_except(req_ids)

        for req_id, token_slice, global_positions in iter_req_token_groups(
            num_tokens=num_tokens,
            query_start_loc=query_start_loc,
            req_ids=req_ids,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
        ):
            k_req = key[token_slice]
            v_req = value[token_slice]
            slots_req = slot_mapping[token_slice]
            self._write_request_tokens(
                req_id,
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
        req_id: str,
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
        for start, end, is_full in split_tokens_into_v_group_segments(global_positions):
            seg_k = key[start:end]
            seg_v = value[start:end]
            seg_slots = slots[start:end]
            if is_full:
                self._write_full_group(seg_k, seg_v, seg_slots, kv_cache, quantize_kv, write_mxfp8)
                continue
            g0 = mxfp_v_group_start(int(global_positions[start].item()))
            self._merge_tail_segment(
                req_id,
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
        req_id: str,
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
        win = self._tail_windows.get(req_id)
        new_slot_group = int(mxfp_slot_v_group(slots[0]).item())
        need_new_window = (
            win is None
            or win.w == 0
            or win.g0 != g0
            or int(mxfp_slot_v_group(win.win_slots[0]).item()) != new_slot_group
        )
        if need_new_window:
            if win is not None and win.w > 0:
                self._rewrite_tail_window_to_cache(
                    win,
                    kv_cache=kv_cache,
                    quantize_kv=quantize_kv,
                    write_mxfp8=write_mxfp8,
                    block_size=block_size,
                )
            win = MxfpTailWindow(
                g0=g0,
                win_k=key.new_zeros((0, *key.shape[1:])),
                win_v=value.new_zeros((0, *value.shape[1:])),
                win_slots=slots.new_zeros((0,), dtype=torch.long),
            )
            self._tail_windows[req_id] = win

        for i in range(int(key.shape[0])):
            win.append(key[i], value[i], int(slots[i].item()))

        self._rewrite_tail_window_to_cache(
            win,
            kv_cache=kv_cache,
            quantize_kv=quantize_kv,
            write_mxfp8=write_mxfp8,
            block_size=block_size,
        )
        if win.w == MXFP_KV_SCALE_GROUP_SIZE:
            del self._tail_windows[req_id]

    def _rewrite_tail_window_to_cache(
        self,
        win: MxfpTailWindow,
        *,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        quantize_kv: QuantizeKvFn,
        write_mxfp8: WriteMxfp8Fn,
        block_size: int,
    ) -> None:
        if win.w == 0:
            return
        quant_key, quant_value, key_scale, value_scale = quantize_kv(win.win_k, win.win_v)
        write_mxfp8(
            quant_key,
            quant_value,
            key_scale,
            value_scale,
            kv_cache,
            win.win_slots,
            win.w,
        )
