# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU unit tests for C8_MXFP tail-window KV write (§5.13).

Uses simplified ``simplified_mx_dynamic_quant`` and a CPU paged-cache writer instead
of ``npu_dynamic_mx_quant`` / ``DeviceOperator.reshape_and_cache``.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch

from vllm_ascend.device.mxfp_compat import (
    MXFP_KV_SCALE_GROUP_SIZE,
    mxfp_resolve_kv_cache_layout,
    scatter_mxfp_v_scale_cache,
)
from vllm_ascend.device.mxfp_tail_window import (
    MxfpTailWindowWriter,
    simplified_mx_dynamic_quant,
    split_tokens_into_v_group_segments,
)


@dataclass
class CacheSnapshot:
    k_fp8: torch.Tensor
    v_fp8: torch.Tensor
    k_scale: torch.Tensor
    v_scale_cache: torch.Tensor


def _make_bf16_kv(
    w: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    g0: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(w, num_kv_heads, head_dim, generator=gen, dtype=torch.float32)
    v = torch.randn(w, num_kv_heads, head_dim, generator=gen, dtype=torch.float32) + g0 * 1e-3
    return k, v


def _make_slots(w: int, g0: int, block_size: int) -> torch.Tensor:
    return torch.tensor(
        [g0 + i for i in range(w)],
        dtype=torch.long,
    )


def _alloc_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k_numel = num_blocks * block_size * num_kv_heads * head_dim
    v_numel = k_numel
    k_scale_numel = num_blocks * num_kv_heads * block_size * head_dim // 32
    v_scale_numel = k_scale_numel
    raw_k = torch.zeros(k_numel, dtype=torch.float32)
    raw_v = torch.zeros(v_numel, dtype=torch.float32)
    raw_k_scale = torch.zeros(k_scale_numel, dtype=torch.uint8)
    raw_v_scale = torch.zeros(v_scale_numel, dtype=torch.uint8)
    shapes = mxfp_resolve_kv_cache_layout(
        raw_k_numel=k_numel,
        raw_v_numel=v_numel,
        raw_k_scale_numel=k_scale_numel,
        raw_v_scale_numel=v_scale_numel,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        k_dim=head_dim,
        v_dim=head_dim,
    )
    return (
        raw_k.view(shapes[0]),
        raw_v.view(shapes[1]),
        raw_k_scale.view(shapes[2]),
        raw_v_scale.view(shapes[3]),
    )


def _quantize_kv_pair(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    quant_key, key_scale = simplified_mx_dynamic_quant(key)
    quant_value, value_scale = simplified_mx_dynamic_quant(value, axis=0)
    return quant_key, quant_value, key_scale, value_scale


def _cpu_write_mxfp8(
    quant_key: torch.Tensor,
    quant_value: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    slot_mapping: torch.Tensor,
    num_actual_tokens: int,
) -> None:
    key_cache, value_cache, key_scale_cache, value_scale_cache = kv_cache
    slots = slot_mapping[:num_actual_tokens].to(torch.long)
    key_cache.reshape(-1, *key_cache.shape[2:])[slots] = quant_key[:num_actual_tokens]
    value_cache.reshape(-1, *value_cache.shape[2:])[slots] = quant_value[:num_actual_tokens]
    key_scale_cache.permute(0, 2, 1, 3, 4).reshape(
        -1,
        key_scale_cache.shape[1],
        key_scale_cache.shape[3],
        key_scale_cache.shape[4],
    )[slots] = key_scale[:num_actual_tokens]
    scatter_mxfp_v_scale_cache(
        value_scale,
        value_scale_cache,
        slots,
        key_cache.shape[1],
    )


def _read_cache_snapshot(
    kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    slots: torch.Tensor,
) -> CacheSnapshot:
    key_cache, value_cache, key_scale_cache, value_scale_cache = kv_cache
    slots_l = slots.to(torch.long)
    k_fp8 = key_cache.reshape(-1, *key_cache.shape[2:])[slots_l].clone()
    v_fp8 = value_cache.reshape(-1, *value_cache.shape[2:])[slots_l].clone()
    k_scale = key_scale_cache.permute(0, 2, 1, 3, 4).reshape(
        -1,
        key_scale_cache.shape[1],
        key_scale_cache.shape[3],
        key_scale_cache.shape[4],
    )[slots_l].clone()
    return CacheSnapshot(
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        k_scale=k_scale,
        v_scale_cache=value_scale_cache.clone(),
    )


def _write_once(
    writer: MxfpTailWindowWriter,
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    g0: int,
    req_id: str = "req-0",
) -> None:
    positions = torch.arange(k.shape[0], dtype=torch.long) + g0
    writer.write_batch(
        k,
        v,
        kv_cache,
        num_tokens=k.shape[0],
        query_start_loc=torch.tensor([0, k.shape[0]], dtype=torch.long),
        req_ids=[req_id],
        num_computed_tokens_cpu=torch.tensor([g0], dtype=torch.long),
        slot_mapping=slots,
        quantize_kv=_quantize_kv_pair,
        write_mxfp8=_cpu_write_mxfp8,
        block_size=kv_cache[0].shape[1],
    )


def _write_iter_decode(
    writer: MxfpTailWindowWriter,
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    g0: int,
    req_id: str = "req-0",
) -> None:
    writer._tail_windows.clear()
    for i in range(k.shape[0]):
        writer.write_batch(
            k[i : i + 1],
            v[i : i + 1],
            kv_cache,
            num_tokens=1,
            query_start_loc=torch.tensor([0, 1], dtype=torch.long),
            req_ids=[req_id],
            num_computed_tokens_cpu=torch.tensor([g0 + i], dtype=torch.long),
            slot_mapping=slots[i : i + 1],
            quantize_kv=_quantize_kv_pair,
            write_mxfp8=_cpu_write_mxfp8,
            block_size=kv_cache[0].shape[1],
        )


class TestC8MxfpTailWindowKVWrite(unittest.TestCase):
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    BLOCK_SIZE = 512
    NUM_BLOCKS = 4
    REQ_ID = "req-0"

    def _fresh_writer(self) -> MxfpTailWindowWriter:
        return MxfpTailWindowWriter()

    def test_split_segments_full_and_tail(self):
        pos = torch.tensor([128, 129, 130, 192, 193], dtype=torch.long)
        segs = split_tokens_into_v_group_segments(pos)
        self.assertEqual(segs, [(0, 3, False), (3, 5, False)])

        pos_full = torch.arange(64, dtype=torch.long)
        segs_full = split_tokens_into_v_group_segments(pos_full)
        self.assertEqual(segs_full, [(0, 64, True)])

    def test_tail_window_64_iter_matches_one_shot(self):
        w = 64
        g0 = 0
        k, v = _make_bf16_kv(w, self.NUM_KV_HEADS, self.HEAD_DIM, g0=g0, seed=42)
        slots = _make_slots(w, g0, self.BLOCK_SIZE)

        cache_ref = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)
        cache_iter = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)

        writer_ref = self._fresh_writer()
        writer_iter = self._fresh_writer()

        _write_once(writer_ref, k, v, slots, cache_ref, g0=g0)
        _write_iter_decode(writer_iter, k, v, slots, cache_iter, g0=g0)

        snap_ref = _read_cache_snapshot(cache_ref, slots)
        snap_iter = _read_cache_snapshot(cache_iter, slots)

        torch.testing.assert_close(snap_ref.k_fp8, snap_iter.k_fp8)
        torch.testing.assert_close(snap_ref.v_fp8, snap_iter.v_fp8)
        torch.testing.assert_close(snap_ref.k_scale, snap_iter.k_scale)
        torch.testing.assert_close(snap_ref.v_scale_cache, snap_iter.v_scale_cache)
        self.assertNotIn(self.REQ_ID, writer_iter._tail_windows)

    def test_tail_window_64_iter_matches_one_shot_g0_128(self):
        w = 64
        g0 = 128
        k, v = _make_bf16_kv(w, self.NUM_KV_HEADS, self.HEAD_DIM, g0=g0, seed=7)
        slots = _make_slots(w, g0, self.BLOCK_SIZE)

        cache_ref = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)
        cache_iter = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)

        _write_once(self._fresh_writer(), k, v, slots, cache_ref, g0=g0)
        _write_iter_decode(self._fresh_writer(), k, v, slots, cache_iter, g0=g0)

        snap_ref = _read_cache_snapshot(cache_ref, slots)
        snap_iter = _read_cache_snapshot(cache_iter, slots)
        torch.testing.assert_close(snap_ref.k_fp8, snap_iter.k_fp8)
        torch.testing.assert_close(snap_ref.v_fp8, snap_iter.v_fp8)
        torch.testing.assert_close(snap_ref.k_scale, snap_iter.k_scale)
        torch.testing.assert_close(snap_ref.v_scale_cache, snap_iter.v_scale_cache)

    def test_tail_window_3_iter_matches_one_shot(self):
        w = 3
        g0 = 128
        k, v = _make_bf16_kv(w, self.NUM_KV_HEADS, self.HEAD_DIM, g0=g0, seed=99)
        slots = _make_slots(w, g0, self.BLOCK_SIZE)

        cache_ref = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)
        cache_iter = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)

        writer_iter = self._fresh_writer()
        _write_once(self._fresh_writer(), k, v, slots, cache_ref, g0=g0)
        _write_iter_decode(writer_iter, k, v, slots, cache_iter, g0=g0)

        snap_ref = _read_cache_snapshot(cache_ref, slots)
        snap_iter = _read_cache_snapshot(cache_iter, slots)
        torch.testing.assert_close(snap_ref.k_fp8, snap_iter.k_fp8)
        torch.testing.assert_close(snap_ref.v_fp8, snap_iter.v_fp8)
        torch.testing.assert_close(snap_ref.k_scale, snap_iter.k_scale)
        torch.testing.assert_close(snap_ref.v_scale_cache, snap_iter.v_scale_cache)
        self.assertIn(self.REQ_ID, writer_iter._tail_windows)
        self.assertEqual(writer_iter._tail_windows[self.REQ_ID].w, 3)

    def test_single_token_write_diverges_from_window(self):
        """Regression: writing only the last token must not match full-window quant."""
        w = 3
        g0 = 128
        k, v = _make_bf16_kv(w, self.NUM_KV_HEADS, self.HEAD_DIM, g0=g0, seed=11)
        slots = _make_slots(w, g0, self.BLOCK_SIZE)

        cache_ref = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)
        cache_bad = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)

        _write_once(self._fresh_writer(), k, v, slots, cache_ref, g0=g0)

        qk, qv, ks, vs = _quantize_kv_pair(k[-1:], v[-1:])
        _cpu_write_mxfp8(qk, qv, ks, vs, cache_bad, slots[-1:], 1)

        snap_ref = _read_cache_snapshot(cache_ref, slots)
        snap_bad = _read_cache_snapshot(cache_bad, slots[-1:])
        self.assertFalse(torch.equal(snap_ref.v_fp8[-1], snap_bad.v_fp8[0]))

    def test_tail_window_iter_progressive_matches_prefix_ref(self):
        w = 64
        g0 = 0
        k, v = _make_bf16_kv(w, self.NUM_KV_HEADS, self.HEAD_DIM, g0=g0, seed=2025)
        slots = _make_slots(w, g0, self.BLOCK_SIZE)

        for i in range(1, w + 1):
            cache_ref = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)
            cache_step = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)
            _write_once(self._fresh_writer(), k[:i], v[:i], slots[:i], cache_ref, g0=g0)
            _write_iter_decode(self._fresh_writer(), k[:i], v[:i], slots[:i], cache_step, g0=g0)
            snap_ref = _read_cache_snapshot(cache_ref, slots[:i])
            snap_step = _read_cache_snapshot(cache_step, slots[:i])
            torch.testing.assert_close(snap_ref.k_fp8, snap_step.k_fp8)
            torch.testing.assert_close(snap_ref.v_fp8, snap_step.v_fp8)
            torch.testing.assert_close(snap_ref.k_scale, snap_step.k_scale)

    def test_prune_tail_windows_removes_stale_req(self):
        writer = self._fresh_writer()
        g0 = 128
        k, v = _make_bf16_kv(2, self.NUM_KV_HEADS, self.HEAD_DIM, g0=g0, seed=1)
        slots = _make_slots(2, g0, self.BLOCK_SIZE)
        cache = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)
        _write_iter_decode(writer, k, v, slots, cache, g0=g0, req_id="stale-req")
        self.assertIn("stale-req", writer._tail_windows)
        writer.prune_tail_windows({"other-req"})
        self.assertNotIn("stale-req", writer._tail_windows)

    def test_tail_windows_isolated_per_request(self):
        writer = self._fresh_writer()
        cache = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)

        k_a, v_a = _make_bf16_kv(2, self.NUM_KV_HEADS, self.HEAD_DIM, g0=128, seed=3)
        k_b, v_b = _make_bf16_kv(2, self.NUM_KV_HEADS, self.HEAD_DIM, g0=256, seed=4)
        slots_a = _make_slots(2, 128, self.BLOCK_SIZE)
        slots_b = _make_slots(2, 256, self.BLOCK_SIZE)

        k_batch = torch.cat([k_a, k_b], dim=0)
        v_batch = torch.cat([v_a, v_b], dim=0)
        writer.write_batch(
            k_batch,
            v_batch,
            cache,
            num_tokens=4,
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.long),
            req_ids=["req-a", "req-b"],
            num_computed_tokens_cpu=torch.tensor([128, 256], dtype=torch.long),
            slot_mapping=torch.cat([slots_a, slots_b]),
            quantize_kv=_quantize_kv_pair,
            write_mxfp8=_cpu_write_mxfp8,
            block_size=self.BLOCK_SIZE,
        )
        self.assertIn("req-a", writer._tail_windows)
        self.assertIn("req-b", writer._tail_windows)

        cache_a_ref = _alloc_kv_cache(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM)
        _write_once(self._fresh_writer(), k_a, v_a, slots_a, cache_a_ref, g0=128, req_id="req-a")
        snap_a = _read_cache_snapshot(cache, slots_a)
        snap_a_ref = _read_cache_snapshot(cache_a_ref, slots_a)
        torch.testing.assert_close(snap_a.k_fp8, snap_a_ref.k_fp8)
        torch.testing.assert_close(snap_a.v_fp8, snap_a_ref.v_fp8)


if __name__ == "__main__":
    unittest.main()
