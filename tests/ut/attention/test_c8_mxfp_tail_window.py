# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

import torch

from vllm_ascend.device.mxfp_compat import MXFP_KV_SCALE_GROUP_SIZE, scatter_mxfp_v_cache
from vllm_ascend.device.mxfp_tail_window import MxfpTailWindowWriter


class TestMxfpTailWindowWriter(unittest.TestCase):
    def setUp(self):
        self.writer = MxfpTailWindowWriter.create(
            max_num_seqs=4,
            num_kv_heads=2,
            v_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def test_save_prefill_tail_partial_group(self):
        req = 1
        num_tokens = 67
        value = torch.arange(num_tokens * 2 * 8, dtype=torch.float32).view(num_tokens, 2, 8)
        slots = torch.arange(100, 100 + num_tokens, dtype=torch.int64)

        self.writer.save_prefill_tail(req, value, slots, num_tokens)

        t = num_tokens % MXFP_KV_SCALE_GROUP_SIZE
        self.assertEqual(int(self.writer.win_lens[req].item()), t)
        torch.testing.assert_close(self.writer.win_v[req, :t], value[-t:])
        torch.testing.assert_close(self.writer.win_slots[req, :t], slots[-t:])

    def test_save_prefill_tail_full_group_clears_window(self):
        req = 0
        num_tokens = 128
        value = torch.ones(num_tokens, 2, 8)
        slots = torch.zeros(num_tokens, dtype=torch.int64)

        self.writer.save_prefill_tail(req, value, slots, num_tokens)

        self.assertEqual(int(self.writer.win_lens[req].item()), 0)

    def test_refresh_decode_append_grows_window(self):
        req = 0
        token = torch.full((2, 8), 3.0)
        quant_len = self.writer.refresh_decode_append(req, token, torch.tensor(42))

        self.assertEqual(quant_len, 1)
        self.assertEqual(int(self.writer.win_lens[req].item()), 1)
        self.assertEqual(int(self.writer.win_slots[req, 0].item()), 42)
        torch.testing.assert_close(self.writer.win_v[req, 0], token)

    def test_refresh_decode_append_seals_group_at_64(self):
        req = 2
        self.writer.win_lens[req] = MXFP_KV_SCALE_GROUP_SIZE - 1
        token = torch.full((2, 8), 7.0)
        quant_len = self.writer.refresh_decode_append(req, token, 999)

        self.assertEqual(quant_len, MXFP_KV_SCALE_GROUP_SIZE)
        self.assertEqual(int(self.writer.win_lens[req].item()), 0)

    def test_prune_clears_inactive_rows(self):
        self.writer.win_lens[0] = 5
        self.writer.win_lens[1] = 10
        self.writer.win_lens[2] = 3
        self.writer.prune(2)
        self.assertEqual(int(self.writer.win_lens[0].item()), 5)
        self.assertEqual(int(self.writer.win_lens[1].item()), 10)
        self.assertEqual(int(self.writer.win_lens[2].item()), 0)


class TestScatterMxfpVCache(unittest.TestCase):
    def test_scatter_mxfp_v_cache_paged_layout(self):
        block_size = 64
        num_blocks = 2
        num_kv_heads = 2
        v_dim = 4
        value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, v_dim)
        quant_value = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
            ]
        )
        slot_mapping = torch.tensor([1, 2], dtype=torch.int64)

        scatter_mxfp_v_cache(quant_value, value_cache, slot_mapping, block_size)

        torch.testing.assert_close(value_cache[0, 1], quant_value[0])
        torch.testing.assert_close(value_cache[0, 2], quant_value[1])


if __name__ == "__main__":
    unittest.main()
