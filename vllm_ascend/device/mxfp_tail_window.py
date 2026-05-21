# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm_ascend.device.mxfp_compat import MXFP_KV_SCALE_GROUP_SIZE


class MxfpTailWindowWriter:
    """Per-layer tail-group bf16 V window for C8_MXFP axis=0 V quantization."""

    def __init__(
        self,
        max_num_seqs: int,
        num_kv_heads: int,
        v_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.win_v = torch.zeros(
            max_num_seqs,
            MXFP_KV_SCALE_GROUP_SIZE,
            num_kv_heads,
            v_dim,
            device=device,
            dtype=dtype,
        )
        self.win_slots = torch.zeros(
            max_num_seqs,
            MXFP_KV_SCALE_GROUP_SIZE,
            dtype=torch.int64,
            device=device,
        )
        self.win_lens = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)

    @classmethod
    def create(
        cls,
        max_num_seqs: int,
        num_kv_heads: int,
        v_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MxfpTailWindowWriter:
        return cls(
            max_num_seqs,
            num_kv_heads,
            v_dim,
            device=device,
            dtype=dtype,
        )

    def prune(self, num_reqs: int) -> None:
        """Clear tail-window state for inactive batch rows."""
        if num_reqs < self.win_lens.numel():
            self.win_lens[num_reqs:].zero_()

    def save_prefill_tail(
        self,
        req_idx: int,
        value: torch.Tensor,
        slots: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Save trailing partial V scale group after prefill."""
        t = num_tokens % MXFP_KV_SCALE_GROUP_SIZE
        if t > 0:
            self.win_v[req_idx, :t].copy_(value[-t:])
            self.win_slots[req_idx, :t].copy_(slots[-t:].to(torch.int64))
            self.win_lens[req_idx] = t
        else:
            self.win_lens[req_idx] = 0

    def refresh_decode_append(
        self,
        req_idx: int,
        value_token: torch.Tensor,
        slot: torch.Tensor | int,
    ) -> int:
        """Append one decode token; return window length to re-quantize."""
        t = int(self.win_lens[req_idx].item())
        self.win_v[req_idx, t].copy_(value_token)
        slot_val = int(slot.item()) if torch.is_tensor(slot) else int(slot)
        self.win_slots[req_idx, t] = slot_val
        if t == MXFP_KV_SCALE_GROUP_SIZE - 1:
            quant_len = MXFP_KV_SCALE_GROUP_SIZE
        else:
            quant_len = t + 1
        self.win_lens[req_idx] = (t + 1) % MXFP_KV_SCALE_GROUP_SIZE
        return quant_len
