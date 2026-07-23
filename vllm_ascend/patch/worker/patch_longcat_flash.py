# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# 1) Skip top-level MTP tensors when loading without speculative MTP.
# 2) Align LongCat dual-stream dtypes before RMSNorm (NPU fused add+rms
#    requires matching dtypes; MLA/MLP can promote hidden to float32 while
#    residual stays bfloat16).
from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from vllm.model_executor.models.longcat_flash import FlashDecoderLayer, FlashModel

logger = logging.getLogger("vllm_ascend.patch.longcat_flash")

_orig_flash_load_weights = FlashModel.load_weights
_orig_decoder_forward = FlashDecoderLayer.forward

# Cap noisy dtype diagnostics (per process).
_DTYPE_LOG_BUDGET = 8


def _should_skip_mtp_weight(name: str) -> bool:
    return name.startswith("mtp.") or ".mtp." in name


def _patched_flash_load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    def _filtered() -> Iterable[tuple[str, torch.Tensor]]:
        for name, loaded_weight in weights:
            if _should_skip_mtp_weight(name):
                continue
            yield name, loaded_weight

    return _orig_flash_load_weights(self, _filtered())


def _target_dtype(layer: FlashDecoderLayer) -> torch.dtype:
    weight = getattr(layer.input_layernorm[0], "weight", None)
    if weight is not None:
        return weight.dtype
    return torch.bfloat16


def _maybe_log_dtype(
    layer: FlashDecoderLayer,
    tag: str,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> None:
    global _DTYPE_LOG_BUDGET
    if _DTYPE_LOG_BUDGET <= 0:
        return
    hs_dt = hidden_states.dtype
    res_dt = None if residual is None else residual.dtype
    w_dt = _target_dtype(layer)
    if residual is not None and (hs_dt != res_dt or hs_dt != w_dt or res_dt != w_dt):
        _DTYPE_LOG_BUDGET -= 1
        logger.warning(
            "[longcat dtype] layer=%s %s hidden=%s residual=%s norm_weight=%s "
            "shape_h=%s shape_r=%s remaining_logs=%s",
            getattr(layer, "layer_idx", "?"),
            tag,
            hs_dt,
            res_dt,
            w_dt,
            tuple(hidden_states.shape),
            None if residual is None else tuple(residual.shape),
            _DTYPE_LOG_BUDGET,
        )
    elif _DTYPE_LOG_BUDGET == 8:
        # First call: always log once for baseline visibility.
        _DTYPE_LOG_BUDGET -= 1
        logger.info(
            "[longcat dtype] layer=%s %s hidden=%s residual=%s norm_weight=%s "
            "(baseline)",
            getattr(layer, "layer_idx", "?"),
            tag,
            hs_dt,
            res_dt,
            w_dt,
        )


def _cast_pair(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if hidden_states.dtype != dtype:
        hidden_states = hidden_states.to(dtype)
    if residual is not None and residual.dtype != dtype:
        residual = residual.to(dtype)
    return hidden_states, residual


def _patched_decoder_forward(
    self: FlashDecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = _target_dtype(self)
    _maybe_log_dtype(self, "enter", hidden_states, residual)
    hidden_states, residual = _cast_pair(hidden_states, residual, dtype)

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm[0](hidden_states)
    else:
        hidden_states, residual = self.input_layernorm[0](hidden_states, residual)

    hidden_states = self.self_attn[0](
        positions=positions,
        hidden_states=hidden_states,
        llama_4_scaling=None,
    )
    _maybe_log_dtype(self, "after_attn0", hidden_states, residual)
    hidden_states, residual = _cast_pair(hidden_states, residual, dtype)

    hidden_states, residual = self.post_attention_layernorm[0](
        hidden_states, residual
    )

    # moe
    hidden_states_copy = hidden_states.clone()
    moe_hidden_states = self.mlp(hidden_states_copy)
    _maybe_log_dtype(self, "after_moe", moe_hidden_states, residual)

    # first mlp
    hidden_states = self.mlps[0](hidden_states)
    _maybe_log_dtype(self, "after_mlp0", hidden_states, residual)
    hidden_states, residual = _cast_pair(hidden_states, residual, dtype)

    hidden_states, residual = self.input_layernorm[1](hidden_states, residual)

    # second_attn
    hidden_states = self.self_attn[1](
        positions=positions,
        hidden_states=hidden_states,
        llama_4_scaling=None,
    )
    _maybe_log_dtype(self, "after_attn1", hidden_states, residual)
    hidden_states, residual = _cast_pair(hidden_states, residual, dtype)

    hidden_states, residual = self.post_attention_layernorm[1](
        hidden_states, residual
    )

    # second_mlp
    hidden_states = self.mlps[1](hidden_states)
    _maybe_log_dtype(self, "after_mlp1", hidden_states, residual)

    moe_hidden_states = moe_hidden_states.to(hidden_states.dtype)
    hidden_states = hidden_states + moe_hidden_states
    _maybe_log_dtype(self, "exit", hidden_states, residual)
    hidden_states, residual = _cast_pair(hidden_states, residual, dtype)

    return hidden_states, residual


FlashModel.load_weights = _patched_flash_load_weights
FlashDecoderLayer.forward = _patched_decoder_forward
