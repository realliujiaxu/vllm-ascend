#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute

ASCEND_ZERO_EXPERT_OUTPUT_ATTR = "_ascend_zero_expert_output"


@dataclass(frozen=True)
class ZeroExpertConfig:
    enabled: bool
    zero_expert_type: str | None
    num_logical_experts: int
    zero_expert_num: int


def _get_hf_config(layer: torch.nn.Module) -> object | None:
    vllm_config = getattr(layer, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None) if vllm_config is not None else None
    return getattr(model_config, "hf_config", None) if model_config is not None else None


def bind_zero_expert_config_to_layer(
    layer: torch.nn.Module,
    router: object | None = None,
    num_logical_experts: int | None = None,
) -> ZeroExpertConfig:
    """Read zero-expert settings from HF config and write them onto ``layer``.

    Sources:
    - ``zero_expert_num`` / ``zero_expert_type`` ← ``hf_config``
    - ``num_logical_experts`` ← ``router.num_logical_experts`` or layer fields
    """
    logical = num_logical_experts
    if logical is None:
        logical = getattr(layer, "logical_num_experts", None)
    if logical is None:
        moe_config = getattr(layer, "moe_config", None)
        logical = getattr(moe_config, "num_logical_experts", None) if moe_config is not None else None
    if logical is None:
        logical = -1

    router_logical = getattr(router, "num_logical_experts", None) if router is not None else None
    if router_logical is not None:
        logical = int(router_logical)

    hf_config = _get_hf_config(layer)
    zero_expert_num = int(getattr(hf_config, "zero_expert_num", 0) or 0) if hf_config is not None else 0
    zero_expert_type = getattr(hf_config, "zero_expert_type", None) if hf_config is not None else None
    if zero_expert_num <= 0:
        zero_expert_num = int(getattr(layer, "zero_expert_num", 0) or 0)
    if zero_expert_type is None:
        zero_expert_type = getattr(layer, "zero_expert_type", None)

    enabled = zero_expert_num > 0 and zero_expert_type is not None
    config = ZeroExpertConfig(
        enabled=enabled,
        zero_expert_type=zero_expert_type if enabled else None,
        num_logical_experts=int(logical),
        zero_expert_num=zero_expert_num if enabled else 0,
    )
    layer.zero_expert_num = config.zero_expert_num
    layer.zero_expert_type = config.zero_expert_type
    setattr(layer, ASCEND_ZERO_EXPERT_OUTPUT_ATTR, None)
    return config


def clear_zero_expert_output(layer: torch.nn.Module) -> None:
    setattr(layer, ASCEND_ZERO_EXPERT_OUTPUT_ATTR, None)


def store_zero_expert_output(layer: torch.nn.Module, output: torch.Tensor | None) -> None:
    setattr(layer, ASCEND_ZERO_EXPERT_OUTPUT_ATTR, output)


def take_zero_expert_output(layer: torch.nn.Module) -> torch.Tensor | None:
    output = getattr(layer, ASCEND_ZERO_EXPERT_OUTPUT_ATTR, None)
    setattr(layer, ASCEND_ZERO_EXPERT_OUTPUT_ATTR, None)
    return output


def maybe_zero_experts_compute(
    layer: torch.nn.Module,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_logical_experts: int,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask zero-expert routes and store identity output on ``layer``.

    AscendMoERunner adds that output in ``_maybe_add_zero_expert_output``.
    """
    clear_zero_expert_output(layer)
    zero_expert_num = getattr(layer, "zero_expert_num", 0)
    zero_expert_type = getattr(layer, "zero_expert_type", None)
    if not (zero_expert_num > 0 and zero_expert_type is not None):
        return topk_ids, topk_weights

    topk_ids, topk_weights, zero_result = zero_experts_compute(
        expert_indices=topk_ids,
        expert_scales=topk_weights,
        num_experts=num_logical_experts,
        zero_expert_type=zero_expert_type,
        hidden_states=hidden_states,
    )
    store_zero_expert_output(layer, zero_result)
    return topk_ids, topk_weights
