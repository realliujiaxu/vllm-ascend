#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import torch


def scatter_pa_kv_cache_with_k_scale(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_scale: torch.Tensor,
    key_scale_cache: torch.Tensor,
) -> None:
    """Call the separately installed CANN transformer custom operator."""
    try:
        import cann_ops_transformer
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax M3 FP8 KV cache requires the cann_ops_transformer "
            "Python package and its custom operator libraries. Install the "
            "4-in-1 operator package and add its op_api/lib directory to "
            "LD_LIBRARY_PATH before starting Python."
        ) from exc

    cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key_scale,
        key_scale_cache,
        cache_layout="BNBD",
    )
