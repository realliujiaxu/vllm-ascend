from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.scatter_pa_kv_cache_with_k_scale import scatter_pa_kv_cache_with_k_scale


def _inputs() -> tuple[torch.Tensor, ...]:
    key = torch.empty((1, 2, 64), dtype=torch.float8_e4m3fn)
    value = torch.empty_like(key)
    key_cache = torch.empty((2, 2, 16, 64), dtype=torch.float8_e4m3fn)
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.tensor([3], dtype=torch.int32)
    key_scale = torch.ones((1, 2), dtype=torch.float32)
    key_scale_cache = torch.empty((2, 2, 16, 1), dtype=torch.float32)
    return (
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key_scale,
        key_scale_cache,
    )


def test_calls_cann_ops_transformer_binding() -> None:
    custom_op = MagicMock()
    custom_module = SimpleNamespace(scatter_pa_kv_cache_with_k_scale=custom_op)

    with patch.dict("sys.modules", {"cann_ops_transformer": custom_module}):
        inputs = _inputs()
        scatter_pa_kv_cache_with_k_scale(*inputs)

    custom_op.assert_called_once_with(*inputs, cache_layout="BNBD")


def test_reports_missing_custom_operator_package() -> None:
    with (
        patch.dict("sys.modules", {"cann_ops_transformer": None}),
        pytest.raises(RuntimeError, match="cann_ops_transformer"),
    ):
        scatter_pa_kv_cache_with_k_scale(*_inputs())
