import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()


def swiglu_no_interleaved_with_alpha_and_limit(
    x: torch.Tensor,
    gemm1_alpha: float,
    gemm1_limit: float,
) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


def _assert_quantized_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.view(torch.uint8).cpu(), expected.view(torch.uint8).cpu(), atol=1, rtol=5e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_swiglu_mx_quant_matches_dynamic_mx_quant(dtype: torch.dtype):
    if not hasattr(torch.ops._C_ascend, "swiglu_mx_quant"):
        pytest.skip("swiglu_mx_quant custom op is not available")

    torch.manual_seed(0)
    x = torch.randn((17, 128), dtype=dtype, device="npu")
    dst_type = torch.float8_e4m3fn
    gemm1_alpha = 1.702
    gemm1_limit = 7.0

    golden_act = swiglu_no_interleaved_with_alpha_and_limit(x, gemm1_alpha, gemm1_limit)
    expected, expected_scale = torch_npu.npu_dynamic_mx_quant(golden_act, dst_type=dst_type)

    actual, actual_scale = torch.ops._C_ascend.swiglu_mx_quant(
        x,
        None,
        dst_type,
        -1,
        True,
        1,
        gemm1_limit,
        gemm1_alpha,
        1.0,
        0,
        -1,
        "rint",
        0,
        0.0,
    )

    _assert_quantized_equal(actual, expected)
    _assert_quantized_equal(actual_scale, expected_scale)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
