"""Single-op smoke test for ScatterPaKvCacheWithKScale on an Ascend device.

Run this file only after installing the custom 4-in-1 operator package and
exporting its op_api library directory through LD_LIBRARY_PATH.
"""

import inspect
import os

import torch
import torch_npu  # noqa: F401

# Load torch_npu before the separately installed custom-op binding.
# isort: off
import cann_ops_transformer

# isort: on

NUM_BLOCKS = 2
NUM_HEADS = 2
BLOCK_SIZE = 16
HEAD_SIZE = 64
NUM_TOKENS = 3
CACHE_LAYOUT = "BNBD"
FIRST_SLOTS = (0, 17, 5)
SECOND_SLOTS = (1, 18, 6)


def _print_binding_info() -> None:
    op = cann_ops_transformer.scatter_pa_kv_cache_with_k_scale
    print(f"cann_ops_transformer: {getattr(cann_ops_transformer, '__file__', '<built-in>')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
    print(f"operator: {op!r}")
    try:
        print(f"signature: {inspect.signature(op)}")
    except (TypeError, ValueError):
        print("signature: unavailable for this compiled binding")


def _assert_slot(
    cache: torch.Tensor,
    expected: torch.Tensor,
    slot: int,
) -> None:
    block_index = slot // BLOCK_SIZE
    block_offset = slot % BLOCK_SIZE
    actual = cache[block_index, :, block_offset, :]
    torch.testing.assert_close(actual.float().cpu(), expected.float().cpu(), rtol=0, atol=0)


def _print_cache_relationships(
    prefix: str,
    cache_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cache_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    names = ("K cache", "V cache", "K-scale cache")
    for name, cache_in, cache_out in zip(names, cache_inputs, cache_outputs):
        print(
            f"{prefix} {name}: output shape={tuple(cache_out.shape)}, "
            f"dtype={cache_out.dtype}, "
            f"same_tensor={cache_out is cache_in}, "
            f"same_storage={cache_out.data_ptr() == cache_in.data_ptr()}"
        )


def _assert_written_slots(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_scale: torch.Tensor,
    slots: tuple[int, ...],
) -> None:
    for token_index, slot in enumerate(slots):
        _assert_slot(key_cache, key[token_index], slot)
        _assert_slot(value_cache, value[token_index], slot)
        block_index = slot // BLOCK_SIZE
        block_offset = slot % BLOCK_SIZE
        actual_scale = key_scale_cache[block_index, :, block_offset, 0]
        torch.testing.assert_close(
            actual_scale.cpu(),
            key_scale[token_index].cpu(),
            rtol=0,
            atol=0,
        )


def main() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("No Ascend NPU is visible to torch_npu")

    torch.npu.set_device(0)
    _print_binding_info()

    key_source = torch.arange(
        NUM_TOKENS * NUM_HEADS * HEAD_SIZE,
        dtype=torch.float32,
        device="npu",
    ).reshape(NUM_TOKENS, NUM_HEADS, HEAD_SIZE)
    key = ((key_source % 31) - 15).to(torch.float8_e4m3fn)
    value = ((key_source % 29) - 14).to(torch.float8_e4m3fn)
    key_cache = torch.zeros(
        (NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE),
        dtype=torch.float8_e4m3fn,
        device="npu",
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.tensor(FIRST_SLOTS, dtype=torch.int32, device="npu")
    key_scale = torch.tensor(
        [[1.0, 1.25], [1.5, 1.75], [2.0, 2.25]],
        dtype=torch.float32,
        device="npu",
    )
    key_scale_cache = torch.zeros(
        (NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, 1),
        dtype=torch.float32,
        device="npu",
    )

    result = cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key_scale,
        key_scale_cache,
        cache_layout=CACHE_LAYOUT,
    )
    torch.npu.synchronize()

    if not isinstance(result, tuple) or len(result) != 3:
        raise AssertionError(
            "Expected the operator to return "
            "(key_cache, value_cache, key_scale_cache), "
            f"but got {type(result).__name__}: {result!r}"
        )

    key_cache_out, value_cache_out, key_scale_cache_out = result
    first_cache_inputs = (key_cache, value_cache, key_scale_cache)
    first_cache_outputs = (
        key_cache_out,
        value_cache_out,
        key_scale_cache_out,
    )
    _print_cache_relationships("first call", first_cache_inputs, first_cache_outputs)
    _assert_written_slots(
        key_cache_out,
        value_cache_out,
        key_scale_cache_out,
        key,
        value,
        key_scale,
        FIRST_SLOTS,
    )

    assert torch.count_nonzero(key_cache_out.float()).item() == torch.count_nonzero(key.float()).item()
    assert torch.count_nonzero(value_cache_out.float()).item() == torch.count_nonzero(value.float()).item()
    assert torch.count_nonzero(key_scale_cache_out).item() == key_scale.numel()
    print("PASS: the first call returned the expected K, V, and K-scale values.")

    second_key_source = key_source + 1000
    second_key = ((second_key_source % 37) - 18).to(torch.float8_e4m3fn)
    second_value = ((second_key_source % 41) - 20).to(torch.float8_e4m3fn)
    second_slot_mapping = torch.tensor(
        SECOND_SLOTS,
        dtype=torch.int32,
        device="npu",
    )
    second_key_scale = torch.tensor(
        [[3.0, 3.25], [3.5, 3.75], [4.0, 4.25]],
        dtype=torch.float32,
        device="npu",
    )

    second_result = cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(
        second_key,
        second_value,
        key_cache_out,
        value_cache_out,
        second_slot_mapping,
        second_key_scale,
        key_scale_cache_out,
        cache_layout=CACHE_LAYOUT,
    )
    torch.npu.synchronize()

    if not isinstance(second_result, tuple) or len(second_result) != 3:
        raise AssertionError(
            "Expected the second operator call to return "
            "(key_cache, value_cache, key_scale_cache), "
            f"but got {type(second_result).__name__}: {second_result!r}"
        )

    second_key_cache_out, second_value_cache_out, second_key_scale_cache_out = second_result
    second_cache_outputs = (
        second_key_cache_out,
        second_value_cache_out,
        second_key_scale_cache_out,
    )
    _print_cache_relationships(
        "second call",
        first_cache_outputs,
        second_cache_outputs,
    )

    _assert_written_slots(
        second_key_cache_out,
        second_value_cache_out,
        second_key_scale_cache_out,
        key,
        value,
        key_scale,
        FIRST_SLOTS,
    )
    _assert_written_slots(
        second_key_cache_out,
        second_value_cache_out,
        second_key_scale_cache_out,
        second_key,
        second_value,
        second_key_scale,
        SECOND_SLOTS,
    )

    expected_key_nonzero = torch.count_nonzero(key.float()).item()
    expected_key_nonzero += torch.count_nonzero(second_key.float()).item()
    expected_value_nonzero = torch.count_nonzero(value.float()).item()
    expected_value_nonzero += torch.count_nonzero(second_value.float()).item()
    assert torch.count_nonzero(second_key_cache_out.float()).item() == expected_key_nonzero
    assert torch.count_nonzero(second_value_cache_out.float()).item() == expected_value_nonzero
    assert torch.count_nonzero(second_key_scale_cache_out).item() == (key_scale.numel() + second_key_scale.numel())
    print(
        "PASS: the second call preserved all first-call slots and added "
        "the expected K, V, and K-scale values at all second-call slots."
    )


if __name__ == "__main__":
    main()
