# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""Independent offline FP8 scale and tensor-parallel slicing checks."""

import pytest
import torch

from xllm.python.models.fp8_block_weight import (
    dequantize_fp8_block_shard_to_bf16,
    dequantize_fp8_block_to_bf16,
)


@pytest.mark.parametrize("shape", [(256, 512), (258, 260)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("rank", [0, 1])
def test_tp_conversion_matches_independent_block_lookup(
    shape: tuple[int, int], dim: int, rank: int
) -> None:
    generator = torch.Generator().manual_seed(1104)
    weight = torch.randint(-4, 5, shape, generator=generator).to(torch.float8_e4m3fn)
    scales = torch.rand(
        ((shape[0] + 127) // 128, (shape[1] + 127) // 128),
        generator=generator,
    )
    row_blocks = torch.arange(shape[0]) // 128
    column_blocks = torch.arange(shape[1]) // 128
    reference = (
        weight.float() * scales[row_blocks[:, None], column_blocks[None, :]]
    ).bfloat16()
    torch.testing.assert_close(
        dequantize_fp8_block_to_bf16(weight, scales), reference, rtol=0, atol=0
    )
    width = shape[dim] // 2
    actual = dequantize_fp8_block_shard_to_bf16(weight, scales, dim, 2, rank)
    torch.testing.assert_close(
        actual, reference.narrow(dim, rank * width, width), rtol=0, atol=0
    )
    assert actual.is_contiguous()


def test_invalid_scale_in_another_rank_is_rejected() -> None:
    weight = torch.ones((256, 256)).to(torch.float8_e4m3fn)
    scales = torch.ones((2, 2), dtype=torch.float32)
    scales[1, 1] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        dequantize_fp8_block_shard_to_bf16(weight, scales, 0, 2, 0)


def test_scale_is_multiplicative_and_not_ue8m0() -> None:
    weight = torch.full((128, 128), 2.0).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.25]], dtype=torch.float32)
    actual = dequantize_fp8_block_to_bf16(weight, scales)
    torch.testing.assert_close(actual, torch.full_like(actual, 0.5), rtol=0, atol=0)
    with pytest.raises(ValueError, match='FP32'):
        dequantize_fp8_block_to_bf16(weight, scales.to(torch.uint8))
