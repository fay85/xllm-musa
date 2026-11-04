# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""Compare MATE adapters with independent operation-level GPU references."""

import pytest
import torch
import torch_musa  # noqa: F401

from xllm.python.kernels_musa import glm_reference, mate_ops


@pytest.mark.parametrize('width', [128, 512, 4096])
def test_rmsnorm(width: int) -> None:
    torch.manual_seed(1104)
    value = torch.randn((1, 3, width), device='musa', dtype=torch.bfloat16)
    weight = torch.randn(width, device='musa', dtype=torch.bfloat16)
    expected = glm_reference.rms_norm(value, weight, 1e-5)
    actual = mate_ops.rms_norm(value, weight, 1e-5)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.01, atol=0.001)


@pytest.mark.parametrize('tokens', [1, 17])
def test_hyperconnection(tokens: int) -> None:
    torch.manual_seed(1104)
    hidden = torch.randn((1, tokens, 4, 4096), device='musa', dtype=torch.bfloat16)
    weight = torch.randn((24, 16384), device='musa', dtype=torch.float32) * 0.01
    scale = torch.tensor([0.5, 0.5, 0.5], device='musa')
    base = torch.randn(24, device='musa') * 0.1
    args = (hidden, weight, scale, base, 4, 20, 1e-5, 1e-6)
    expected = glm_reference.hc_pre_fused(*args)
    actual = mate_ops.hc_pre_fused(*args)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.float(), reference.float(), rtol=0.02, atol=0.015)
