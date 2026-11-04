# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""Numerical/state tests on real MUSA, outside the CPU-stub test package."""

import pytest
import torch
import torch_musa  # noqa: F401

from xllm.python.kernels_musa.kda import recurrent_kda


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor,
    beta: torch.Tensor, a_log: torch.Tensor, bias: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # CPU FP64 is the test oracle only. Inference stays on MUSA.
    q, k, v, gate, beta, a_log, bias, state = (
        t.detach().cpu().double() for t in (q, k, v, gate, beta, a_log, bias, initial)
    )
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    # MATE's chunk preparation stores normalized Q/K in the input dtype.
    q, k = q.to(torch.bfloat16).double(), k.to(torch.bfloat16).double()
    decay = (-5 * (a_log.exp()[None, None, :, None] * (gate + bias)).sigmoid()).exp()
    beta = beta.sigmoid()
    output = []
    for step in range(q.shape[1]):
        state = state * decay[:, step, :, None, :]
        residual = v[:, step] - torch.einsum('bhvk,bhk->bhv', state, k[:, step])
        state = state + torch.einsum('bhv,bhk->bhvk', residual * beta[:, step, :, None], k[:, step])
        output.append(torch.einsum('bhvk,bhk->bhv', state, q[:, step] * 128**-0.5))
    return torch.stack(output, 1), state


@pytest.mark.parametrize('tokens', [1, 17, 65, 129, 257])
def test_mate_kda_matches_recurrence_and_preserves_input(tokens: int) -> None:
    torch.manual_seed(1104)
    shape = (1, tokens, 8, 128)
    q, k, v, gate = [torch.randn(shape, dtype=torch.bfloat16, device='musa') for _ in range(4)]
    beta = torch.randn(shape[:3], dtype=torch.bfloat16, device='musa')
    a_log = torch.zeros(8, dtype=torch.float32, device='musa')
    bias = torch.zeros((8, 128), dtype=torch.float32, device='musa')
    initial = torch.randn((1, 8, 128, 128), device='musa') * 0.02
    saved = initial.clone()
    actual, final = recurrent_kda(q, k, v, gate, beta, a_log, bias, initial)
    expected, expected_state = _reference(q, k, v, gate, beta, a_log, bias, initial)
    torch.musa.synchronize()
    torch.testing.assert_close(initial, saved, rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=0.03, atol=0.001)
    torch.testing.assert_close(final.cpu().double(), expected_state, rtol=0.03, atol=0.008)
    if tokens > 1:
        first, state = recurrent_kda(q[:, :-1], k[:, :-1], v[:, :-1], gate[:, :-1], beta[:, :-1], a_log, bias, initial)
        last, state = recurrent_kda(q[:, -1:], k[:, -1:], v[:, -1:], gate[:, -1:], beta[:, -1:], a_log, bias, state)
        torch.testing.assert_close(last.float(), actual[:, -1:].float(), rtol=0.03, atol=0.001)
        torch.testing.assert_close(state, final, rtol=0.03, atol=0.008)
