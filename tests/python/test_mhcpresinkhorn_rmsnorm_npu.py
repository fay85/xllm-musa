# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU tests for the PyPTO-Pro mhcpresinkhorn_rmsnorm fused operator.

The host-side contract (argument validation, workspace cache, launch buffers)
is covered by ``test_mhcpresinkhorn_rmsnorm.py`` on CPU.  This module runs the
real kernel on an Ascend device and compares it against a pure-torch
re-implementation of the fused chain:

1. ``mix = x @ hc_fn``            (bf16 inputs, FP32 output; K = 4 * 4096 split
   over cores)
2. ``r = 1 / sqrt(mean(x^2) + eps)`` per token over all ``4 * 4096`` lanes
3. ``z = mix * r * hc_scale + hc_base``
4. ``pp = sigmoid(z) * pmul + eps`` for the first 8 lanes (4 pre + 4 post),
   ``z`` for the 16 comb_flag lanes
5. ``y = bf16(sum_m pre[m] * x[m])`` then ``y_norm = y * rstd * gamma``
6. Sinkhorn normalization (20 iterations) over the 4x4 comb matrix
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

torch_npu = pytest.importorskip(
    "torch_npu",
    reason="mhcpresinkhorn_rmsnorm tests require torch_npu",
)
pytest.importorskip(
    "pypto_pro",
    reason="mhcpresinkhorn_rmsnorm tests require the pypto_pro package",
)

_MODULE_NAME = "xllm.python.kernels_npu.pypto_pro_mhcpresinkhorn_rmsnorm_npu"
_SOURCE = (Path(__file__).parents[2] / "xllm" / "python" / "kernels_npu"
           / "pypto-pro" / "mhcpresinkhorn_rmsnorm.py")


def _load_operator():
    """Load the operator module from its file path.

    ``pypto-pro`` is not a valid Python package name, so the module cannot be
    imported by name.
    """
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


op = _load_operator()

HC = op.HC
D = op.D
MIX_HC = op.MIX_HC
HC_D = op.HC_D
EPS = op.EPS
RMS_EPS = op.RMS_EPS
SINKHORN_ITERS = op.SINKHORN_ITERS
_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))


def _pmul(device: torch.device) -> torch.Tensor:
    """Per-lane multiplier used by the kernel (4x1, 4x2, 16x1)."""
    return torch.tensor([1.0] * 4 + [2.0] * 4 + [1.0] * 16,
                        dtype=torch.float32, device=device)


def _reference(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 reference of the whole fused chain."""
    t = x.shape[0]
    x_fp32 = x.float()
    x_2d = x_fp32.reshape(t, HC_D)

    # 1. mix = x @ hc_fn with fp32 accumulation (the Cube output is FP32).
    mix = x_fp32.reshape(t, HC_D) @ hc_fn.float().T

    # 2. r = 1 / sqrt(mean(x^2) + eps) over the whole 4 * 4096 row.
    ssq = x_2d.square().sum(dim=-1, keepdim=True)
    r = torch.rsqrt(ssq * (1.0 / HC_D) + EPS)

    # 3. z = mix * r * scale + base; scale is broadcast per 4/4/16 lanes.
    scale = torch.cat([
        hc_scale[0].expand(4),
        hc_scale[1].expand(4),
        hc_scale[2].expand(16),
    ])
    z = mix * r * scale + hc_base

    # 4. sigmoid + pmul + eps on lanes 0-7, linear comb_flag on lanes 8-23.
    sig = torch.sigmoid(z[:, :8])
    pp = torch.cat([sig * _pmul(sig.device)[:8] + EPS, z[:, 8:]], dim=-1)

    # 5. y = bf16(sum_m pre[m] * x[m]); RmsNorm reads that bf16 y.
    y = torch.einsum("tm,tmd->td", pp[:, :HC], x_fp32).to(torch.bfloat16)
    rstd = torch.rsqrt(y.float().square().mean(dim=-1, keepdim=True) + RMS_EPS)
    y_norm = (y.float() * rstd * gamma.float()).to(torch.bfloat16)

    # 6. Sinkhorn over comb_flag lanes 8-23 (row softmax, then 20 iterations).
    comb = pp[:, 2 * HC:].reshape(t, HC, HC).clone()
    comb = torch.softmax(comb, dim=-1) + EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)

    post = pp[:, HC:2 * HC].contiguous()
    return y_norm, rstd, post, comb


def _inputs(t: int, seed: int = 2026) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return dict(
        x=torch.randn(t, HC, D, dtype=torch.bfloat16, device="npu"),
        hc_fn=torch.randn(MIX_HC, HC_D, dtype=torch.bfloat16, device="npu") * 0.02,
        hc_scale=torch.randn(3, dtype=torch.float32, device="npu"),
        hc_base=torch.randn(MIX_HC, dtype=torch.float32, device="npu"),
        gamma=torch.randn(D, dtype=torch.bfloat16, device="npu"),
    )


def _call_with_comb_canary(
        args: dict[str, torch.Tensor], t: int
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None, int]:
    """Run the operator with an extra canary block right after comb16.

    ``comb16`` is allocated inside the wrapper, so the canary is appended by
    intercepting the exact allocation.  The kernel computes its stores from
    the comb16 base pointer, therefore any row written past the requested
    ``_comb16_rows(t)`` shows up as a modified NaN canary row.
    """
    real_empty = torch.empty
    rows = op._comb16_rows(t)
    canary_rows = op.NAIV
    captured: dict[str, torch.Tensor] = {}

    def tracked_empty(shape: tuple[int, ...], *a: object,
                      **kw: object) -> torch.Tensor:
        if (tuple(shape) == (rows, HC * HC)
                and kw.get("dtype") is torch.float32):
            padded = real_empty((rows + canary_rows, HC * HC), *a, **kw)
            padded[rows:].fill_(float("nan"))
            captured["padded"] = padded
            return padded[:rows]
        return real_empty(shape, *a, **kw)

    with mock.patch.object(torch, "empty", tracked_empty):
        outs = op.mhcpresinkhorn_rmsnorm(**args)
    torch_npu.npu.synchronize()
    return outs, captured.get("padded"), rows


@pytest.mark.parametrize("t", [1, 15, 16, 33, 513, 560, 1024, 1064, 2048])
def test_matches_reference(t: int) -> None:
    torch_npu.npu.set_device(_DEVICE_ID)
    args = _inputs(t)

    (y, rstd, post, comb), padded, rows = _call_with_comb_canary(args, t)
    if padded is not None:
        assert torch.isnan(padded[rows:]).all(), (
            f"comb16 overflowed its {rows}-row allocation at t={t}")

    ref_y, ref_rstd, ref_post, ref_comb = _reference(**args)
    assert (tuple(y.shape), y.dtype) == ((t, D), torch.bfloat16)
    assert (tuple(rstd.shape), rstd.dtype) == ((t, 1), torch.float32)
    assert (tuple(post.shape), post.dtype) == ((t, HC), torch.float32)
    assert (tuple(comb.shape), comb.dtype) == ((t, HC, HC), torch.float32)

    torch.testing.assert_close(y.cpu(), ref_y.cpu(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(rstd.cpu(), ref_rstd.cpu(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(post.cpu(), ref_post.cpu(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(comb.cpu(), ref_comb.cpu(), rtol=2e-3, atol=2e-3)


def test_sinkhorn_output_columns_are_normalized() -> None:
    torch_npu.npu.set_device(_DEVICE_ID)
    args = _inputs(8)

    _, _, _, comb = op.mhcpresinkhorn_rmsnorm(**args)
    torch_npu.npu.synchronize()

    comb = comb.cpu()
    assert torch.isfinite(comb).all()
    assert (comb > 0).all()
    # The final Sinkhorn step normalizes columns, so every column sum is
    # S / (S + eps) ~= 1.  Rows are not normalized afterwards, so they are only
    # approximately stochastic and are covered by the reference comparison.
    col_sums = comb.sum(dim=-2)
    torch.testing.assert_close(col_sums, torch.ones_like(col_sums),
                               rtol=1e-3, atol=1e-3)


def test_rmsnorm_output_matches_standalone_rmsnorm() -> None:
    torch_npu.npu.set_device(_DEVICE_ID)
    args = _inputs(4)

    y, rstd, _, _ = op.mhcpresinkhorn_rmsnorm(**args)
    torch_npu.npu.synchronize()

    # y is the normalized value, so mean(y^2) must be ~1 when gamma is all ones.
    ones = dict(args)
    ones["gamma"] = torch.ones(D, dtype=torch.bfloat16, device="npu")
    y_ones, rstd_ones, _, _ = op.mhcpresinkhorn_rmsnorm(**ones)
    torch_npu.npu.synchronize()

    mean_sq = y_ones.float().square().mean(dim=-1, keepdim=True)
    torch.testing.assert_close(mean_sq.cpu(), torch.ones_like(mean_sq).cpu(),
                               rtol=2e-2, atol=2e-2)
    # rstd must not depend on gamma.
    torch.testing.assert_close(rstd.cpu(), rstd_ones.cpu(), rtol=0, atol=0)
    assert torch.isfinite(y.float()).all()
