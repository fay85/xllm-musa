# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for Python prefill GDN slot-0 reserve and fused MLP routing."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import torch

import xllm.python as python_pkg

_mock_kernels = MagicMock()
_mock_kernels.resolve_gdn_prefill_backend.return_value = "mate"
_mock_kernels.resolve_gdn_decode_backend.return_value = "mate"
python_pkg.kernels = _mock_kernels
python_pkg._runtime_kernels = _mock_kernels
sys.modules["xllm.python.kernels"] = _mock_kernels

from xllm.python.layers.gated_delta_net import (  # noqa: E402
    Qwen3_5GatedDeltaNet,
)
from xllm.python.layers.gated_mlp import GatedMLP  # noqa: E402
from xllm.python.layers.linear import ColumnParallelLinear  # noqa: E402


class _GdnConfig:
    hidden_size = 32
    rms_norm_eps = 1e-6
    linear_conv_kernel_dim = 4
    linear_key_head_dim = 8
    linear_value_head_dim = 16
    linear_num_key_heads = 2
    linear_num_value_heads = 2
    tp_size = 1


def test_gdn_tp1_merges_ba_projection() -> None:
    layer = Qwen3_5GatedDeltaNet(
        _GdnConfig(),
        layer_id=0,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert layer.in_proj_ba is not None
    assert layer.in_proj_b is None
    assert layer.in_proj_a is None
    assert int(layer.in_proj_ba.weight.size(0)) == 2 * layer.num_v_heads


def test_gdn_reserve_packs_z_and_ssm_write() -> None:
    layer = Qwen3_5GatedDeltaNet(
        _GdnConfig(),
        layer_id=0,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    layer.reserve_graph_workspace(4)
    assert layer._gdn_z is not None
    assert layer._gdn_z.dtype == torch.bfloat16
    assert tuple(layer._gdn_z.shape) == (4, 2, 16)
    assert layer._gdn_ssm_write is not None
    assert layer._gdn_ssm_write.dtype == torch.float32


def test_gdn_pack_z_from_split_view() -> None:
    layer = Qwen3_5GatedDeltaNet(
        _GdnConfig(),
        layer_id=0,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    layer.reserve_graph_workspace(4)
    qkvz = torch.zeros(
        3, layer.conv_dim + layer.value_dim, dtype=torch.bfloat16
    )
    _mixed_qkv, z_flat = qkvz.split([layer.conv_dim, layer.value_dim], dim=-1)
    assert not z_flat.is_contiguous()
    packed = layer._pack_gdn_z(z_flat)
    assert packed.is_contiguous()
    assert tuple(packed.shape) == (3, 2, 16)
    torch.testing.assert_close(
        packed.reshape(3, layer.value_dim), z_flat.contiguous()
    )


def test_gdn_refresh_matches_bound_ssm() -> None:
    layer = Qwen3_5GatedDeltaNet(
        _GdnConfig(),
        layer_id=0,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    layer.reserve_graph_workspace(4)
    ssm = torch.zeros(3, 2, 8, 16, dtype=torch.float32)
    layer.refresh_graph_ssm_workspace(ssm)
    assert layer._gdn_ssm_write is not None
    assert tuple(layer._gdn_ssm_write.shape) == (1, 2, 8, 16)
    assert layer._gdn_ssm_write.dtype == torch.float32


def test_gated_mlp_uses_fused_swiglu_when_fp8_down() -> None:
    _mock_kernels.reset_mock()
    mlp = GatedMLP(
        hidden_size=128,
        intermediate_size=256,
        tp_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    mlp.gate_up_proj._fp8_weight = torch.empty(512, 128)
    mlp.gate_up_proj._weight_scale_inv = torch.empty(4, 1)
    mlp.gate_up_proj._output_dtype = torch.bfloat16
    mlp.down_proj._fp8_weight = torch.empty(128, 256)
    mlp.down_proj._weight_scale_inv = torch.empty(1, 2)
    mlp.down_proj._output_dtype = torch.bfloat16
    act_q = torch.empty(3, 256, dtype=torch.float8_e4m3fn)
    act_s = torch.empty(3, 2, dtype=torch.float32)
    _mock_kernels.block_fp8_linear.return_value = torch.zeros(
        3, 512, dtype=torch.bfloat16
    )
    _mock_kernels.fused_swiglu_quant_fp8.return_value = (act_q, act_s)
    _mock_kernels.block_fp8_linear_quantized.return_value = torch.zeros(
        3, 128, dtype=torch.bfloat16
    )

    hidden = torch.zeros(3, 128, dtype=torch.bfloat16)
    output = mlp(hidden)

    assert output.shape == (3, 128)
    _mock_kernels.block_fp8_linear.assert_called_once()
    _mock_kernels.fused_swiglu_quant_fp8.assert_called_once()
    _mock_kernels.block_fp8_linear_quantized.assert_called_once()
    _mock_kernels.silu_and_mul.assert_not_called()
    _mock_kernels.per_token_group_quant_fp8.assert_not_called()


def test_gated_mlp_keeps_silu_without_fp8_down() -> None:
    _mock_kernels.reset_mock()
    mlp = GatedMLP(
        hidden_size=8,
        intermediate_size=256,
        tp_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    activated = torch.zeros(3, 256, dtype=torch.bfloat16)
    _mock_kernels.silu_and_mul.return_value = activated

    hidden = torch.zeros(3, 8, dtype=torch.bfloat16)
    output = mlp(hidden)

    assert output.shape == (3, 8)
    _mock_kernels.silu_and_mul.assert_called_once()
    _mock_kernels.fused_swiglu_quant_fp8.assert_not_called()


def test_lm_head_prefers_nn_gemm_layout() -> None:
    torch.manual_seed(0)
    layer = ColumnParallelLinear(
        8,
        16,
        tp_size=1,
        gather_output=True,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    layer.weight.data.normal_()
    hidden = torch.randn(1, 8, dtype=torch.float32)
    before = layer(hidden).clone()
    layer.prefer_nn_gemm_layout()
    assert tuple(layer.weight.shape) == (16, 8)
    assert layer.weight.t().is_contiguous()
    after = layer(hidden)
    torch.testing.assert_close(before, after)
    layer.prefer_nn_gemm_layout()
    torch.testing.assert_close(before, layer(hidden))
