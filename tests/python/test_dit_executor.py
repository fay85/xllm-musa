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

"""Contracts for the supported Python Joyo pipeline entry point."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from xllm.python.model_executor.dit_executor import DiTPipelineExecutor


@pytest.mark.parametrize("policy", ["FBCache", "TaylorSeer", "FBCacheTaylorSeer", "ResidualCache"])
def test_rejects_unsupported_cache_policy_before_component_loading(policy: str) -> None:
    with pytest.raises(ValueError, match="does not support dit_cache_policy"):
        DiTPipelineExecutor({"dit_cache_policy": policy})


def test_execute_calls_pipeline_with_request_only() -> None:
    request = SimpleNamespace(prompts=["a test video"], generation_params=SimpleNamespace(seed=0, guidance_scale=0.0))
    expected = SimpleNamespace(tensors=[], audio_tensors=[])
    seen = []

    def pipeline(value: object) -> object:
        seen.append(value)
        return expected

    executor = DiTPipelineExecutor.__new__(DiTPipelineExecutor)
    executor.pipeline = pipeline
    assert executor.execute(request) is expected
    assert seen == [request]


def test_component_precision_does_not_inherit_worker_vae_dtype() -> None:
    executor = DiTPipelineExecutor.__new__(DiTPipelineExecutor)
    executor.dtype = torch.float32
    assert executor._resolve_component_dtype({"class_name": "JoyOV2Transformer3DModel", "config": {}}) == torch.bfloat16
    for name in ("AutoencoderKLXVAE", "AutoencoderKLXDAC"):
        assert executor._resolve_component_dtype({"class_name": name, "config": {"dtype": "bfloat16"}}) == torch.float32
    assert (
        executor._resolve_component_dtype(
            {"class_name": "JoyoTextEncoder", "config": {"text_config": {"dtype": "bfloat16"}}}
        )
        == torch.bfloat16
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_xdac_weight_norm_matches_native_sqrt_then_guard(transpose: bool) -> None:
    from xllm.python.models.dit.autoencoders.autoencoder_kl_xdac import (
        WeightNormConv1d,
        WeightNormConvTranspose1d,
    )

    layer = WeightNormConvTranspose1d(2, 2, 3, 1) if transpose else WeightNormConv1d(2, 2, 3)
    with torch.no_grad():
        layer.weight_v.fill_(1e-5)
        layer.weight_g.fill_(0.5)
        layer.bias.zero_()
    value = torch.arange(10.0).reshape(1, 2, 5)
    norm = torch.linalg.vector_norm(layer.weight_v, dim=(1, 2), keepdim=True) + 1e-8
    weight = layer.weight_v * layer.weight_g / norm
    conv = torch.nn.functional.conv_transpose1d if transpose else torch.nn.functional.conv1d
    torch.testing.assert_close(layer(value), conv(value, weight, layer.bias), rtol=1e-6, atol=1e-6)
