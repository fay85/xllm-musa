# Copyright 2026 The xLLM Authors.
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

"""Packed expert storage must preserve the established BF16 expert result.

CPU is an independent numerical oracle here, not a serving fallback. The GPU
path additionally needs device, TP/EP and full-model integration validation.
"""

import pytest
import torch

from xllm.python.models.deepseek_v41 import DeepseekV41Config, DeepseekV41MoE


def _reference_weight(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    nibbles = torch.stack((packed.to(torch.int64) % 16, packed.to(torch.int64) // 16), dim=-1).flatten(-2)
    exponents = scale.to(torch.int32).repeat_interleave(32, dim=-1) - 127
    return torch.ldexp(values[nibbles], exponents).to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [1, 17])
def test_packed_experts_preserve_routing_clamp_and_shared_output(tokens: int) -> None:
    torch.manual_seed(57)
    cfg = DeepseekV41Config(
        hidden_size=64,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_activated_experts=2,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )
    packed = DeepseekV41MoE(cfg, 0, torch.bfloat16, torch.device("cpu"), packed_fp4=True)
    expanded = DeepseekV41MoE(cfg, 0, torch.bfloat16, torch.device("cpu"))
    with torch.no_grad():
        for name, parameter in expanded.named_parameters():
            if not name.startswith("experts_"):
                parameter.copy_(torch.randn_like(parameter.float()).to(parameter.dtype) * 0.2)
                packed.get_parameter(name).copy_(parameter)
        for name in ("w1", "w2", "w3"):
            weight = getattr(packed, f"experts_{name}")
            scale = getattr(packed, f"experts_{name}_scale")
            weight.copy_(torch.randint(0, 256, weight.shape, dtype=torch.uint8))
            scale.copy_(torch.randint(121, 125, scale.shape, dtype=torch.uint8))
            getattr(expanded, f"experts_{name}").copy_(_reference_weight(weight, scale))
        hidden = (torch.randn(tokens, 64) * 4).to(torch.bfloat16)
        expected = expanded(hidden)
        actual = packed(hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert packed.experts_w1.dtype == torch.uint8
    assert packed.experts_w1.numel() * 4 == expanded.experts_w1.numel() * expanded.experts_w1.element_size()


def test_packed_expert_geometry_fails_before_allocating_weights() -> None:
    cfg = DeepseekV41Config(hidden_size=65, moe_intermediate_size=32)
    with pytest.raises(ValueError, match="divisible by 32"):
        DeepseekV41MoE(cfg, 0, torch.bfloat16, torch.device("cpu"), packed_fp4=True)
