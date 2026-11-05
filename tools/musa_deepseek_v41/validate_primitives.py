# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Finite single-GPU numerical checks; not a whole-model accuracy claim."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch_musa  # noqa: F401

from xllm.python import initialize_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError("Expose exactly one idle GPU to this validation process")
    initialize_runtime()
    from xllm.python import kernels
    from xllm.python.models.deepseek_v41 import (
        DeepseekV41Config,
        DeepseekV41HyperConnection,
        DeepseekV41MoE,
        dequant_fp4_group,
    )

    torch.manual_seed(57)
    device = torch.device("musa:0")
    rows: list[dict] = []

    def check(name: str, actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> None:
        if actual.device.type != "musa":
            raise AssertionError(f"{name}: result did not execute on MUSA")
        actual_cpu = actual.detach().cpu()
        torch.testing.assert_close(actual_cpu, expected, rtol=rtol, atol=atol)
        rows.append({"case": name, "status": "PASS", "max_abs_error": (actual_cpu.float() - expected.float()).abs().max().item(), "rtol": rtol, "atol": atol})

    with torch.inference_mode():
        packed = torch.randint(0, 256, (128, 128), dtype=torch.uint8)
        scale = torch.randint(119, 127, (128, 8), dtype=torch.uint8)
        check("mxfp4_dequant", dequant_fp4_group(packed.to(device), scale.to(device)), dequant_fp4_group(packed, scale), 0, 0)
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(17, 256).to(dtype)
            w = torch.randn(256).to(dtype)
            ref = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * w.float()).to(dtype)
            check(f"rms_norm_{dtype}", kernels.rms_norm(x.to(device), w.to(device), 1e-6), ref, 0.01 if dtype == torch.bfloat16 else 1e-5, 0.01 if dtype == torch.bfloat16 else 1e-5)
        cfg = DeepseekV41Config(hidden_size=64, moe_intermediate_size=32, n_routed_experts=4, n_activated_experts=2, routed_scaling_factor=1.5, swiglu_limit=10.0, hc_mult=4, hc_sinkhorn_iters=20)
        cpu_moe = DeepseekV41MoE(cfg, 0, torch.bfloat16, torch.device("cpu"), packed_fp4=True)
        for parameter in cpu_moe.parameters():
            if parameter.dtype == torch.uint8:
                parameter.copy_(torch.randint(0, 256, parameter.shape, dtype=torch.uint8))
            else:
                parameter.copy_(torch.randn_like(parameter.float()).to(parameter.dtype) * 0.2)
        for name in ("w1", "w2", "w3"):
            scale = getattr(cpu_moe, f"experts_{name}_scale")
            scale.copy_(torch.randint(121, 125, scale.shape, dtype=torch.uint8))
        gpu_moe = copy.deepcopy(cpu_moe).to(device)
        for tokens in (1, 17):
            x = (torch.randn(tokens, 64) * 2).to(torch.bfloat16)
            check(f"packed_expert_{tokens}_tokens", gpu_moe(x.to(device)), cpu_moe(x), 0.03, 0.03)
        cpu_hc = DeepseekV41HyperConnection(cfg, torch.bfloat16, torch.device("cpu"))
        for parameter in cpu_hc.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
        gpu_hc = copy.deepcopy(cpu_hc).to(device)
        x = torch.randn(17, 4, 64).to(torch.bfloat16)
        expected = cpu_hc.hc_mixes(x, cpu_hc.hc_attn_fn, cpu_hc.hc_attn_scale, cpu_hc.hc_attn_base)
        actual = gpu_hc.hc_mixes(x.to(device), gpu_hc.hc_attn_fn, gpu_hc.hc_attn_scale, gpu_hc.hc_attn_base)
        for name, value, reference in zip(("pre", "post", "comb"), actual, expected):
            check("mega_mhc_" + name, value, reference, 1e-4, 1e-5)
        torch.musa.synchronize()
    result = {"status": "PASS", "device": torch.musa.get_device_name(0), "torch": torch.__version__, "cases": rows, "scope": "single GPU synthetic primitives; no EP/TP or full model acceptance"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
