# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Finite EP8 packed-MoE numerical check on eight otherwise idle GPUs."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401

from scripts.logger import logger
from xllm.python import initialize_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 8 or not 0 <= args.rank < 8:
        raise RuntimeError("This check requires exactly eight idle GPUs")
    torch.set_num_threads(2)
    torch.musa.set_device(args.rank)
    initialize_runtime()
    from xllm.python.distributed.collectives import init_process_group
    from xllm.python.models.deepseek_v41 import DeepseekV41Config, DeepseekV41MoE

    device = torch.device("musa", args.rank)
    group = init_process_group("moe_ep", args.host, args.port, args.rank, 8,
                               str(device), args.rank, 8, 0)
    cfg = DeepseekV41Config(hidden_size=64, moe_intermediate_size=32,
                           n_routed_experts=16, n_activated_experts=6,
                           routed_scaling_factor=1.5, swiglu_limit=10.0)
    torch.manual_seed(1105)
    reference_model = DeepseekV41MoE(cfg, 0, torch.bfloat16, torch.device("cpu"), packed_fp4=True)
    local_cfg = dataclasses.replace(cfg, ep_size=8, ep_rank=args.rank)
    local_model = DeepseekV41MoE(local_cfg, 0, torch.bfloat16, device, packed_fp4=True)
    rows = []
    with torch.inference_mode():
        for name, value in reference_model.named_parameters():
            if name.startswith("experts_"):
                value.random_(0, 256)
                local_model.get_parameter(name).copy_(value[args.rank * 2:(args.rank + 1) * 2])
            else:
                value.copy_(torch.randn_like(value.float()).to(value.dtype) * 0.2)
                local_model.get_parameter(name).copy_(value)
        for name in ("w1", "w2", "w3"):
            scale = getattr(reference_model, f"experts_{name}_scale")
            scale.random_(121, 125)
            getattr(local_model, f"experts_{name}_scale").copy_(scale[args.rank * 2:(args.rank + 1) * 2])
        probe = torch.full((64,), args.rank + 1, device=device, dtype=torch.float32)
        dist.all_reduce(probe, group=group)
        torch.testing.assert_close(probe.cpu(), torch.full((64,), 36.0), rtol=0, atol=0)
        for count in (1, 17):
            hidden = (torch.randn(count, 64) * 4).to(torch.bfloat16)
            expected = reference_model(hidden)
            actual = local_model(hidden.to(device)).cpu()
            error = float((actual.float() - expected.float()).abs().max())
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            rows.append({"tokens": count, "status": "PASS", "max_abs_error": error})
            logger.info("rank=%s tokens=%s EP8 max_abs_error=%s", args.rank, count, error)
        dist.barrier(group=group)
        torch.musa.synchronize()
    args.output.write_text(json.dumps({"rank": args.rank, "status": "PASS", "cases": rows,
                                     "scope": "synthetic MoE, EP8 MCCL, same shared expert once; not full model or cross-host PD"}, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
