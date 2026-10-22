#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""CPU checkpoint replay to isolate FP8 activation semantics of layer0 Q."""

import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight.float() * expanded[:weight.shape[0], :weight.shape[1]]


def _activation_quant(x: torch.Tensor) -> torch.Tensor:
    groups = x.float().reshape(x.shape[0], -1, 128)
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0
    fp8 = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (fp8.float() * scale).reshape_as(x)


def _compare(x: torch.Tensor, y: torch.Tensor) -> dict:
    x, y = x.float(), y.float()
    delta = x - y
    return {"max_abs": delta.abs().max().item(),
            "rel_l2": (delta.norm() / y.norm().clamp_min(1e-30)).item()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xllm", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        parser.error("output must be a new checkout file")
    torch.set_num_threads(4)
    xstate = torch.jit.load(str(args.xllm / "layer_0_rank7.pt"), map_location="cpu").state_dict()
    xinput = torch.jit.load(str(args.xllm / "layer_0_input_rank7.pt"), map_location="cpu").state_dict()["value"]
    sstate = torch.load(args.sglang / "layer_0_prepare_rank0.pt", map_location="cpu", weights_only=True)
    start = int(xstate["metadata"][0])
    end = start + xinput.shape[0]
    sinput = sstate["input"][start:end]
    # Non-RoPE columns allow exact projection tests without importing either
    # implementation's position/rotation code.
    target_q = sstate["q"][start:end, :, :448].float()
    xllm_q = xstate["q"][:, :8, :448].float()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    weight_names = ["wq_a.weight", "wq_a.scale", "wq_b.weight", "wq_b.scale", "q_norm.weight"]
    tensors = {}
    for suffix in weight_names:
        name = "layers.0.attn." + suffix
        with safe_open(args.model / index[name], framework="pt", device="cpu") as checkpoint:
            tensors[suffix] = checkpoint.get_tensor(name)
    wa = _dequant(tensors["wq_a.weight"], tensors["wq_a.scale"])
    wb = _dequant(tensors["wq_b.weight"][:4096], tensors["wq_b.scale"][:32])
    gamma = tensors["q_norm.weight"].float()
    results = {"input_difference": _compare(xinput, sinput),
               "observed_q_difference": _compare(xllm_q, target_q), "variants": {}}
    for input_name, hidden in (("xllm", xinput), ("sglang", sinput)):
        for quantized in (False, True):
            active = _activation_quant(hidden) if quantized else hidden.float()
            w_a = wa if quantized else wa.bfloat16().float()
            q_down = (active @ w_a.T).bfloat16().float()
            qr = (q_down * torch.rsqrt(q_down.square().mean(-1, keepdim=True) + 1e-6) * gamma).bfloat16()
            active_qr = _activation_quant(qr) if quantized else qr.float()
            w_b = wb if quantized else wb.bfloat16().float()
            q = (active_qr @ w_b.T).bfloat16().float().reshape(-1, 8, 512)
            q = (q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)).bfloat16().float()
            candidate = q[..., :448]
            name = f"{input_name}_input_{'fp8_act' if quantized else 'bf16_act'}"
            results["variants"][name] = {"vs_sglang": _compare(candidate, target_q),
                                         "vs_xllm": _compare(candidate, xllm_q)}
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("Layer0 projection audit %s", results)


if __name__ == "__main__":
    main()
