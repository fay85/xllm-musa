# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""Real checkpoint MoE layer: EP8 routed + TP8 shared against CPU FP32 math.

Synthetic activations deliberately isolate layer integration from preceding
attention/mHC. Passing this test does not qualify complete-model accuracy.
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401
from safetensors import safe_open

from xllm.python.kernels_musa.mega_moe import MegaMoEWorkspace, MusaMegaMoEExperts


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    return (weight.float().reshape(n // 128, 128, k // 128, 128) * scale[:, None, :, None]).reshape(n, k)


def _swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)


def _worker(rank: int, model_dir: str, layer: int, output: str, port: int) -> None:
    torch.set_num_threads(2)
    torch.musa.set_device(rank)
    device = torch.device("musa", rank)
    dist.init_process_group(
        "mccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=8, timeout=timedelta(seconds=300)
    )
    root = Path(model_dir)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp."

    def load(name: str) -> torch.Tensor:
        key = prefix + name
        with safe_open(root / index[key], framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def pair(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return load(name + ".weight"), load(name + ".weight_scale_inv")

    e, h, width, k = 288, 4096, 2048, 8
    local = e // 8
    gate_up = torch.empty((local, width * 2, h), dtype=torch.float8_e4m3fn)
    gate_scale = torch.empty((local, width * 2 // 128, h // 128), dtype=torch.float32)
    down = torch.empty((local, h, width), dtype=torch.float8_e4m3fn)
    down_scale = torch.empty((local, h // 128, width // 128), dtype=torch.float32)
    for j in range(local):
        name = f"experts.{rank * local + j}."
        for offset, projection in ((0, "gate_proj"), (width, "up_proj")):
            weight, scale = pair(name + projection)
            gate_up[j, offset : offset + width].copy_(weight)
            gate_scale[j, offset // 128 : (offset + width) // 128].copy_(scale)
        weight, scale = pair(name + "down_proj")
        down[j].copy_(weight)
        down_scale[j].copy_(scale)
    router = load("gate.weight").float()
    bias = load("gate.e_score_correction_bias").float()
    shared = [_dequant(*pair("shared_experts." + projection)) for projection in ("gate_proj", "up_proj", "down_proj")]
    split = slice(rank * (width // 8), (rank + 1) * (width // 8))
    shared_gpu = [
        shared[0][split].bfloat16().to(device),
        shared[1][split].bfloat16().to(device),
        shared[2][:, split].bfloat16().to(device),
    ]
    workspace = MegaMoEWorkspace(dist.group.WORLD, e, h, width, k, 256)
    experts = MusaMegaMoEExperts(gate_up, gate_scale, down, down_scale, workspace, 10.0)
    rows = []
    with torch.inference_mode():
        for step, (count, amplitude) in enumerate(((1, 1.0), (17, 3.0), (1, 1.0))):
            torch.manual_seed(1105 + count)
            x = (torch.randn(count, h) * amplitude).bfloat16()
            scores = (x.float() @ router.T).sigmoid()
            ids = (scores + bias).topk(k, dim=-1).indices
            weights = scores.gather(1, ids)
            weights *= 2.5 / weights.sum(-1, keepdim=True)
            groups = x.float().reshape(count, -1, 128)
            scale = groups.abs().amax(-1).clamp_min(1e-4) / 448
            dq = ((groups / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]).reshape(count, h)
            expected_routed = torch.zeros(count, h)
            for j in range(local):
                routed_weight = (weights * (ids == rank * local + j)).sum(-1)
                selected = routed_weight != 0
                if not selected.any():
                    continue
                w13 = _dequant(gate_up[j], gate_scale[j])
                w2 = _dequant(down[j], down_scale[j])
                gate, up = (dq[selected] @ w13.T).chunk(2, -1)
                expected_routed[selected] += (_swiglu(gate, up) @ w2.T) * routed_weight[selected, None]
            dist.all_reduce(expected_routed, group=workspace.ipc_group)
            x_gpu = x.to(device)
            actual_routed = experts(x_gpu, ids.to(device), weights.to(device))
            actual_shared = _swiglu(x_gpu @ shared_gpu[0].T, x_gpu @ shared_gpu[1].T) @ shared_gpu[2].T
            dist.all_reduce(actual_shared)
            expected_shared = _swiglu(x.float() @ shared[0].T, x.float() @ shared[1].T) @ shared[2].T
            torch.musa.synchronize()
            actual_parts = {
                "routed": actual_routed.float().cpu(),
                "shared": actual_shared.float().cpu(),
                "combined": (actual_routed + actual_shared).float().cpu(),
            }
            expected_parts = {
                "routed": expected_routed,
                "shared": expected_shared,
                "combined": expected_routed + expected_shared,
            }
            metrics = {}
            for name, actual in actual_parts.items():
                expected = expected_parts[name]
                rmse = ((actual - expected).square().mean() / expected.square().mean().clamp_min(1e-20)).sqrt().item()
                cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
                metrics[name] = {"relative_rmse": rmse, "cosine": cosine, "finite": bool(torch.isfinite(actual).all())}
            row = {
                "rank": rank,
                "layer": layer,
                "tokens": count,
                "amplitude": amplitude,
                "workspace_bytes": workspace.buffer.num_bytes,
                "metrics": metrics,
            }
            rows.append(row)
            Path(output, f"rank{rank}.json").write_text(json.dumps(rows, indent=2))
            torch.save(
                {"actual": actual_parts, "expected": expected_parts, "ids": ids, "weights": weights},
                Path(output, f"rank{rank}_step{step}.pt"),
            )
            print(json.dumps(row), flush=True)
            for metric in metrics.values():
                assert metric["finite"] and metric["relative_rmse"] < 0.08 and metric["cosine"] > 0.997, row
    workspace.close()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--port", type=int, default=45283)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=False)
    torch.multiprocessing.spawn(_worker, args=(args.model, args.layer, args.output, args.port), nprocs=8, join=True)
    Path(args.output, "passed.json").write_text(json.dumps({"passed": True, "layer": args.layer}))


if __name__ == "__main__":
    main()
