# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See http://www.apache.org/licenses/LICENSE-2.0 for the License.
"""Distributed FP8/GLM routing regression with independent FP32 reference."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401

from xllm.python.kernels_musa.mega_moe import MegaMoEWorkspace, MusaMegaMoEExperts


def _quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e, n, k = weight.shape
    blocks = weight.reshape(e, n // 128, 128, k // 128, 128)
    scales = blocks.abs().amax((2, 4)).clamp_min(1e-4) / 448
    q = (blocks / scales[:, :, None, :, None]).to(torch.float8_e4m3fn)
    dq = q.float() * scales[:, :, None, :, None]
    return q.reshape(weight.shape), scales, dq.reshape(weight.shape)


def _worker(rank: int, world: int, output: str, port: int) -> None:
    torch.set_num_threads(1)
    torch.musa.set_device(rank)
    device = torch.device("musa", rank)
    dist.init_process_group(
        "mccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world, timeout=timedelta(seconds=120)
    )
    torch.manual_seed(1105)
    e, h, i, k = 16, 256, 512, 8
    w13, s13, d13 = _quantize(torch.randn(e, 2 * i, h) * 0.05)
    w2, s2, d2 = _quantize(torch.randn(e, h, i) * 0.05)
    sl = slice(rank * (e // world), (rank + 1) * (e // world))
    workspace = MegaMoEWorkspace(dist.group.WORLD, e, h, i, k, 256)
    model = MusaMegaMoEExperts(
        w13[sl].to(device), s13[sl].to(device), w2[sl].to(device), s2[sl].to(device), workspace, 10.0
    )
    bias = torch.linspace(-0.2, 0.2, e)
    results = []
    with torch.inference_mode():
        # 129 crosses MATE's decode/split-prefill boundary; the final 1 detects stale state.
        counts = tuple(int(value) for value in os.getenv("MEGA_TEST_TOKENS", "1,16,65,129,1").split(","))
        for count in counts:
            torch.manual_seed(1500 + count)
            amplitude = float(os.getenv("MEGA_TEST_AMPLITUDE", "3"))
            x = (torch.randn(count, h) * amplitude).bfloat16()
            logits = torch.randn(count, e)
            score = logits.sigmoid()
            ids = (score + bias).topk(k, dim=-1).indices
            weights = score.gather(1, ids)
            weights *= 2.5 / weights.sum(-1, keepdim=True)
            groups = x.float().reshape(count, -1, 128)
            scale = groups.abs().amax(-1).clamp_min(1e-4) / 448
            dq = ((groups / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]).reshape(count, h)
            expected = torch.zeros(count, h)
            for expert in range(e):
                gate, up = (dq @ d13[expert].T).chunk(2, -1)
                activation = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
                expert_weight = (weights * (ids == expert)).sum(-1)
                expected += (activation @ d2[expert].T) * expert_weight[:, None]
            x_device = x.to(device)
            actual = model(x_device, ids.to(device), weights.to(device))
            # The shared branch is TP-partial; routed output is already complete.
            shared = x_device * ((rank + 1) / 100)
            dist.all_reduce(shared)
            actual += shared
            expected += x.float() * (world * (world + 1) / 200)
            torch.musa.synchronize()
            actual = actual.float().cpu()
            rmse = ((actual - expected).square().mean() / expected.square().mean()).sqrt().item()
            cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
            row = dict(
                rank=rank,
                world=world,
                tokens=count,
                amplitude=amplitude,
                relative_rmse=rmse,
                cosine=cosine,
                norm_ratio=(actual.norm() / expected.norm()).item(),
                finite=bool(torch.isfinite(actual).all()),
                workspace_bytes=workspace.buffer.num_bytes,
            )
            results.append(row)
            torch.save(
                {"actual": actual, "expected": expected, "ids": ids, "weights": weights},
                Path(output, f"rank{rank}_step{len(results)}.pt"),
            )
            Path(output, f"rank{rank}.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(row), flush=True)
            assert row["finite"] and rmse < 0.08 and cosine > 0.997, row
    workspace.close()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, choices=(1, 2, 8), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=45251)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=False)
    torch.multiprocessing.spawn(_worker, args=(args.world, args.output, args.port), nprocs=args.world, join=True)
    Path(args.output, "passed.json").write_text(json.dumps({"passed": True, "world": args.world}))


if __name__ == "__main__":
    main()
