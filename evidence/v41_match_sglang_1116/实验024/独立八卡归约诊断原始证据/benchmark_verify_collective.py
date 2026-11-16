# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure production eight-rank graph reductions without model rank skew."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import torch_musa
import v41_verify_collective_test_module as native

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 8 and torch.musa.device_count() == 8
    torch.set_num_threads(1)
    torch.musa.set_device(rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    group = native.VerifyCollective(rank, world, args.native_port)
    device = torch.device("musa", rank)
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"rank{rank}.json"
    modes = [("bf16_1", [torch.bfloat16]),
             ("fp32_1", [torch.float32]),
             ("bf16_40", [torch.bfloat16] * 40),
             ("fp32_40", [torch.float32] * 40),
             ("mixed_80", [torch.bfloat16, torch.float32] * 40)]
    with torch.inference_mode():
        for use_custom in (True, False):
            for name, dtypes in modes:
                stream = torch.musa.Stream()
                stream.wait_stream(torch.musa.current_stream())
                with torch.musa.stream(stream):
                    # Distinct live addresses for every layer, exact binary
                    # fractions, and empty ranks avoid overflow and aliasing.
                    base = (torch.arange(4 * 5120).reshape(4, 5120) % 17 - 8 + rank).float() / 16
                    if rank in (2, 6):
                        base.zero_()
                    expected = base.double()
                    dist.all_reduce(expected)
                    inputs = [base.to(dtype=dtype, device=device) for dtype in dtypes]
                    dist.barrier()
                    outputs = group.capture_many(inputs, use_custom)
                    stream.synchronize()
                    checks = 0
                    for multiplier in (1, 2, -1, 0):
                        for tensor, dtype in zip(inputs, dtypes):
                            tensor.copy_((base * multiplier).to(dtype=dtype, device=device))
                        stream.synchronize()
                        dist.barrier()
                        group.replay_many()
                        stream.synchronize()
                        for actual in outputs:
                            assert torch.equal(actual.cpu().double(), expected * multiplier)
                            checks += 1
                    samples = []
                    # Final inputs are zero, but these dense collectives do not
                    # branch on values. Every address still executes each replay.
                    for _ in range(3):
                        group.replay_many()
                    stream.synchronize()
                    dist.barrier()
                    for _ in range(3):
                        start = torch.musa.Event(enable_timing=True)
                        end = torch.musa.Event(enable_timing=True)
                        dist.barrier()
                        wall = time.perf_counter()
                        start.record(stream)
                        for _ in range(10):
                            group.replay_many()
                        end.record(stream)
                        end.synchronize()
                        samples.append({"graph_ms": start.elapsed_time(end) / 10,
                                        "host_ms": (time.perf_counter() - wall) * 100,
                                        "replays": 10})
                    records.append({"case": name, "custom": use_custom,
                                    "reductions_per_graph": len(dtypes),
                                    "rows": 4, "width": 5120,
                                    "payload_bytes": sum(4 * 5120 * (2 if d == torch.bfloat16 else 4) for d in dtypes),
                                    "exact_checks": checks, "samples": samples,
                                    "mccl_includes_clone": not use_custom})
                    group.release_graph()
                    del outputs, inputs
                stream.synchronize()
                dist.barrier()
                output.write_text(json.dumps({"rank": rank, "records": records, "status": "RUNNING"}, indent=2))
    dist.barrier()
    del group
    dist.destroy_process_group()
    output.write_text(json.dumps({"rank": rank, "records": records, "status": "PASS",
                                  "scope": "Isolated contiguous q4 reductions; no layer compute or model rank skew. MCCL reference includes clone, custom does not. Not model critical-path share."}, indent=2))
    logger.info("Rank %d completed %d cases", rank, len(records))


if __name__ == "__main__":
    main()
