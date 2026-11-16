# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight-rank production TP/EP reductions for speculative verification rows."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch_musa
import v41_verify_collective_test_module as native

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def _check(actual: torch.Tensor, expected: torch.Tensor, exact: bool) -> dict:
    observed = actual.cpu().double()
    wanted = expected.double()
    assert torch.isfinite(observed).all()
    error = observed - wanted
    rms = float(error.square().mean().sqrt())
    relative = rms / max(float(wanted.square().mean().sqrt()), 1e-12)
    if exact:
        assert torch.equal(observed, wanted)
    else:
        limit = 0.007 if actual.dtype == torch.bfloat16 else 2e-6
        assert relative < limit, (relative, limit)
    return {"max_abs": float(error.abs().max()), "relative_rms": relative,
            "exact_expected": exact}


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
    records = []
    device = torch.device("musa", rank)
    with torch.inference_mode():
        # Capture requires a non-default stream. Both eager stream policies
        # are checked; graph replay is then checked on a fresh side stream.
        for nondefault in (False, True):
            stream = torch.musa.Stream() if nondefault else torch.musa.current_stream()
            stream.wait_stream(torch.musa.default_stream())
            with torch.musa.stream(stream):
                for dtype in (torch.bfloat16, torch.float32):
                    for rows in (1, 2, 3, 4, 5, 6):
                        for exact in (True, False):
                            generator = torch.Generator().manual_seed(111645 + rank)
                            cpu = ((torch.arange(rows * 5120).reshape(rows, 5120) % 17 - 8 + rank).float() / 16
                                   if exact else torch.randn(rows, 5120, generator=generator))
                            # Two ranks have no local routed-expert contribution.
                            if rank in (2, 6):
                                cpu.zero_()
                            cpu = cpu.to(dtype)
                            wanted = cpu.double()
                            dist.all_reduce(wanted)
                            data = cpu.to(device)
                            mccl = _check(group.mccl(data), wanted, exact)
                            custom = _check(group.custom(data), wanted, exact)
                            row = {"rows": rows, "dtype": str(dtype), "nondefault": nondefault,
                                   "exact_pattern": exact, "mccl": mccl, "custom": custom}
                            if nondefault and exact:
                                captured = group.capture(data)
                                row["capture"] = _check(captured, wanted, True)
                                replays = []
                                for multiplier in (2, -1, 0):
                                    data.copy_((cpu * multiplier).to(device))
                                    replays.append(_check(group.replay(), wanted * multiplier, True))
                                row["replays"] = replays
                                group.release_graph()
                            records.append(row)
                            dist.barrier()
            stream.synchronize()
    dist.barrier()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps({
        "status": "PASS_REAL_EIGHT_RANK_VERIFY_COLLECTIVE", "rank": rank,
        "records": records, "scope": "BF16 Attention TP and FP32 routed/shared EP payloads; not full-model accuracy",
    }, indent=2))
    del group
    dist.destroy_process_group()
    logger.info("Rank %d passed %d eager cases and changed-input graph replays", rank, len(records))


if __name__ == "__main__":
    main()
