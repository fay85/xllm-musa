# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight-rank native gather diagnostic; not an end-to-end model benchmark."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch.distributed as dist
import torch_musa

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
from v41_gather_probe_test_module import GatherProbe


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    assert size == 8
    torch.set_num_threads(1)
    torch.musa.set_device(rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    probe = GatherProbe(rank, size, args.native_port)
    result = {"rank": rank, "checks": [], "timings": [],
              "scope": "Isolated collective, no intervening model compute; not model critical-path share"}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"rank{rank}.json"

    def save() -> None:
        target.write_text(json.dumps(result, indent=2))

    with torch.inference_mode():
        for dtype in (torch.bfloat16, torch.float32):
            for rows in (1, 4, 6):
                for width in (640, 16160):
                    case = {"rows": rows, "local_width": width, "dtype": str(dtype)}
                    base = (torch.arange(rows * width, device=f"musa:{rank}", dtype=torch.float32)
                            .reshape(rows, width).remainder(64) / 64).to(dtype)
                    value = base + rank
                    expected = torch.cat([base + peer for peer in range(size)], dim=1)
                    for use_custom, label in ((False, "mccl"), (True, "padded_custom"), (False, "reverse_mccl")):
                        dist.barrier()
                        actual = probe.run(value, use_custom)
                        torch.musa.synchronize()
                        assert torch.equal(actual, expected), (case, label, "eager")
                        actual = probe.capture(value, use_custom, 8)
                        for update in (1, 2, 0):
                            value.copy_(base + rank + update)
                            actual = probe.replay()
                            torch.musa.synchronize()
                            assert torch.equal(actual, expected + update), (case, label, update)
                            result["checks"].append(dict(case, mode=label, update=update))
                        for _ in range(3):
                            probe.replay()
                        torch.musa.synchronize()
                        samples = []
                        for _ in range(3):
                            dist.barrier()
                            begin = torch.musa.Event(enable_timing=True)
                            end = torch.musa.Event(enable_timing=True)
                            begin.record()
                            for _ in range(10):
                                probe.replay()
                            end.record()
                            end.synchronize()
                            samples.append(begin.elapsed_time(end) * 1000 / (8 * 10))
                        result["timings"].append(dict(case, mode=label, samples_us=samples,
                                                      median_us=statistics.median(samples)))
                        probe.release()
                        save()
        result["status"] = "FINITE_VALUE_CHECKS_PASSED_DIAGNOSTIC_ONLY"
        save()
    dist.barrier()
    del probe
    dist.destroy_process_group()
    logger.info("Gather diagnostic finished on rank %d", rank)


if __name__ == "__main__":
    _main()
