# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native MUSA graph re-instantiation under an active profiler."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
import torch_musa
import v41_verify_collective_test_module as native


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    rows = []
    with torch.inference_mode():
        stream = torch.musa.Stream()
        stream.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(stream):
            x = torch.arange(1024, device="musa", dtype=torch.float32)
            graph = native.NativeProfileGraph()
            graph.capture(x)
            graph.replay()
            stream.synchronize()
            for phase in ("before", "reinstantiate", "after_second_profile"):
                # No Python MUSAGraph exists here, and no graph is reset while
                # still registered with a Python profiler lifecycle hook.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.PrivateUse1]) as prof:
                    if phase != "before":
                        assert native.NativeProfileGraph.reinstantiate_registered() == 1
                    for i in range(3):
                        x.fill_(i + 2)
                        with torch.profiler.record_function("E27/native_graph/" + phase):
                            actual = graph.replay()
                        stream.synchronize()
                        assert torch.equal(actual.cpu(), torch.full((1024,), float((i + 2) ** 2 + 1)))
                path = args.output / f"{phase}.pt.trace.json"
                prof.export_chrome_trace(str(path))
                events = json.loads(path.read_text())["traceEvents"]
                kernels = [e for e in events if e.get("ph") == "X" and "kernel" in e.get("cat", "").lower()]
                row = {"phase": phase, "kernel_count": len(kernels),
                       "kernel_names": dict(collections.Counter(e["name"] for e in kernels)),
                       "output_checks": 3, "output_exact": True}
                rows.append(row)
                (args.output / "partial.json").write_text(json.dumps(rows, indent=2))
            graph.release()
            assert native.NativeProfileGraph.reinstantiate_registered() == 0
            del graph
            stream.synchronize()
    (args.output / "result.json").write_text(json.dumps({"status": "PASS_OUTPUT_CHECKS",
                                                        "rows": rows,
                                                        "scope": "Native graph profiling visibility probe, not model performance."}, indent=2))


if __name__ == "__main__":
    main()
