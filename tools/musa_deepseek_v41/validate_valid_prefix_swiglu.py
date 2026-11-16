# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare padded and valid-prefix SwiGLU on real MUSA streams and graphs."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def _assert_equal(torch: Any, old: Any, new: Any, counts: Any) -> None:
    for expert, count in enumerate(counts.cpu().tolist()):
        assert torch.equal(old[0][expert, :count].view(torch.uint8),
                           new[0][expert, :count].view(torch.uint8))
        assert torch.equal(old[1][expert, :count], new[1][expert, :count])


def _time_ms(torch: Any, module: Any, x: Any, counts: Any, optimized: bool) -> float:
    for _ in range(5):
        module.swiglu(x, counts, 10.0, 0.0, optimized)
    start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        module.swiglu(x, counts, 10.0, 0.0, optimized)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    import torch_musa
    import v41_static_w4a8_test_module as module

    torch.manual_seed(1701)
    records = []
    for capacity in (128, 256):
        for stream_name in ("default", "nondefault"):
            stream = torch.musa.default_stream() if stream_name == "default" else torch.musa.Stream()
            with torch.musa.stream(stream):
                x = (torch.randn(48, capacity, 4608, device="musa", dtype=torch.bfloat16) * 4).contiguous()
                for max_count in (0, 1, 4, 6, 17, capacity):
                    counts = torch.randint(0, max_count + 1, (48,), device="musa", dtype=torch.int32)
                    counts[0] = 0
                    counts[1] = max_count
                    for scale in (0.0, 0.125):
                        old = module.swiglu(x, counts, 10.0, scale, False)
                        new = module.swiglu(x, counts, 10.0, scale, True)
                        stream.synchronize()
                        _assert_equal(torch, old, new, counts)
                        records.append(dict(capacity=capacity, stream=stream_name, max_count=max_count,
                                            static_scale=scale, fp8_bytes_equal=True, scales_equal=True))
    graphs = []
    for max_count in (1, 4, 6):
        stream = torch.musa.Stream()
        with torch.musa.stream(stream):
            x = torch.randn(48, 128, 4608, device="musa", dtype=torch.bfloat16)
            counts = torch.ones(48, device="musa", dtype=torch.int32)
            for _ in range(3):
                module.swiglu(x, counts, 10.0, 0.0, True)
            graph = torch.musa.MUSAGraph()
            with torch.musa.graph(graph, stream=stream):
                new = module.swiglu(x, counts, 10.0, 0.0, True)
            for replay in range(3):
                x.normal_()
                counts.copy_(torch.randint(0, max_count + 1, (48,), device="musa", dtype=torch.int32))
                counts[0] = 0
                counts[1] = max_count
                graph.replay()
                old = module.swiglu(x, counts, 10.0, 0.0, False)
                stream.synchronize()
                _assert_equal(torch, old, new, counts)
                graphs.append(dict(max_count=max_count, replay=replay, passed=True))
    x = torch.randn(48, 128, 4608, device="musa", dtype=torch.bfloat16)
    counts = torch.randint(0, 5, (48,), device="musa", dtype=torch.int32)
    timings = [_time_ms(torch, module, x, counts, flag) for flag in (False, True, False)]
    torch.musa.synchronize()
    result = dict(status="PASS", cases=records, graphs=graphs,
                  standalone_event_ms_off_on_off=timings,
                  timing_scope="Standalone eager allocation+kernel stream span, not end-to-end performance")
    Path(args.output).write_text(json.dumps(result, indent=2))
    logger.info(f"Passed {len(records)} exact FP8/scale comparisons and {len(graphs)} changed-input graph replays")


if __name__ == "__main__":
    main()
