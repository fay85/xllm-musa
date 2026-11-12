# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check fused GPU Engram lookup against independent original-format CPU math."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa  # noqa: F401


def _time(function: Callable[[], Any]) -> float:
    for _ in range(2):
        function()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 10


def _cpu_reference(
    weight: torch.Tensor,
    scales: torch.Tensor,
    ids: torch.Tensor,
    row_lo: int,
    head_start: int,
) -> torch.Tensor:
    columns = ids[:, head_start : head_start + 3].long()
    owned = (columns >= row_lo) & (columns < row_lo + weight.shape[0])
    local = (columns - row_lo).clamp(0, weight.shape[0] - 1)
    # CPU casts use the original formats rather than reusing the ported
    # software decode. Keep NaN codes and UE8M0 subnormals in the oracle.
    values = weight.view(torch.float8_e4m3fn).float()[local]
    decoded_scales = scales.view(torch.float8_e8m0fnu).float()[local]
    decoded = (values * decoded_scales.repeat_interleave(32, -1)).bfloat16()
    return torch.where(owned[:, :, None], decoded, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    import musa_python_kernel_test_module as native

    torch.manual_seed(111211)
    device = torch.device("musa:0")
    row_lo = 1000003
    weight_cpu = torch.randint(0, 256, (257, 256), dtype=torch.uint8)
    weight = weight_cpu.to(device)
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for edge_scales in (False, True):
            codes = torch.tensor(
                [0, 1, 110, 120, 126, 127, 130, 254, 255], dtype=torch.uint8
            )
            scales_cpu = (
                codes[torch.arange(257 * 8) % codes.numel()].reshape(257, 8)
                if edge_scales
                else torch.randint(110, 128, (257, 8), dtype=torch.uint8)
            )
            scales = scales_cpu.to(device)
            for rows in (1, 17, 128, 256):
                for head_start in (0, 9, 21):
                    reference = native.NativeV41EngramEmbedding(
                        weight, scales, row_lo, head_start, 3
                    )
                    target = native.NativeV41EngramEmbedding(
                        weight,
                        scales,
                        row_lo,
                        head_start,
                        3,
                        str(args.library_directory),
                    )
                    for dtype in (torch.int32, torch.int64):
                        ids_cpu = torch.randint(
                            row_lo - 17, row_lo + 257 + 17, (rows, 24), dtype=dtype
                        )
                        ids_cpu[:, head_start] = row_lo - 1
                        ids_cpu[:, head_start + 1] = row_lo
                        ids_cpu[-1, head_start + 2] = row_lo + 257
                        ids = ids_cpu.to(device)
                        expected = _cpu_reference(
                            weight_cpu, scales_cpu, ids_cpu, row_lo, head_start
                        )
                        cpp = reference.local(ids)
                        actual = target.local(ids)
                        try:
                            torch.testing.assert_close(
                                cpp.cpu(), expected, atol=0, rtol=0, equal_nan=True
                            )
                            torch.testing.assert_close(
                                actual.cpu(), expected, atol=0, rtol=0, equal_nan=True
                            )
                        except AssertionError:
                            args.output.write_text(
                                json.dumps(
                                    {
                                        "status": "FAIL",
                                        "rows": rows,
                                        "head_start": head_start,
                                        "dtype": str(dtype),
                                        "edge_scales": edge_scales,
                                        "cpp_first": cpp[0, 1, :32]
                                        .float()
                                        .cpu()
                                        .tolist(),
                                        "actual_first": actual[0, 1, :32]
                                        .float()
                                        .cpu()
                                        .tolist(),
                                        "expected_first": expected[0, 1, :32]
                                        .float()
                                        .tolist(),
                                    },
                                    indent=2,
                                )
                            )
                            raise
                        timings = {}
                        if not edge_scales and head_start == 0 and dtype == torch.int64:
                            baseline = _time(lambda: reference.local(ids))
                            fused = _time(lambda: target.local(ids))
                            timings = {
                                "reference_gpu_ms": baseline,
                                "native_gpu_ms": fused,
                                "speedup": baseline / fused,
                                "warmups": 2,
                                "repeats": 10,
                            }
                        results.append(
                            {
                                "rows": rows,
                                "head_start": head_start,
                                "local_heads": 3,
                                "dtype": str(dtype),
                                "edge_scales": edge_scales,
                                "exact_cpu_reference_including_nan": True,
                                **timings,
                            }
                        )
                    logger.info(
                        "Engram gather M%d head%d edge_scales=%s exact PASS",
                        rows,
                        head_start,
                        edge_scales,
                    )
                    args.output.write_text(
                        json.dumps(
                            {"status": "IN_PROGRESS", "results": results}, indent=2
                        )
                    )
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS_NATIVE_FUSED_ENGRAM_GATHER",
                "results": results,
                "scope": "GPU-sharded local lookup, CPU original-format oracle; not full table bandwidth or full-model accuracy",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
