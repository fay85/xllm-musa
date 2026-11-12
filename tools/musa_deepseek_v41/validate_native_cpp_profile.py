# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the exact serving TorchProfiler hook on native GPU operations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    import musa_python_kernel_test_module as native

    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["XLLM_CPP_STACK_DIR"] = str(args.output)
    torch.manual_seed(111221)
    device = torch.device("musa:0")
    with torch.inference_mode():
        value = torch.randn(17, 128, device=device, dtype=torch.bfloat16)
        weight = torch.randn(512, 128, device=device) / 16
        gate = torch.randn_like(weight) / 8
        norm = torch.randn(512, device=device, dtype=torch.bfloat16)
        states = [torch.zeros(4, 32, 512, device=device) for _ in range(2)]
        table = torch.tensor([[3, 1, 0, 2]], device=device, dtype=torch.int64)
        target = native.NativeV41Compressor(
            weight, norm, gate, *states, 2, 128, 1e-6, True
        )
        reference = native.NativeV41Compressor(
            weight,
            norm,
            gate,
            states[0].clone(),
            states[1].clone(),
            2,
            128,
            1e-6,
            False,
        )
        expected, _ = reference.forward(value, 159, table)
        torch.musa.synchronize()
        if not native.v41_profile_start():
            raise RuntimeError("Native serving profiler rejected start")
        actual, positions = target.forward(value, 159, table)
        torch.musa.synchronize()
        if not native.v41_profile_stop(str(args.output), 0):
            raise RuntimeError("Native serving profiler did not save both traces")
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.02, rtol=0.004)
    traces = list(args.output.glob("*.pt.trace.json"))
    sidecars = list(args.output.glob("native_cpp_stacks_*.jsonl"))
    if len(traces) != 1 or len(sidecars) != 1:
        raise AssertionError(
            "Expected one native Kineto trace and one real stack sidecar"
        )
    events = json.loads(traces[0].read_text())["traceEvents"]
    records = [json.loads(line) for line in sidecars[0].read_text().splitlines()]
    calls = [row for row in records if row["kind"] == "record"]
    stacks = {
        row["stack_id"]: row["frames"] for row in records if row["kind"] == "stack"
    }
    by_record_id = {
        row["args"]["Record function id"]: row
        for row in events
        if "Record function id" in row.get("args", {})
    }
    matches = [row for row in calls if row["record_id"] in by_record_id]
    if any(
        by_record_id[row["record_id"]]["name"] != row["name"]
        or by_record_id[row["record_id"]]["tid"] != row["os_thread"]
        for row in matches
    ):
        raise AssertionError("Native RF id must match both CPU operator and OS thread")
    semantic = [row for row in matches if row["name"] == "V41/Attention/compressor"]
    kernels = [row for row in events if "kernel" in row.get("cat", "").lower()]
    if not semantic or not any("pair_pool_kernel" in row["name"] for row in kernels):
        raise AssertionError(
            "Missing native semantic correlation or actual GPU pair-pool kernel"
        )
    decoder = shutil.which("addr2line")
    if decoder is None:
        raise ValueError("Need addr2line to prove model-module native symbols")
    frames = [
        frame
        for row in semantic
        for frame in stacks[row["stack_id"]]
        if "musa_python_kernel_test_module" in frame["module"]
    ]
    resolved = []
    for frame in frames:
        result = subprocess.run(
            [decoder, "-C", "-f", "-e", frame["module"], hex(frame["module_offset"])],
            capture_output=True,
            text=True,
            check=True,
        )
        resolved.append({**frame, "symbolization": result.stdout.strip()})
    if not any(
        "DeepseekV41Compressor::forward" in row["symbolization"] for row in resolved
    ):
        raise AssertionError(
            "Actual native forward frame not resolved; labels alone are insufficient"
        )
    (args.output / "resolved_native_frames.json").write_text(
        json.dumps(resolved, indent=2)
    )
    summary = {
        "status": "PASS_SERVING_CPP_PROFILER_GPU_AND_REAL_NATIVE_STACK",
        "trace": str(traces[0]),
        "sidecar": str(sidecars[0]),
        "matched_record_ids": len(matches),
        "semantic_ranges": len(semantic),
        "gpu_kernels": len(kernels),
        "native_module_frames": len(resolved),
        "module_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "scope": "exact serving start/stop hook on native C2 GPU forward, actual PCs resolved to C++ forward; full-model16rank trace still separate",
    }
    (args.output / "result.json").write_text(json.dumps(summary, indent=2))
    logger.info("Native C++/GPU profiler validation: %s", summary)


if __name__ == "__main__":
    main()
