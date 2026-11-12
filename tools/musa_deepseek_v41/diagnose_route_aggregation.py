# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the independent route oracle's scatter operator on CPU and MUSA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
from scripts.logger import logger


def _probe() -> dict:
    rows = torch.tensor([0, 3, 8, 16], dtype=torch.int64)
    result = {}
    for columns in (4, 5120):
        values = torch.arange(1, 4 * columns + 1, dtype=torch.float32).reshape(
            4, columns
        )
        expected = torch.zeros(17, columns).index_add_(0, rows, values)
        for dtype in (torch.int64, torch.int32):
            for layout in ("contiguous", "stride2"):
                index = rows.to(device="musa", dtype=dtype)
                if layout == "stride2":
                    index = torch.stack((index, torch.zeros_like(index)), -1)[:, 0]
                target = torch.zeros(17, columns, device="musa")
                target.index_add_(0, index, values.to("musa"))
                actual = target.cpu()
                result[f"columns={columns},{dtype},{layout}"] = {
                    "matches_cpu": torch.equal(expected, actual),
                    "index_stride": list(index.stride()),
                    "max_abs": (expected - actual).abs().max().item(),
                    "expected_first4": expected[:, :4].tolist(),
                    "actual_first4": actual[:, :4].tolist(),
                }
    return {"index_add": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose one verified idle GPU")
    record = {"before_native_import": _probe()}
    import musa_python_kernel_test_module  # noqa: F401

    record["after_native_import"] = _probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    logger.info("Saved CPU/MUSA route-aggregation diagnostic to %s", args.output)


if __name__ == "__main__":
    main()
