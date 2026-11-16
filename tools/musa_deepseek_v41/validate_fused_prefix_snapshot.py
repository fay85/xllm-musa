# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare fused cache snapshots against the existing bounded transaction.

Exercise arbitrary later writes, padding/aliased physical pages, all/no/partial
acceptance and both streams. Model arithmetic is intentionally not evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch_musa
import v41_prefix_checkpoint_test_module as native

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.musa.device_count() == 1
    torch.set_num_threads(1)
    torch.manual_seed(111610)
    cases = []
    with torch.inference_mode():
        for nondefault in (False, True):
            stream = torch.musa.Stream() if nondefault else torch.musa.current_stream()
            with torch.musa.stream(stream):
                for dtype in (torch.bfloat16, torch.float32, torch.int32):
                    for mode in ("swa", "c1", "c2", "recurrent"):
                        block = 32 if mode == "recurrent" else 128
                        width = 1 if dtype == torch.int32 else 129
                        old_cpu = torch.randint(-20, 21, (7, block, width)).to(dtype)
                        changed_cpu = torch.randint(-40, 41, old_cpu.shape).to(dtype)
                        for begin in (0, 31, 32, 127, 255, 511):
                            for count in range(1, 7):
                                table_dtype = torch.int32 if begin % 2 else torch.int64
                                # Invalid negative/high pages and repeated page IDs:
                                # later saved values must never restore padding rows.
                                table = torch.tensor([[3, -1, 8, 3, 1, 5]],
                                                     device="musa:0", dtype=table_dtype)
                                for accepted in sorted({0, 1, count}):
                                    expected = old_cpu.to("musa:0")
                                    actual = expected.clone()
                                    legacy = native.PrefixCheckpoint(begin, count, 64 * 1024 * 1024, False)
                                    fused = native.PrefixCheckpoint(begin, count, 64 * 1024 * 1024, True)
                                    for checkpoint, cache in ((legacy, expected), (fused, actual)):
                                        if mode == "recurrent":
                                            checkpoint.add_recurrent(cache, table, 128)
                                        else:
                                            checkpoint.add_paged(cache, table, 2 if mode == "c2" else 1, mode == "swa")
                                    assert legacy.saved_bytes() == fused.saved_bytes()
                                    expected.copy_(changed_cpu)
                                    actual.copy_(changed_cpu)
                                    legacy.restore_suffix(accepted)
                                    fused.restore_suffix(accepted)
                                    assert torch.equal(actual.cpu(), expected.cpu()), (mode, begin, count, accepted, dtype)
                                    cases.append({"mode": mode, "begin": begin, "count": count,
                                                  "accepted": accepted, "dtype": str(dtype),
                                                  "table_dtype": str(table_dtype), "nondefault": nondefault,
                                                  "saved_bytes": fused.saved_bytes(), "exact": True})
                    logger.info("Fused snapshot cases passed: stream=%s dtype=%s", nondefault, dtype)
            stream.synchronize()
    args.output.write_text(json.dumps({"status": "PASS_EXACT_FUSED_SNAPSHOT_AND_SUFFIX_RESTORE",
                                      "cases": cases, "scope": "GPU bytewise cache transaction vs existing production transaction; not full model accuracy"}, indent=2))
    logger.info("All fused snapshot exact checks passed: %d", len(cases))


if __name__ == "__main__":
    main()
