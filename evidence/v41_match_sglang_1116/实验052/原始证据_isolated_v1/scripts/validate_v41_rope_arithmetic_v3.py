"""Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0.

Bounded comparison of explicit FP32 pair arithmetic; no model accuracy test.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_musa


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import v41_rope_table_test_module as native

    torch.manual_seed(111752)
    torch.set_num_threads(1)
    assert torch.musa.device_count() == 1
    rows_out = []
    totals = [0] * 6
    with torch.inference_mode():
        angle = torch.randn(4096, 32, device="musa", dtype=torch.float32)
        table = torch.cat((angle.cos(), angle.sin()), dim=1).contiguous()
        for heads in (1, 8, 64):
            for rows in (4, 6, 256):
                for dtype in (torch.int32, torch.int64):
                    positions = torch.randint(0, 4096, (rows,), device="musa", dtype=dtype)
                    x = torch.randn((rows, heads, 512), device="musa", dtype=torch.bfloat16)
                    for inverse in (False, True):
                        reference = native.reference(x, positions, table, inverse).cpu()
                        repeat = native.reference(x, positions, table, inverse).cpu()
                        assert torch.equal(reference.view(torch.int16), repeat.view(torch.int16))
                        modes = []
                        for mode in range(6):
                            actual = native.candidate(x, positions, table, inverse, 256, mode).cpu()
                            delta_bits = (actual.view(torch.int16).to(torch.int32) - reference.view(torch.int16).to(torch.int32)).abs()
                            mismatches = int((delta_bits != 0).sum())
                            totals[mode] += mismatches
                            modes.append({"mode": mode, "mismatches": mismatches,
                                          "max_bf16_ulp": int(delta_bits.max()),
                                          "max_abs": float((actual.float() - reference.float()).abs().max())})
                        rows_out.append({"heads": heads, "rows": rows, "dtype": str(dtype),
                                         "inverse": inverse, "modes": modes})
                        args.output.write_text(json.dumps({
                            "status": "CHECKING", "cases": rows_out,
                            "mismatch_totals": totals,
                            "exact_modes": [i for i, count in enumerate(totals) if count == 0],
                            "module_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                        }, indent=2))
    result = json.loads(args.output.read_text())
    result["status"] = "COMPARED_EXPLICIT_FP32_ARITHMETIC"
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
