# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Validate MATE FP4FP8 on one idle GPU; does not select a model backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_musa  # noqa: F401

from scripts.logger import logger
from xllm.python import initialize_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError("Expose exactly one idle MUSA GPU")
    initialize_runtime()
    from mate.gemm import masked_moe_gemm_mixed_dtype
    from xllm.python.models.deepseek_v41 import dequant_fp4_group

    device = torch.device("musa:0")
    torch.manual_seed(1105)
    rows = []
    with torch.inference_mode():
        for case in ("ones", "random_weights", "random_activation", "small_scales", "random_scales", "factored_scales", "factored_up_shape", "factored_down_shape"):
            experts, capacity, n, k = 2, 256, 256, 256
            if case == "factored_up_shape":
                experts, n, k = 1, 2304, 5120
            elif case == "factored_down_shape":
                experts, n, k = 1, 5120, 2304
            counts = [17] if experts == 1 else [3, 17]
            packed = torch.full((experts, n, k // 2), 0x22, dtype=torch.uint8)
            scales = torch.full((experts, n, k // 32), 127, dtype=torch.uint8)
            a = torch.ones((experts, capacity, k), dtype=torch.bfloat16)
            if case in ("random_weights", "random_scales") or case.startswith("factored_"):
                packed.random_(0, 256)
            if case in ("small_scales", "random_scales") or case.startswith("factored_"):
                scales.random_(120, 125)
            if case in ("random_activation", "random_scales") or case.startswith("factored_"):
                a.normal_(0, 0.5)
            # The reference uses the SAME quantized activations; this checks
            # packed layout and GEMM, not BF16-versus-FP8 activation error.
            a_fp8_cpu = a.float().to(torch.float8_e4m3fn)
            expected = torch.stack([
                a_fp8_cpu[e].float() @ dequant_fp4_group(packed[e], scales[e]).float().T
                for e in range(experts)
            ]).to(torch.bfloat16)
            residual_scales = scales
            epilogue = torch.ones((experts, n), dtype=torch.float32)
            if case.startswith("factored_"):
                # Exact power-of-two factorization: keep residual exponents
                # nonnegative; restore each row's common factor in epilogue.
                minimum = scales.to(torch.int32).amin(dim=-1)
                residual_scales = (scales.to(torch.int32) - minimum.unsqueeze(-1) + 127).to(torch.uint8)
                epilogue = torch.ldexp(epilogue, minimum - 127)
            output = torch.full((experts, capacity, n), float("nan"), dtype=torch.bfloat16, device=device)
            logger.info("Launching FP4FP8 case=%s E=%s M=%s N=%s K=%s", case, experts, capacity, n, k)
            masked_moe_gemm_mixed_dtype(
                (a_fp8_cpu.to(device), torch.ones((experts, capacity, 1), dtype=torch.float32, device=device)),
                (packed.view(torch.int8).to(device), (residual_scales.to(device), epilogue.to(device))),
                torch.tensor(counts, dtype=torch.int32, device=device),
                output,
                expect_tokens=max(counts),
                mixed_dtype="fp4fp8",
                backend="mubin",
                a_quant_recipe=(1, -1),
                b_quant_recipe=(1, 32),
            )
            torch.musa.synchronize()
            error = 0.0
            comparisons = []
            passed = True
            for expert, count in enumerate(counts):
                actual = output[expert, :count].cpu()
                reference = expected[expert, :count]
                finite = torch.isfinite(actual)
                difference = (actual.float() - reference.float()).abs()
                close = torch.isclose(actual.float(), reference.float(), rtol=0.02, atol=0.02)
                passed = passed and bool(close.all())
                finite_error = difference[finite].max().item() if finite.any() else None
                comparisons.append({"expert": expert, "finite": int(finite.sum()),
                                    "total": actual.numel(), "close": int(close.sum()),
                                    "max_finite_abs_error": finite_error,
                                    "actual_head": [float(v) if torch.isfinite(v) else str(float(v)) for v in actual.flatten()[:16]],
                                    "reference_head": reference.flatten()[:16].float().tolist()})
                if finite_error is not None:
                    error = max(error, finite_error)
            rows.append({"case": case, "status": "PASS" if passed else "FAIL", "max_finite_abs_error": error, "comparisons": comparisons})
            logger.info("FP4FP8 case %s passed=%s, max_finite_abs_error=%s", case, passed, error)
    result = {"status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL", "cases": rows, "scope": "MATE masked FP4FP8 primitive only; no full MoE or model integration"}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
