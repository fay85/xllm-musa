# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small native GPU regression and launch-timing check for fused mHC reduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch_musa


def _time_ms(fn: object) -> float:
    for _ in range(3):
        fn()
    torch.musa.synchronize()
    start = time.perf_counter()
    for _ in range(30):
        fn()
    torch.musa.synchronize()
    return (time.perf_counter() - start) * 1000 / 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--library-directory", type=Path)
    args = parser.parse_args()
    import musa_python_kernel_test_module as native

    torch.set_num_threads(1)
    torch.musa.set_device(0)
    torch.manual_seed(11121112)
    results = []
    with torch.inference_mode():
        for splits in [16, 32]:
            for rows in [0, 1, 17, 128, 512]:
                for magnitude in [0.0, 1e-4, 1.0, 100.0]:
                    partial = torch.randn(splits, rows, 24, device="musa", dtype=torch.float32) * magnitude
                    squares = torch.rand(splits, rows, device="musa", dtype=torch.float32) * magnitude * magnitude
                    reference = partial.sum(0) * torch.rsqrt((squares.sum(0) / 20480).unsqueeze(-1) + 1e-6)
                    actual = native.v41_mhc_prenorm_reduce(partial, squares, 20480, 1e-6)
                    torch.testing.assert_close(actual, reference, rtol=3e-5, atol=1e-4)
                    assert bool(torch.isfinite(actual).all().item())
                    results.append({"splits": splits, "rows": rows, "magnitude": magnitude, "max_abs": float((actual-reference).abs().max().item()) if actual.numel() else 0.0})
        real = []
        if args.fixture is not None:
            assert args.library_directory is not None
            from safetensors.torch import load_file
            source = load_file(str(args.fixture), device="cpu")
            for part in ["attn", "ffn"]:
                fn = source[f"layers.0.hc_{part}_fn"].to(device="musa", dtype=torch.float32)
                for rows in [1, 17, 128]:
                    residual = torch.randn(rows, fn.size(1), device="musa", dtype=torch.bfloat16)
                    ref = lambda: native.v41_mhc_prenorm(residual, fn, str(args.library_directory), 1e-6)
                    fused = lambda: native.v41_mhc_prenorm_fused(residual, fn, str(args.library_directory), 1e-6)
                    a,b = fused(),ref()
                    torch.testing.assert_close(a,b,rtol=3e-5,atol=1e-4)
                    real.append({"part":part,"rows":rows,"max_abs":float((a-b).abs().max().item()),"reference_host_enqueue_plus_device_ms":_time_ms(ref),"fused_host_enqueue_plus_device_ms":_time_ms(fused)})
    result = {"status":"PASS_FUSED_MHC_REDUCTION_BASIC_CHECKS","synthetic_cases":results,"real_weight_cases":real,"scope":"Short directly-related operator checks and enqueue+device microtiming; not full-model accuracy or official serving performance.","accuracy_scope":"仅通过基础功能检查，未完成模型精度验收"}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({"status":result["status"],"synthetic_case_count":len(results),"real_case_count":len(real)}))


if __name__ == "__main__":
    main()
