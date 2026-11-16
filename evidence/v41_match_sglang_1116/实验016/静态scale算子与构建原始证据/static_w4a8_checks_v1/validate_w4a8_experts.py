# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate native W4A8 stages against an independent real-weight GPU oracle.

This checks two exact checkpoint experts, including empty local routes and
decode/prefill row counts. It does not establish full-model accuracy, EP
collective correctness, or serving throughput. Oracle operations use GPU
FP32 arithmetic; CPU is used only for test metadata and reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
from safetensors.torch import load_file
from scripts.logger import logger


def _quantize(value: torch.Tensor, floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    groups = value.float().reshape(*value.shape[:-1], -1, 128)
    scale = groups.abs().amax(-1).clamp_min(floor) / 448.0
    quantized = (groups / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.reshape(value.shape), scale


def _dequantize(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (
        value.float().reshape(*value.shape[:-1], -1, 128) * scale.unsqueeze(-1)
    ).reshape(value.shape)


def _weight(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    bits = value.to(torch.uint8)
    nibbles = torch.stack((bits & 15, bits >> 4), -1).flatten(-2).to(torch.int32)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles).float()
    return signed * scale.to(torch.bfloat16).float().repeat_interleave(128, -1)


def _error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - reference.float()
    rms = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return {
        "max_abs": delta.abs().max().item(),
        "relative_rms": (delta.square().mean().sqrt() / rms).item(),
    }


def _time_ms(run: Callable[[], Any], repeats: int = 10) -> float:
    for _ in range(2):
        run()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--gemm-so", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-sync", action="store_true")
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    import musa_python_kernel_test_module as native

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("musa:0")
    source = load_file(str(args.fixture), device="cpu")
    packed, scales, decoded = [], [], []
    for projection in ("w1", "w3", "w2"):
        weights = torch.stack(
            [
                source[f"layers.0.ffn.experts.{expert}.{projection}.weight"]
                for expert in (0, 1)
            ]
        ).to(device)
        scale = torch.stack(
            [
                source[f"layers.0.ffn.experts.{expert}.{projection}.weight_scale_inv"]
                for expert in (0, 1)
            ]
        ).to(device=device, dtype=torch.bfloat16)
        packed.append(weights)
        scales.append(scale)
        decoded.append(_weight(weights, scale))
    gate_up = torch.cat(packed[:2], 1).contiguous()
    gate_up_scale = torch.cat(scales[:2], 1).contiguous()
    loaded = native.v41_load_w4a8_fixture(
        str(args.fixture), torch.empty(0, device=device, dtype=torch.bfloat16)
    )
    for actual, expected in zip(loaded, (gate_up, gate_up_scale, packed[2], scales[2])):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    implementation = native.NativeV41W4A8Experts(*loaded, 0, 10.0, str(args.gemm_so))
    hidden = decoded[0].size(2)
    intermediate = decoded[0].size(1)
    results = []
    with torch.inference_mode():
        for rows in (1, 17, 128, 256, 512):
            torch.manual_seed(111200 + rows)
            inputs = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
            ids = torch.stack(
                (
                    torch.arange(rows, device=device) % 4,
                    (torch.arange(rows, device=device) + 1) % 4,
                ),
                1,
            ).to(torch.int64)
            route_weights = torch.softmax(torch.randn(rows, 2, device=device), -1) * 1.5
            if args.diagnostic_sync:
                torch.musa.synchronize()
            q, input_scale, counts, route_rows = native.v41_w4a8_dispatch(
                inputs, ids, 0, 2
            )
            if args.diagnostic_sync:
                torch.musa.synchronize()
            result_up = native.v41_w4a8_gemm(
                q, input_scale, gate_up, gate_up_scale, counts, str(args.gemm_so)
            )
            if args.diagnostic_sync:
                torch.musa.synchronize()
            q_down, scale_down = native.v41_w4a8_swiglu_quant(result_up, counts, 10.0)
            if args.diagnostic_sync:
                torch.musa.synchronize()
            result_down = native.v41_w4a8_gemm(
                q_down, scale_down, packed[2], scales[2], counts, str(args.gemm_so)
            )
            if args.diagnostic_sync:
                torch.musa.synchronize()
            result = native.v41_w4a8_combine(
                result_down, ids, route_rows, route_weights, 0
            )
            oracle = torch.zeros_like(inputs, dtype=torch.float32)
            combine_oracle = torch.zeros_like(oracle)
            stage_errors = []
            for expert in range(2):
                token_rows, choices = torch.where(ids == expert)
                # This TorchMUSA build's index_add_ reads stride-2 indices as
                # contiguous. The separate CPU/MUSA primitive diagnostic
                # reproduces it before importing our native module. Keep the
                # independent oracle correct; do not weaken kernel tolerances.
                token_rows = token_rows.contiguous()
                choices = choices.contiguous()
                count = token_rows.numel()
                if counts[expert].item() != count:
                    raise AssertionError(
                        "Dispatch count differs from independent route selection"
                    )
                if count == 0:
                    continue
                slots = route_rows[token_rows, choices].long()
                combine_oracle.index_add_(
                    0,
                    token_rows,
                    result_down[expert, slots].float()
                    * route_weights[token_rows, choices, None],
                )
                if sorted(slots.cpu().tolist()) != list(range(count)):
                    raise AssertionError(
                        "Dispatch inverse map is not a valid bijection"
                    )
                xq, xs = _quantize(inputs[token_rows], 1e-4)
                torch.testing.assert_close(
                    q[expert, slots].float(), xq.float(), atol=0, rtol=0
                )
                torch.testing.assert_close(
                    input_scale[expert, slots], xs, atol=0, rtol=0
                )
                x = _dequantize(xq, xs)
                expected_up = torch.cat(
                    (x @ decoded[0][expert].T, x @ decoded[1][expert].T), -1
                ).to(torch.bfloat16)
                got_up = result_up[expert, slots]
                up_error = _error(got_up, expected_up)
                # Validate fused activation quantization separately using the
                # actual GEMM output, so a GEMM error cannot mask a glue bug.
                gate, up = got_up.float().chunk(2, -1)
                activation = torch.nn.functional.silu(gate.clamp_max(10)) * up.clamp(
                    -10, 10
                )
                aq, a_scale = _quantize(activation, 1e-10)
                torch.testing.assert_close(
                    scale_down[expert, slots], a_scale, atol=1e-8, rtol=2e-6
                )
                activation_error = _error(
                    _dequantize(q_down[expert, slots], scale_down[expert, slots]),
                    _dequantize(aq, a_scale),
                )
                expected_down = (
                    _dequantize(q_down[expert, slots], scale_down[expert, slots])
                    @ decoded[2][expert].T
                ).to(torch.bfloat16)
                down_error = _error(result_down[expert, slots], expected_down)
                # End-to-end oracle has independent GEMMs and quantization.
                reference_gate, reference_up = expected_up.float().chunk(2, -1)
                reference_activation = torch.nn.functional.silu(
                    reference_gate.clamp_max(10)
                ) * reference_up.clamp(-10, 10)
                ref_q, ref_scale = _quantize(reference_activation, 1e-10)
                reference_down = (
                    _dequantize(ref_q, ref_scale) @ decoded[2][expert].T
                ).to(torch.bfloat16)
                oracle.index_add_(
                    0,
                    token_rows,
                    reference_down.float() * route_weights[token_rows, choices, None],
                )
                stage_errors.append(
                    {
                        "expert": expert,
                        "gate_up": up_error,
                        "swiglu_quant": activation_error,
                        "down": down_error,
                    }
                )
                if max(up_error["relative_rms"], down_error["relative_rms"]) > 0.01:
                    raise AssertionError(
                        f"MATE GEMM error exceeds 1% relative RMS: {stage_errors[-1]}"
                    )
                if activation_error["relative_rms"] > 0.002:
                    raise AssertionError(
                        "SwiGLU/quantization error exceeds 0.2% relative RMS"
                    )
            full_error = _error(result, oracle)
            combine_error = _error(result, combine_oracle)
            if full_error["relative_rms"] > 0.02:
                failure = {
                    "status": "FAIL_NUMERICAL_REGRESSION",
                    "rows": rows,
                    "diagnostic_sync": args.diagnostic_sync,
                    "stage_errors": stage_errors,
                    "combine_error": combine_error,
                    "full_error": full_error,
                    "ids": ids.cpu().tolist(),
                    "route_rows": route_rows.cpu().tolist(),
                    "counts": counts.cpu().tolist(),
                    "result_first8": result[:, :8].cpu().tolist(),
                    "combine_oracle_first8": combine_oracle[:, :8].cpu().tolist(),
                    "full_oracle_first8": oracle[:, :8].cpu().tolist(),
                    "earlier_results": results,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(failure, indent=2) + "\n")
                raise AssertionError(
                    f"Routed output error exceeds 2% relative RMS: {full_error}"
                )
            direct = implementation.forward(inputs, ids, route_weights)
            torch.testing.assert_close(direct, result, atol=0, rtol=0)
            timing = _time_ms(
                lambda: implementation.forward(inputs, ids, route_weights)
            )
            row = {
                "diagnostic_sync": args.diagnostic_sync,
                "combine_error": combine_error,
                "rows": rows,
                "hidden": hidden,
                "intermediate": intermediate,
                "local_experts": 2,
                "topk": 2,
                "stage_errors": stage_errors,
                "routed_error": full_error,
                "native_gpu_event_ms": timing,
                "timing_warmups": 2,
                "timing_repeats": 10,
            }
            results.append(row)
            logger.info(
                "W4A8 rows=%d relative_rms=%.6g GPU_ms=%.4f",
                rows,
                full_error["relative_rms"],
                timing,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({"status": "IN_PROGRESS", "results": results}, indent=2)
                + "\n"
            )
    record = {
        "status": "PASS_TWO_EXPERT_GPU_REGRESSION",
        "native_checkpoint_loader_matches_independent_loader": True,
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "gemm_sha256": hashlib.sha256(args.gemm_so.read_bytes()).hexdigest(),
        "native_module": native.__file__,
        "native_module_sha256": hashlib.sha256(
            Path(native.__file__).read_bytes()
        ).hexdigest(),
        "epoch": time.time(),
        "results": results,
        "scope": "two real experts and local permutation/quantization/GEMM/combine; not full-model, EP collective, PD or serving validation",
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
