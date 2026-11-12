# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate V4.1 fused rotary/FP4 and compressor chunk-state semantics."""

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


def _rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        (actual.float() - expected.float()).square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-10)
    ).item()


def _split_rope(
    native: Any,
    value: torch.Tensor,
    positions: torch.Tensor,
    table: torch.Tensor,
    group: int,
    e4m3: bool,
) -> torch.Tensor:
    frequency = table.index_select(0, positions.long())
    cosine, sine = frequency.chunk(2, dim=-1)
    rotated = native.musa_partial_rope_fp32(
        value.float(),
        cosine.repeat_interleave(2, -1).contiguous(),
        sine.repeat_interleave(2, -1).contiguous(),
        False,
    ).to(torch.bfloat16)
    return native.musa_v41_attention_fp4_quant(
        rotated.reshape(-1, value.shape[-1]).contiguous(), group, e4m3
    )[0].reshape(value.shape)


def _norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    fp32 = value.float()
    return (
        fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + 1e-6) * weight.float()
    ).to(torch.bfloat16)


def _compressor_cases(native: Any, device: torch.device) -> list[dict[str, Any]]:
    results = []
    # Odd chunk boundaries and lengths beyond the parked state ring test
    # the actual compressor class, not only its normalization primitive.
    chunks = (1, 16, 1, 31, 64, 1, 143)
    table = torch.tensor([[3, 1, 0, 2]], device=device, dtype=torch.int32)
    for dimension in (128, 512):
        for ratio in (1, 2):
            value = torch.randn(sum(chunks), 128, device=device, dtype=torch.bfloat16)
            bf16_weight = (
                torch.randn(dimension, 128, device=device, dtype=torch.bfloat16) / 16
            )
            weight = bf16_weight if ratio == 1 else bf16_weight.float()
            norm_weight = torch.randn(dimension, device=device, dtype=torch.bfloat16)
            gate = torch.randn_like(weight) / 8 if ratio == 2 else None
            state = (
                torch.full(
                    (4, 32, dimension),
                    float("nan"),
                    device=device,
                    dtype=torch.float32,
                )
                if ratio == 2
                else None
            )
            score_state = state.clone() if state is not None else None
            compressor = native.NativeV41Compressor(
                weight, norm_weight, gate, state, score_state, ratio, 32, 1e-6
            )
            projected = torch.nn.functional.linear(
                value if ratio == 1 else value.float(), weight
            )
            if ratio == 1:
                expected = _norm(projected, norm_weight)
            else:
                scores = torch.nn.functional.linear(value.float(), gate)
                pairs = projected[: value.shape[0] // 2 * 2].reshape(-1, 2, dimension)
                scores = scores[: pairs.shape[0] * 2].reshape(-1, 2, dimension)
                expected = _norm(
                    (pairs * scores.softmax(1)).sum(1).to(torch.bfloat16),
                    norm_weight,
                )
            outputs = []
            starts = []
            offset = 0
            for size in chunks:
                output, positions = compressor.forward(
                    value[offset : offset + size], offset, table
                )
                outputs.append(output)
                starts.append(positions)
                offset += size
            actual = torch.cat(outputs)
            actual_starts = torch.cat(starts)
            expected_starts = torch.arange(expected.shape[0], device=device) * ratio
            torch.testing.assert_close(actual_starts, expected_starts, atol=0, rtol=0)
            error = _rms(actual, expected)
            if error > 0.003 or not torch.isfinite(actual).all():
                raise AssertionError(f"Compressor D{dimension} R{ratio}: RMS {error}")
            # Ratio1 has no state mutation: compare the complete class call
            # with its unfused PyTorch projection/norm reference.
            timing = {}
            if ratio == 1:
                baseline = _time(
                    lambda: _norm(
                        torch.nn.functional.linear(value, weight), norm_weight
                    )
                )
                fused = _time(lambda: compressor.forward(value, 0, table))
                timing = {"reference_gpu_ms": baseline, "native_gpu_ms": fused}
            record = {
                "dimension": dimension,
                "ratio": ratio,
                "chunks": chunks,
                "rows": value.shape[0],
                "relative_rms": error,
                "positions_exact": True,
                **timing,
            }
            results.append(record)
            logger.info(
                "Compressor D%d ratio%d odd chunks PASS RMS %.5g",
                dimension,
                ratio,
                error,
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    import musa_python_kernel_test_module as native

    torch.manual_seed(111205)
    device = torch.device("musa:0")
    angles = torch.randn(8192, 32, device=device)
    table = torch.cat((angles.cos(), angles.sin()), -1).contiguous()
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for rows in (1, 17, 128, 256):
            positions = torch.arange(rows, device=device) * 7 + 113
            for heads, dimension, group, e4m3 in (
                (1, 128, 32, False),
                (32, 128, 32, False),
                (1, 512, 16, True),
            ):
                for amplitude in (0.001, 1.0, 64.0):
                    shape = (
                        (rows, dimension) if heads == 1 else (rows, heads, dimension)
                    )
                    value = (
                        torch.randn(shape, device=device, dtype=torch.bfloat16)
                        * amplitude
                    )
                    expected = _split_rope(native, value, positions, table, group, e4m3)
                    actual = native.musa_v41_rope_fp4_quant(
                        value, positions, table, group, e4m3
                    )
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    if amplitude == 1.0:
                        baseline = _time(
                            lambda: _split_rope(
                                native, value, positions, table, group, e4m3
                            )
                        )
                        fused = _time(
                            lambda: native.musa_v41_rope_fp4_quant(
                                value, positions, table, group, e4m3
                            )
                        )
                    else:
                        baseline, fused = None, None
                    results.append(
                        {
                            "rows": rows,
                            "heads": heads,
                            "dimension": dimension,
                            "group": group,
                            "e4m3_scale": e4m3,
                            "amplitude": amplitude,
                            "bitwise_equal": True,
                            "reference_gpu_ms": baseline,
                            "native_gpu_ms": fused,
                        }
                    )
                    args.output.write_text(
                        json.dumps({"status": "IN_PROGRESS", "rope": results}, indent=2)
                    )
                logger.info(
                    "Fused RoPE/FP4 M%d H%d D%d group%d exact PASS",
                    rows,
                    heads,
                    dimension,
                    group,
                )
        # E4M3 compressed-KV scales saturate at 448, including large input.
        large = torch.full((1, 512), 10000.0, device=device, dtype=torch.bfloat16)
        saturated = native.musa_v41_attention_fp4_quant(large, 16, True)
        if not torch.isfinite(saturated[0]).all() or not (saturated[2] == 126).all():
            raise AssertionError("Compressed-KV E4M3 scale must clamp to448")
        compressors = _compressor_cases(native, device)
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS_ROPE_AND_COMPRESSOR",
                "rope": results,
                "compressor": compressors,
                "compressed_scale_saturation": "PASS",
                "scope": "native operators and chunk-state equivalence; not full-model accuracy",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
