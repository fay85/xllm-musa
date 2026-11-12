# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native Dense AOT against real FP8/UE8M0 checkpoint projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa  # noqa: F401


def _tensor(model: Path, index: dict[str, str], name: str) -> torch.Tensor:
    with (model / index[name]).open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        record = header[name]
        start, end = record["data_offsets"]
        stream.seek(8 + length + start)
        payload = bytearray(stream.read(end - start))
    if len(payload) != end - start:
        raise EOFError(name)
    dtype = {
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E4M3FN": torch.float8_e4m3fn,
        "F8_E8M0": torch.uint8,
        "U8": torch.uint8,
    }[record["dtype"]]
    return torch.frombuffer(payload, dtype=dtype).reshape(record["shape"])


def _time(run: Callable[[], Any]) -> float:
    for _ in range(2):
        run()
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 10


def _qdq(value: torch.Tensor) -> torch.Tensor:
    grouped = value.float().reshape(value.shape[0], -1, 32)
    raw = grouped.abs().amax(-1).clamp_min(1e-10) / 448.0
    scale = ((raw.view(torch.int32) + 0x7FFFFF) & -0x800000).view(torch.float32)
    quantized = (grouped / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (quantized.float() * scale.unsqueeze(-1)).reshape(value.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 17, 128, 256])
    parser.add_argument("--specialized", action="store_true")
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    import musa_python_kernel_test_module as native

    torch.manual_seed(1112)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("musa:0")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    cases = [
        ("layers.0.attn.wq_a", None),
        ("layers.0.attn.wkv", None),
        ("layers.0.attn.wq_b", 0),
        ("layers.0.attn.wo_b", 1),
        ("layers.0.ffn.shared_experts.w1", None),
        ("layers.0.ffn.shared_experts.w3", None),
        ("layers.0.ffn.shared_experts.w2", None),
    ]
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for prefix, dimension in cases:
            packed = _tensor(args.model, index, prefix + ".weight")
            exponent = _tensor(args.model, index, prefix + ".scale").to(torch.int32)
            if exponent.eq(255).any():
                raise ValueError("UE8M0 NaN scale")
            scale = (
                torch.where(exponent.eq(0), 0x00400000, exponent << 23)
                .contiguous()
                .view(torch.float32)
            )
            if dimension is not None:
                packed = packed.chunk(8, dimension)[0].contiguous()
                scale = scale.chunk(8, dimension)[0].contiguous()
            weight = packed.to(device)
            scales = scale.to(device)
            decoded = weight.float() * scales.repeat_interleave(
                32, 0
            ).repeat_interleave(32, 1)
            carrier = decoded.to(torch.bfloat16)
            for rows in args.rows:
                directory = args.library_directory
                if args.specialized:
                    directory = (
                        directory / f"m{rows}_n{weight.shape[0]}_k{weight.shape[1]}"
                    )
                    expected_shape = [rows, weight.shape[0], weight.shape[1]]
                    if (
                        json.loads((directory / "manifest.json").read_text())[
                            "static_shape"
                        ]
                        != expected_shape
                    ):
                        raise ValueError("Static dense artifact shape mismatch")
                inputs = torch.randn(
                    rows, weight.shape[1], device=device, dtype=torch.bfloat16
                )
                reference = _qdq(inputs) @ decoded.T
                actual = native.v41_dense_block32(
                    inputs, weight, scales, str(directory)
                )
                difference = actual.float() - reference
                error = (
                    difference.square().mean().sqrt()
                    / reference.square().mean().sqrt().clamp_min(1e-12)
                ).item()
                if not torch.isfinite(actual).all() or error > 0.005:
                    raise AssertionError(
                        f"{prefix} M{rows}: relative RMS {error} > predeclared0.5%"
                    )
                native_ms = _time(
                    lambda: native.v41_dense_block32(
                        inputs, weight, scales, str(directory)
                    )
                )
                carrier_ms = _time(
                    lambda: torch.nn.functional.linear(
                        native.musa_v41_block32_qdq(inputs)[0], carrier
                    )
                )
                record = {
                    "projection": prefix,
                    "tp_rank": 0,
                    "tp": 8 if dimension is not None else 1,
                    "weight_shape": list(weight.shape),
                    "rows": rows,
                    "relative_rms": error,
                    "max_abs": difference.abs().max().item(),
                    "native_aot_gpu_ms": native_ms,
                    "reference_qdq_bf16_gpu_ms": carrier_ms,
                    "speedup": carrier_ms / native_ms,
                    "warmups": 2,
                    "repeats": 10,
                    "library_directory": str(directory),
                }
                results.append(record)
                args.output.write_text(
                    json.dumps({"status": "IN_PROGRESS", "results": results}, indent=2)
                )
                logger.info(
                    "Dense %s M%d RMS %.5g prepare %.4f -> %.4f ms",
                    prefix,
                    rows,
                    error,
                    carrier_ms,
                    native_ms,
                )
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS_REAL_WEIGHT_DENSE_BLOCK32_NATIVE_AOT",
                "results": results,
                "scope": "7 real layer0 projections, TP slices; not full-model or PD accuracy",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
