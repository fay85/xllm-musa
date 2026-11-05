# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Check one real checkpoint expert against MATE; does not enable it in serving."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch
import torch_musa  # noqa: F401

from scripts.logger import logger
from xllm.python import initialize_runtime


def read_bytes_tensor(model: Path, index: dict[str, str], name: str) -> tuple[torch.Tensor, dict]:
    path = model / index[name]
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_size < 100 * 1024**2:
            raise ValueError("Unexpected safetensors header size")
        entry = json.loads(stream.read(header_size))[name]
        if entry["dtype"] not in ("I8", "U8", "F8_E8M0"):
            raise ValueError(f"Unexpected packed expert dtype: {entry}")
        start, end = entry["data_offsets"]
        stream.seek(8 + header_size + start)
        data = bytearray(stream.read(end - start))
    tensor = torch.frombuffer(data, dtype=torch.uint8).reshape(entry["shape"])
    return tensor, {"name": name, "file": index[name], "entry": entry, "payload_sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError("Expose only one idle GPU")
    initialize_runtime()
    from mate.gemm import masked_moe_gemm_mixed_dtype
    from xllm.python.models.deepseek_v41 import dequant_fp4_group

    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    torch.manual_seed(1105)
    rows = []
    with torch.inference_mode():
        for projection in ("w1", "w3", "w2"):
            prefix = f"layers.0.ffn.experts.0.{projection}"
            packed, weight_receipt = read_bytes_tensor(args.model, index, prefix + ".weight")
            scales, scale_receipt = read_bytes_tensor(args.model, index, prefix + ".scale")
            n, half_k = packed.shape
            k, count, capacity = half_k * 2, 17, 256
            minimum = scales.int().amin(dim=-1)
            residual = scales.int() - minimum.unsqueeze(-1) + 127
            if int(residual.max()) >= 255 or int(scales.max()) == 255:
                raise ValueError("Scale range requires separate handling; cannot wrap E8M0")
            epilogue = torch.ldexp(torch.ones(n, dtype=torch.float32), minimum - 127)
            activation = torch.zeros((1, capacity, k), dtype=torch.float32)
            activation[0, :count].normal_(0, 0.5)
            activation = activation.to(torch.float8_e4m3fn)
            reference = (activation[0, :count].float() @ dequant_fp4_group(packed, scales).float().T).to(torch.bfloat16)
            output = torch.full((1, capacity, n), float("nan"), device="musa:0", dtype=torch.bfloat16)
            masked_moe_gemm_mixed_dtype(
                (activation.to("musa:0"), torch.ones((1, capacity, 1), device="musa:0")),
                (packed.view(torch.int8).unsqueeze(0).to("musa:0"),
                 (residual.to(torch.uint8).unsqueeze(0).to("musa:0"), epilogue.unsqueeze(0).to("musa:0"))),
                torch.tensor([count], device="musa:0", dtype=torch.int32), output,
                expect_tokens=count, mixed_dtype="fp4fp8", backend="mubin",
                a_quant_recipe=(1, -1), b_quant_recipe=(1, 32),
            )
            torch.musa.synchronize()
            actual = output[0, :count].cpu()
            close = torch.isclose(actual.float(), reference.float(), rtol=0.02, atol=0.002)
            finite = torch.isfinite(actual)
            error = (actual.float() - reference.float()).abs()
            row = {"projection": projection, "shape_N_K": [n, k], "tokens": count,
                   "status": "PASS" if bool(close.all()) else "FAIL",
                   "finite_elements": int(finite.sum()), "total_elements": actual.numel(),
                   "close_elements": int(close.sum()),
                   "max_abs_error": float(error.max()) if bool(finite.all()) else None,
                   "scale_min": int(scales.min()), "scale_max": int(scales.max()),
                   "max_residual_exponent": int(residual.max()),
                   "weight": weight_receipt, "scale": scale_receipt}
            rows.append(row)
            result = {"status": "PASS" if len(rows) == 3 and all(r["status"] == "PASS" for r in rows) else "INCOMPLETE_OR_FAIL",
                      "scope": "one real layer-0 expert, three projections, synthetic quantized activations; no MoE integration",
                      "cases": rows}
            args.output.write_text(json.dumps(result, indent=2))
            logger.info("Checkpoint %s status=%s error=%s", projection, row["status"], row["max_abs_error"])
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
