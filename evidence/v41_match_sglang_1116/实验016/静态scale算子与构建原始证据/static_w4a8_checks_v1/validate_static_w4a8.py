# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Short real-weight GPU checks for checkpoint-static W4A8 and Graph reuse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
from safetensors.torch import load_file
from scripts.logger import logger
from validate_w4a8_experts import _error, _weight


def _quant(value: torch.Tensor, scale: float) -> torch.Tensor:
    return (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)


def _stages(native: Any, inputs: torch.Tensor, ids: torch.Tensor,
            route_weights: torch.Tensor, packed: list[torch.Tensor],
            scales: list[torch.Tensor], library: str, scale: float
            ) -> tuple[torch.Tensor, ...]:
    q, qs, counts, rows = native.dispatch(inputs, ids, 0, 2, scale)
    up = native.gemm(q, qs, packed[0], scales[0], counts, library, True)
    down_q, down_s = native.swiglu(up, counts, 10.0, scale)
    down = native.gemm(down_q, down_s, packed[1], scales[1], counts, library, True)
    result = native.combine(down, ids, rows, route_weights, 0)
    return q, qs, counts, rows, up, down_q, down_s, down, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--gemm-so", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise ValueError("Expose exactly one verified idle GPU")
    import v41_static_w4a8_test_module as native
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    source = load_file(str(args.fixture), device="cpu")
    device = torch.device("musa:0")
    packed, scales, decoded = [], [], []
    for projection in ("w1", "w3", "w2"):
        weight = torch.stack([source[f"layers.0.ffn.experts.{i}.{projection}.weight"]
                              for i in (0, 1)]).to(device)
        scale = torch.stack([source[f"layers.0.ffn.experts.{i}.{projection}.weight_scale_inv"]
                             for i in (0, 1)]).to(device, dtype=torch.bfloat16)
        packed.append(weight)
        scales.append(scale)
        decoded.append(_weight(weight, scale))
    packed = [torch.cat(packed[:2], 1).contiguous(), packed[2]]
    scales = [torch.cat(scales[:2], 1).contiguous(), scales[2]]
    hidden = decoded[0].shape[-1]
    intermediate = decoded[0].shape[1]
    results = []
    with torch.inference_mode():
        for rows in (1, 4, 6, 17, 128, 256):
            for scale in (1.0, 0.125):
                torch.manual_seed(111600 + rows)
                x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
                ids = torch.stack((torch.arange(rows, device=device) % 4,
                                   (torch.arange(rows, device=device) + 1) % 4), 1).long()
                weights = torch.softmax(torch.randn(rows, 2, device=device), -1) * 1.5
                for alternate_stream in (False, True):
                    stream = torch.musa.Stream() if alternate_stream else torch.musa.current_stream()
                    stream.wait_stream(torch.musa.current_stream())
                    with torch.musa.stream(stream):
                        q, qs, counts, slots, up, qd, sd, down, y = _stages(
                            native, x, ids, weights, packed, scales, str(args.gemm_so), scale)
                        oracle = torch.zeros_like(x, dtype=torch.float32)
                        errors = []
                        for expert in (0, 1):
                            tokens, selected = torch.where(ids == expert)
                            tokens, selected = tokens.contiguous(), selected.contiguous()
                            assert counts[expert].item() == tokens.numel()
                            if not tokens.numel():
                                continue
                            local = slots[tokens, selected].long()
                            assert sorted(local.cpu().tolist()) == list(range(tokens.numel()))
                            xq = _quant(x[tokens], scale)
                            torch.testing.assert_close(q.float()[expert, local], xq.float(), atol=0, rtol=0)
                            torch.testing.assert_close(qs[expert, local], torch.full_like(qs[expert, local], scale), atol=0, rtol=0)
                            ref_up = (xq.float() * scale) @ torch.cat(decoded[:2], 1)[expert].T
                            err_up = _error(up[expert, local], ref_up)
                            assert err_up["relative_rms"] <= 0.01, err_up
                            gate, value = up[expert, local].float().split(intermediate, -1)
                            swiglu = torch.nn.functional.silu(gate.clamp(max=10)) * value.clamp(-10, 10)
                            ref_qd = _quant(swiglu, scale)
                            err_quant = _error(qd.float()[expert, local], ref_qd.float())
                            assert err_quant["relative_rms"] <= 0.002, err_quant
                            torch.testing.assert_close(sd[expert, local], torch.full_like(sd[expert, local], scale), atol=0, rtol=0)
                            ref_down = (qd.float()[expert, local] * scale) @ decoded[2][expert].T
                            err_down = _error(down[expert, local], ref_down)
                            assert err_down["relative_rms"] <= 0.01, err_down
                            oracle.index_add_(0, tokens, down[expert, local].float() * weights[tokens, selected, None])
                            errors.append({"up": err_up, "quant": err_quant, "down": err_down})
                        torch.testing.assert_close(y, oracle, atol=1e-5, rtol=1e-5)
                        # Reusing buffers with no local routes must return zero.
                        empty_ids = torch.full_like(ids, 3)
                        empty = _stages(native, x, empty_ids, weights, packed, scales, str(args.gemm_so), scale)[-1]
                        assert torch.count_nonzero(empty).item() == 0
                    stream.synchronize()
                    results.append({"rows": rows, "scale": scale,
                                    "alternate_stream": alternate_stream, "errors": errors})
        graph_results = []
        for rows in (1, 4, 6):
            x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
            ids = torch.zeros(rows, 2, device=device, dtype=torch.int64)
            ids[:, 1] = 1
            weights = torch.full((rows, 2), 0.75, device=device)
            model = native.Experts(packed[0], scales[0], packed[1], scales[1], 0, 10.0,
                                   str(args.gemm_so), 1.0, 1.0)
            stream = torch.musa.Stream()
            stream.wait_stream(torch.musa.current_stream())
            with torch.musa.stream(stream):
                for _ in range(2):
                    model.forward(x, ids, weights)
            stream.synchronize()
            graph = torch.musa.MUSAGraph()
            with torch.musa.graph(graph, stream=stream):
                result = model.forward(x, ids, weights)
            for iteration in range(3):
                x.copy_(torch.randn_like(x))
                ids.fill_(3 if iteration == 1 else 0)
                ids[:, 1] = 1 if iteration != 1 else 3
                graph.replay()
                ref = model.forward(x, ids, weights)
                torch.testing.assert_close(result, ref, atol=1e-5, rtol=1e-5)
            graph_results.append({"rows": rows, "replays": 3, "changed_input_and_route": True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS", "real_weight_cases": results,
                                      "graph_cases": graph_results,
                                      "scope": "Operator and Graph checks only; not full model accuracy"}, indent=2))
    logger.info("Static W4A8 passed %d real-weight cases and %d Graph buckets", len(results), len(graph_results))


if __name__ == "__main__":
    main()
