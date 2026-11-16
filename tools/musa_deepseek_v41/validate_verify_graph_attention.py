# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape verification Graph against actual native ratio2 Attention.

This checks cache transactions and causal prefix outputs, not model accuracy.
The reference executes only the accepted input prefix from the same old state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch_musa
import v41_verify_graph_inputs_test_module as native

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger

CACHE_NAMES = ("swa", "key", "index", "compressor_kv_state", "compressor_score_state")


def _random(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * 0.005).to(device="musa:0", dtype=torch.bfloat16)


def _clone(cache: object) -> object:
    result = native.NativeV41AttentionCache()
    for name in CACHE_NAMES:
        value = getattr(cache, name)
        setattr(result, name, value.clone() if value is not None else None)
    return result


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, b = actual.cpu().float(), expected.cpu().float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    delta = a - b
    return {"exact": torch.equal(a, b), "relative_rms": float(delta.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)), "max_absolute": float(delta.abs().max())}


def _checkpoint(cache: object, swa: torch.Tensor, table: torch.Tensor, begin: int) -> object:
    result = native.PrefixCheckpoint(begin, 6, 64 * 1024 * 1024)
    result.add_paged(cache.swa, swa, 1, True)
    result.add_paged(cache.key, table, 2, False)
    result.add_paged(cache.index, table, 2, False)
    result.add_recurrent(cache.compressor_kv_state, swa, 128)
    result.add_recurrent(cache.compressor_score_state, swa, 128)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sparse-so", required=True)
    parser.add_argument("--indexer-dir", required=True)
    parser.add_argument("--ratio", type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args()
    assert torch.musa.device_count() == 1
    torch.set_num_threads(1)
    torch.manual_seed(111635)
    device, dtype = "musa:0", torch.bfloat16
    cache_names = CACHE_NAMES if args.ratio == 2 else CACHE_NAMES[:3]
    with torch.inference_mode():
        weights = native.NativeV41AttentionWeights()
        weights.projection = [_random((1280, 5120)), torch.ones(1280, device=device, dtype=dtype), _random((4096, 1280)), _random((512, 5120)), torch.ones(512, device=device, dtype=dtype), _random((1024, 4096)), _random((5120, 1024))]
        weights.sink = torch.zeros(8, device=device)
        weights.compressor_norm = torch.ones(512, device=device, dtype=dtype)
        weights.compressor_kv = _random((512, 5120)).float()
        weights.compressor_gate = _random((512, 5120)).float()
        weights.indexer = [_random((4096, 1280)), _random((32, 5120)), _random((128, 512)), torch.ones(128, device=device, dtype=dtype)]
        config = native.NativeV41AttentionConfig()
        config.mode = (native.NativeV41AttentionMode.SWA if args.ratio == 0
                       else native.NativeV41AttentionMode.FULL)
        config.layer = config.source_layer = 0
        if args.ratio == 0:
            config.source_layer = -1
            weights.indexer = []
        config.ratio, config.heads, config.groups = args.ratio, 8, 1
        if args.ratio == 1:
            weights.compressor_kv = weights.compressor_kv.to(dtype)
            weights.compressor_gate = None
        config.output_rank, config.window = 1024, 128
        config.index_heads, config.index_dim, config.index_topk = 32, 128, 256
        config.epsilon, config.scale = 1e-6, 512 ** -0.5
        config.original_fp8_activation = True
        config.sparse_library_path = args.sparse_so
        config.indexer_score_directory = args.indexer_dir
        config.decode_cache_read_reuse = config.decode_fused_swa_read = True
        layer = native.NativeV41Attention(config, weights)
        angles = torch.arange(1024, device=device).float()[:, None] * torch.pow(10000.0, -torch.arange(0, 64, 2, device=device).float() / 64)[None, :]
        rope = torch.cat((angles.cos(), angles.sin()), -1).contiguous()
        # One-page SWA forces physical ring overwrite; compressed pages are permuted.
        swa = torch.tensor([[3]], device=device, dtype=torch.int32)
        table = torch.tensor([[4, 1, 5, 2]], device=device, dtype=torch.int32)
        cases = []
        stream = torch.musa.Stream()
        stream.wait_stream(torch.musa.default_stream())
        with torch.musa.stream(stream):
            for count in range(1, 7):
                for parity in (0, 1):
                    begin = 32 - parity
                    trial = native.NativeV41AttentionCache()
                    trial.swa = torch.zeros(7, 128, 512, device=device, dtype=dtype)
                    trial.key = torch.zeros_like(trial.swa)
                    trial.index = torch.zeros(7, 128, 128, device=device, dtype=dtype)
                    if args.ratio == 2:
                        trial.compressor_kv_state = torch.zeros(7, 32, 512, device=device)
                        trial.compressor_score_state = torch.zeros_like(trial.compressor_kv_state)
                    hidden = _random((count, 5120))
                    positions = torch.arange(begin, begin + count, device=device)
                    stable_table, stable_swa = table.clone(), swa.clone()

                    def forward() -> torch.Tensor:
                        runtime = native.NativeV41AttentionRuntime()
                        runtime.decode_capacity = 512
                        return layer.forward(hidden, positions, rope, trial, stable_swa,
                                             stable_table, stable_table, begin + count, runtime)

                    for _ in range(2):
                        forward()
                    stream.synchronize()
                    graph = torch.musa.MUSAGraph()
                    with torch.musa.graph(graph, stream=stream):
                        captured = forward()
                    # Different positions and page mappings reuse the same capture.
                    for scene, start in enumerate((32 - parity, 128 - parity, 256 - parity, 502 - parity)):
                        reference = _clone(trial)
                        for name in cache_names:
                            getattr(reference, name).zero_()
                        wanted_swa = swa if scene < 3 else torch.tensor([[2]], device=device, dtype=torch.int32)
                        wanted_table = table if scene < 3 else torch.tensor([[1, 5, 2, 4]], device=device, dtype=torch.int32)
                        layer.forward(_random((start, 5120)), torch.arange(start, device=device),
                                      rope, reference, wanted_swa, wanted_table, wanted_table,
                                      start, native.NativeV41AttentionRuntime())
                        for name in cache_names:
                            getattr(trial, name).copy_(getattr(reference, name))
                        stable_table.copy_(wanted_table)
                        stable_swa.copy_(wanted_swa)
                        hidden.copy_(_random((count, 5120)))
                        positions.copy_(torch.arange(start, start + count, device=device))
                        expected = layer.forward(hidden, positions, rope, reference, wanted_swa,
                                                 wanted_table, wanted_table, start + count,
                                                 native.NativeV41AttentionRuntime())
                        graph.replay()
                        output_error = _metrics(captured, expected)
                        states = {name: _metrics(getattr(trial, name), getattr(reference, name)) for name in cache_names}
                        next_hidden = _random((1, 5120))
                        next_position = torch.tensor([start + count], device=device)
                        expected_next = layer.forward(next_hidden, next_position, rope, reference,
                                                      wanted_swa, wanted_table, wanted_table,
                                                      start + count + 1, native.NativeV41AttentionRuntime())
                        actual_next = layer.forward(next_hidden, next_position, rope, trial,
                                                    wanted_swa, wanted_table, wanted_table,
                                                    start + count + 1, native.NativeV41AttentionRuntime())
                        next_error = _metrics(actual_next, expected_next)
                        case = {"rows": count, "parity": parity, "position": start,
                                "changed_physical_pages": scene == 3, "output": output_error,
                                "states": states, "next_output": next_error}
                        cases.append(case)
                        with args.output.with_suffix(".jsonl").open("a") as stream_out:
                            stream_out.write(json.dumps(case) + "\n")
                        assert output_error["relative_rms"] <= 0.01, case
                        assert next_error["relative_rms"] <= 0.01, case
                        assert all(value["relative_rms"] <= 1e-6 for value in states.values()), case
                    stream.synchronize()
                    graph.reset()
                    logger.info("Static native Attention graph verified: rows=%d parity=%d", count, parity)
            stream.synchronize()
    args.output.write_text(json.dumps({"status": "PASS_REAL_MULTIROW_ATTENTION_GRAPH",
                                      "compress_ratio": args.ratio,
                                      "cases": cases, "scope": "Actual Attention only; native model/auxiliary/scheduler integration pending",
                                      "accuracy_scope": "仅通过基础功能检查，未完成模型精度验收"}, indent=2))
    logger.info("Graph Attention replay scenarios passed: %d", len(cases))


if __name__ == "__main__":
    main()
