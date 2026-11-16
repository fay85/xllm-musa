# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selective speculative rollback against actual native ratio2 Attention.

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
import v41_prefix_checkpoint_test_module as native

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger

CACHE_NAMES = ("swa", "key", "index", "compressor_kv_state", "compressor_score_state")


def _random(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * 0.005).to(device="musa:0", dtype=torch.bfloat16)


def _clone(cache: object) -> object:
    result = native.NativeV41AttentionCache()
    for name in CACHE_NAMES:
        setattr(result, name, getattr(cache, name).clone())
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
    args = parser.parse_args()
    assert torch.musa.device_count() == 1
    torch.set_num_threads(1)
    torch.manual_seed(111635)
    device, dtype = "musa:0", torch.bfloat16
    with torch.inference_mode():
        weights = native.NativeV41AttentionWeights()
        weights.projection = [_random((1280, 5120)), torch.ones(1280, device=device, dtype=dtype), _random((4096, 1280)), _random((512, 5120)), torch.ones(512, device=device, dtype=dtype), _random((1024, 4096)), _random((5120, 1024))]
        weights.sink = torch.zeros(8, device=device)
        weights.compressor_norm = torch.ones(512, device=device, dtype=dtype)
        weights.compressor_kv = _random((512, 5120)).float()
        weights.compressor_gate = _random((512, 5120)).float()
        weights.indexer = [_random((4096, 1280)), _random((32, 5120)), _random((128, 512)), torch.ones(128, device=device, dtype=dtype)]
        config = native.NativeV41AttentionConfig()
        config.mode = native.NativeV41AttentionMode.FULL
        config.layer = config.source_layer = 0
        config.ratio, config.heads, config.groups = 2, 8, 1
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
        torch.musa.synchronize()
        for nondefault in (False, True):
            stream = torch.musa.Stream() if nondefault else torch.musa.current_stream()
            with torch.musa.stream(stream):
                for begin in (31, 32, 127, 128, 257, 511):
                    base = native.NativeV41AttentionCache()
                    base.swa = torch.zeros(7, 128, 512, device=device, dtype=dtype)
                    base.key = torch.zeros_like(base.swa)
                    base.index = torch.zeros(7, 128, 128, device=device, dtype=dtype)
                    base.compressor_kv_state = torch.zeros(7, 32, 512, device=device)
                    base.compressor_score_state = torch.zeros_like(base.compressor_kv_state)
                    prefill = _random((begin, 5120))
                    layer.forward(prefill, torch.arange(begin, device=device), rope, base, swa, table, table, begin, native.NativeV41AttentionRuntime())
                    proposals = _random((7, 5120))
                    for accepted in range(7):
                        trial, reference = _clone(base), _clone(base)
                        checkpoint = _checkpoint(trial, swa, table, begin)
                        verified = layer.forward(proposals[:6], torch.arange(begin, begin + 6, device=device), rope, trial, swa, table, table, begin + 6, native.NativeV41AttentionRuntime())
                        checkpoint.restore_suffix(accepted)
                        prefix_error = None
                        if accepted:
                            expected = layer.forward(proposals[:accepted], torch.arange(begin, begin + accepted, device=device), rope, reference, swa, table, table, begin + accepted, native.NativeV41AttentionRuntime())
                            prefix_error = _metrics(verified[:accepted], expected)
                        state_errors = {name: _metrics(getattr(trial, name), getattr(reference, name)) for name in CACHE_NAMES}
                        # Rows not changed by the accepted-only actual forward
                        # must match byte-for-byte, including overwritten SWA history.
                        unchanged_exact = {}
                        for name in CACHE_NAMES:
                            old = getattr(base, name).cpu().contiguous()
                            wanted = getattr(reference, name).cpu().contiguous()
                            actual = getattr(trial, name).cpu().contiguous()
                            same_rows = (old == wanted).all(dim=-1)
                            unchanged_exact[name] = torch.equal(actual[same_rows], wanted[same_rows])
                        positions = torch.tensor([begin + accepted], device=device)
                        hidden = proposals[accepted:accepted + 1]
                        expected_next = layer.forward(hidden, positions, rope, reference, swa, table, table, begin + accepted + 1, native.NativeV41AttentionRuntime())
                        actual_next = layer.forward(hidden, positions, rope, trial, swa, table, table, begin + accepted + 1, native.NativeV41AttentionRuntime())
                        next_error = _metrics(actual_next, expected_next)
                        case = {"nondefault_stream": nondefault, "begin": begin, "verified_inputs": 6, "accepted_inputs": accepted, "checkpoint_bytes": checkpoint.saved_bytes(), "prefix_output": prefix_error, "state_before_next": state_errors, "untouched_rows_exact": unchanged_exact, "next_output": next_error}
                        cases.append(case)
                        with args.output.with_suffix(".jsonl").open("a") as out:
                            out.write(json.dumps(case) + "\n")
                        # Empty acceptance must restore exactly. Nonempty shapes can
                        # select different BF16 GEMM/GEMV reductions; report their errors.
                        if accepted == 0:
                            assert all(item["exact"] for item in state_errors.values()), case
                        assert all(unchanged_exact.values()), case
                        assert next_error["relative_rms"] <= 0.03, case
                        assert prefix_error is None or prefix_error["relative_rms"] <= 0.03, case
                    logger.info("Actual prefix rollback checked: stream=%s begin=%d", nondefault, begin)
            stream.synchronize()
    result = {"status": "PASS_ACTUAL_RATIO2_PREFIX_RESTORE", "cases": cases, "scope": "One actual native Attention layer with MATE sparse/AOT indexer; ring overwrite, compressed-page permutation, even/odd positions, all acceptance lengths, next-token continuation. Model/scheduler/draft integration is not covered.", "accuracy_scope": "仅通过基础功能检查，未完成模型精度验收"}
    args.output.write_text(json.dumps(result, indent=2))
    logger.info("Prefix transaction checks passed: %d", len(cases))


if __name__ == "__main__":
    main()
