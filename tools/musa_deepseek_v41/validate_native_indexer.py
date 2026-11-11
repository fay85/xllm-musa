# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Native original-QDQ Indexer and paged cache against chronological oracles."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
import torch.nn.functional as F

from scripts.logger import logger
from tools.musa_deepseek_v41.validate_attention_quant import _oracle
from tools.musa_deepseek_v41.validate_native_attention_projection import _norm, _quant


def _fp4(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value.clone()
    return _oracle(value.reshape(-1, value.shape[-1]).cpu(), 32, False)[0].to(value.device).reshape_as(value)


def _select_cpu(scores: torch.Tensor, visible: torch.Tensor,
                topk: int, pool: torch.Tensor | None = None) -> torch.Tensor:
    scores = scores.cpu().float()
    visible = visible.cpu()
    if pool is not None:
        scores = scores.masked_fill(~pool.cpu(), -float('inf'))
    rows = []
    for row, limit in zip(scores.tolist(), visible.tolist()):
        indices = sorted(range(len(row)), key=lambda index: (-row[index], index))[:topk]
        rows.append([index if index < limit and row[index] > -float('inf') else -1
                     for index in sorted(indices)])
    return torch.tensor(rows, dtype=torch.int32).reshape(len(scores), min(topk, scores.shape[1]))


def _candidates_cpu(scores: torch.Tensor, visible: torch.Tensor,
                    count: int, block_size: int) -> torch.Tensor:
    scores, visible = scores.cpu(), visible.cpu()
    result = torch.zeros_like(scores, dtype=torch.bool)
    for row, limit in enumerate(visible.tolist()):
        blocks = [float(scores[row, begin:begin+block_size].max())
                  for begin in range(0, scores.shape[1], block_size)]
        last = (limit-1)//block_size
        if 0 <= last < len(blocks):
            blocks[last] = float('inf')
        chosen = sorted(range(len(blocks)), key=lambda index: (-blocks[index], index))[:count]
        for block in chosen:
            if blocks[block] > -float('inf'):
                result[row, block*block_size:(block+1)*block_size] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime'] = native
    os.environ['XLLM_DSV41_MUSA_ROPE'] = 'v4_fused'
    from xllm.python import initialize_runtime
    initialize_runtime()
    from xllm.python.kernels_musa.rotary import select_rotary
    from xllm.python.attention.csa2_attention import _select_candidate_blocks
    select_rotary.cache_clear()
    rope = select_rotary()
    torch.manual_seed(111156)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {'status': 'RUNNING',
        'scope': 'native replicated-head Indexer Q/K/group32, scoring/candidates/topk and scheduler paged token cache; synthetic full/small shapes; excludes full model and real network P/D',
        'checks': []}

    def record(value: dict[str, Any]) -> None:
        result['checks'].append(value)
        args.output.write_text(json.dumps(result, indent=2))

    with torch.inference_mode():
        for hidden_dim, latent_dim, heads, dim, key_dim in [(64, 32, 4, 32, 32), (5120, 1280, 32, 128, 512)]:
            weights = [torch.randn(heads*dim, latent_dim, device='musa', dtype=torch.bfloat16)*.01,
                       torch.randn(heads, hidden_dim, device='musa', dtype=torch.bfloat16)*.01,
                       torch.randn(dim, key_dim, device='musa', dtype=torch.bfloat16)*.01,
                       (torch.randn(dim, device='musa')*.01+1).bfloat16()]
            width = min(64, dim)
            angles = torch.arange(600, device='musa')[:, None].float()*torch.linspace(.01, .5, width//2, device='musa')
            table = torch.cat((angles.cos(), angles.sin()), dim=-1)
            for qdq in (False, True):
                layer = native.NativeV41Indexer(weights, heads, dim, 1e-20, qdq)
                for count in (0, 1, 17, 129):
                    hidden = torch.randn(count, hidden_dim, device='musa', dtype=torch.bfloat16)
                    qr = torch.randn(count, latent_dim, device='musa', dtype=torch.bfloat16)
                    latent = torch.randn(count, key_dim, device='musa', dtype=torch.bfloat16)
                    positions = torch.arange(count, device='musa', dtype=torch.int64)*2+1
                    projected = F.linear(_quant(qr, qdq), weights[0]).reshape(count, heads, dim)
                    expected_q = _fp4(rope(projected, positions, table, False))
                    expected_w = F.linear(hidden, weights[1])*(dim**-.5*heads**-.5)
                    expected_k = _fp4(rope(_norm(F.linear(latent, weights[2]), weights[3], 1e-20), positions, table, False))
                    query, head_weights = layer.prepare_query(hidden, qr, positions, table)
                    key = layer.prepare_key(latent, positions, table)
                    torch.testing.assert_close(query, expected_q, rtol=0, atol=0)
                    torch.testing.assert_close(head_weights, expected_w, rtol=0, atol=0)
                    torch.testing.assert_close(key, expected_k, rtol=0, atol=0)
                    record({'hidden': hidden_dim, 'rows': count, 'dense_qdq': qdq, 'query_key_and_head_weights': 'bitwise'})
            # Reindex layers do not own the source key projection.
            reindex = native.NativeV41Indexer(weights[:2], heads, dim, 1e-20, True)
            rq, rw = reindex.prepare_query(hidden, qr, positions, table)
            torch.testing.assert_close(rq, query, rtol=0, atol=0)
            torch.testing.assert_close(rw, head_weights, rtol=0, atol=0)
            record({'hidden': hidden_dim, 'reindex_without_key_weights': True})
            for ratio in (1, 2):
                for capacity, committed in [(0, 0), (1, 1), (17, 13), (129, 65), (260, 130)]:
                    count = 17
                    q = _fp4(torch.randn(count, heads, dim, device='musa', dtype=torch.bfloat16))
                    keys = _fp4(torch.randn(capacity, dim, device='musa', dtype=torch.bfloat16))
                    keys[committed:] = float('nan')
                    hw = torch.randn(count, heads, device='musa', dtype=torch.bfloat16)
                    pos = torch.arange(count, device='musa', dtype=torch.int64)+max(0, committed*ratio-8)
                    visible = (pos+1)//ratio
                    safe = torch.where((torch.arange(capacity, device='musa') < committed)[:, None], keys, 0)
                    expected = (torch.einsum('shd,td->sht', q.float(), safe.float()).relu()*hw.float()[..., None]).sum(1)
                    columns = torch.arange(capacity, device='musa')
                    expected.masked_fill_((columns >= visible[:, None]) | (columns >= committed), -float('inf'))
                    actual = layer.score(q, keys, hw, pos, committed, ratio)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(layer.select(actual, visible, 32, None).cpu(),
                                               _select_cpu(expected, visible, 32), rtol=0, atol=0)
                    pool = layer.candidates(actual, visible, 2, 16)
                    pool_ref = _candidates_cpu(expected, visible, 2, 16)
                    if capacity:
                        python_pool = _select_candidate_blocks(expected, visible, 2, 16, stable_ties=True)
                        assert torch.equal(python_pool.cpu(), pool_ref)
                    if not torch.equal(pool.cpu(), pool_ref):
                        result.update(status='FAIL', failure={'hidden': hidden_dim, 'ratio': ratio,
                            'capacity': capacity, 'committed': committed, 'pool': pool.cpu().tolist(),
                            'pool_reference': pool_ref.tolist(), 'visible': visible.cpu().tolist(),
                            'scores': expected.cpu().tolist(),
                            'gpu_block_amax': F.pad(expected, (0, -capacity % 16), value=-float('inf')).unflatten(-1, (-1, 16)).amax(-1).cpu().tolist(),
                            'cpu_block_amax': F.pad(expected.cpu(), (0, -capacity % 16), value=-float('inf')).unflatten(-1, (-1, 16)).amax(-1).tolist()})
                        args.output.write_text(json.dumps(result, indent=2))
                        torch.save({'scores': expected.cpu(), 'visible': visible.cpu(),
                                    'actual': pool.cpu(), 'expected': pool_ref}, args.output.with_suffix('.failure.pt'))
                        raise AssertionError('Native candidate block mask differs from scalar CPU oracle')
                    torch.testing.assert_close(layer.select(actual, visible, 32, pool).cpu(),
                                               _select_cpu(expected, visible, 32, pool_ref), rtol=0, atol=0)
                    record({'hidden': hidden_dim, 'ratio': ratio, 'capacity': capacity, 'committed': committed,
                            'score': 'bitwise GPU oracle', 'selection': 'independent scalar CPU', 'nan_tail': True})
            # Capacity-independent zero-score ties, newest partial block, no visibility.
            for capacity in (0, 17, 129, 260):
                visible = torch.tensor([0, 1, 17, 33], device='musa', dtype=torch.int64).clamp_max(capacity)
                scores = torch.zeros(4, capacity, device='musa')
                scores.masked_fill_(torch.arange(capacity, device='musa') >= visible[:, None], -float('inf'))
                pool = layer.candidates(scores, visible, 2, 16)
                assert torch.equal(pool.cpu(), _candidates_cpu(scores, visible, 2, 16))
                assert torch.equal(layer.select(scores, visible, 32, pool).cpu(), _select_cpu(scores, visible, 32, pool))
                record({'hidden': hidden_dim, 'capacity': capacity, 'zero_score_ties_and_no_visible': True})

        # Physical token-cache layout checked independently, including CPU
        # state transfer and new physical page IDs (not a Mooncake test).
        for dim in (32, 128, 512):
            bs, pages = 8, [5, 2, 7, 1]
            table = torch.tensor([pages], device='musa', dtype=torch.int32)
            cache = torch.full((9, bs, dim), float('nan'), device='musa', dtype=torch.bfloat16)
            expected = cache.cpu()
            layer = native.NativeV41TokenCache(cache)
            committed = 0
            history = []
            for count in (1, 7, 9, 1):
                values = torch.randn(count, dim, device='musa', dtype=torch.bfloat16)
                rows = torch.arange(committed, committed+count, device='musa', dtype=torch.int64)
                layer.commit(values, rows, table)
                for row, value in zip(rows.cpu().tolist(), values.cpu()):
                    expected[pages[row//bs], row % bs] = value
                committed += count
                history.append(values.cpu())
                torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0, equal_nan=True)
                gathered = layer.gather(table, committed, 32).cpu()
                torch.testing.assert_close(gathered[:committed], torch.cat(history), rtol=0, atol=0)
                assert torch.equal(gathered[committed:], torch.zeros_like(gathered[committed:]))
                record({'cache_dim': dim, 'committed': committed, 'physical_and_gather': 'bitwise', 'nan_tail_zeroed': True})
            changed = [3, 6, 0, 4]
            restored = torch.full_like(expected, float('nan'))
            for old, new in zip(pages, changed):
                restored[new] = expected[old]
            copy = restored.to('musa')
            remap = torch.tensor([changed], device='musa', dtype=torch.int32)
            restored_layer = native.NativeV41TokenCache(copy)
            torch.testing.assert_close(restored_layer.gather(remap, committed, 32).cpu(),
                                       layer.gather(table, committed, 32).cpu(), rtol=0, atol=0)
            restored_layer.commit(values[:0], rows[:0], remap)
            torch.testing.assert_close(copy.cpu(), restored, rtol=0, atol=0, equal_nan=True)
            record({'cache_dim': dim, 'cpu_restore_physical_remap_and_empty': True})
    result.update(status='PASS', count=len(result['checks']))
    args.output.write_text(json.dumps(result, indent=2))
    logger.info('Native Indexer/token-cache checks passed: %s', result['count'])


if __name__ == '__main__':
    main()
