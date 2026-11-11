# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Native ring paging against full chronological KV and independent CPU writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401

from scripts.logger import logger
from tools.musa_deepseek_v41.validate_native_attention_dense import _attention_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    torch.manual_seed(111153)
    record = {'status': 'RUNNING', 'scope': 'C1 native paged SWA read/commit and sparse attention; optional precomputed compressed picks; not compressor/indexer or whole native model',
              'checks': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for window, dim, heads, chunks in [(8, 32, 4, [1, 3, 2, 17, 1, 9]),
                                           (128, 512, 8, [129, 128, 3, 1])]:
            for compressed_count in (None, 0, 17):
                cache = torch.full((8, window, dim), float('nan'), device='musa', dtype=torch.bfloat16)
                expected_cache = cache.cpu()
                physical = [5, 2]
                table = torch.tensor([physical], device='musa', dtype=torch.int32)
                layer = native.NativeV41PagedAttention(cache, window, dim**-.5)
                history = torch.empty(0, dim, device='musa', dtype=torch.bfloat16)
                sink = torch.randn(heads, device='musa', dtype=torch.float32)
                for step, count in enumerate(chunks):
                    previous = len(history)
                    current = torch.randn(count, dim, device='musa', dtype=torch.bfloat16)
                    query = torch.randn(count, heads, dim, device='musa', dtype=torch.bfloat16)
                    query_copy, current_copy = query.clone(), current.clone()
                    full = torch.cat((history, current))
                    positions = torch.arange(previous, previous+count, device='musa')
                    wanted = positions[:, None]-window+1+torch.arange(window, device='musa')
                    picks = torch.where(wanted >= 0, wanted, -1)
                    compressed, extra = None, None
                    values = full
                    if compressed_count is not None:
                        width = 12 if dim == 32 else 512
                        compressed = torch.randn(compressed_count, dim, device='musa', dtype=torch.bfloat16)
                        if compressed_count:
                            extra = torch.arange(width, device='musa').repeat(count, 1) % compressed_count
                            extra[:, ::3] = -1
                            extra[:, 1] = 999999
                        else:
                            extra = torch.full((count, width), -1, device='musa', dtype=torch.int64)
                        valid = (extra >= 0) & (extra < compressed_count)
                        picks = torch.cat((picks, torch.where(valid, extra+len(full), -1)), dim=-1)
                        values = torch.cat((full, compressed))
                    expected = _attention_reference(query, values, picks, sink, dim**-.5, 32*1024*1024)
                    actual = layer.forward(query, current, table, len(full), sink, compressed, extra)
                    try:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    except AssertionError:
                        record.update(status='FAIL', failure={'window': window, 'step': step,
                                      'compressed_count': compressed_count, 'count': count})
                        args.output.write_text(json.dumps(record, indent=2))
                        torch.save({'query': query.cpu(), 'current': current.cpu(), 'history': history.cpu(),
                                    'actual': actual.cpu(), 'expected': expected.cpu(), 'cache': cache.cpu(),
                                    'table': table.cpu()}, args.output.with_suffix('.failure.pt'))
                        raise
                    # Independent test-only CPU sequential writes define latest
                    # absolute-position ownership, rather than sharing argsort.
                    rows = current.cpu()
                    for row in range(count):
                        absolute = previous+row
                        block = physical[(absolute//window) % len(physical)]
                        expected_cache[block, absolute % window] = rows[row]
                    torch.testing.assert_close(cache.cpu(), expected_cache, rtol=0, atol=0, equal_nan=True)
                    assert torch.equal(query, query_copy) and torch.equal(current, current_copy)
                    record['checks'].append({'window': window, 'heads': heads, 'dim': dim,
                                             'previous': previous, 'rows': count,
                                             'compressed_count': compressed_count,
                                             'output_and_physical_cache': 'bitwise', 'ownership': True})
                    args.output.write_text(json.dumps(record, indent=2))
                    history = full
                    if step == 1:
                        # Simulate a CPU state transfer and reversed physical
                        # page assignment. Both cache and table are real tensors.
                        changed = [3, 6]
                        restored = torch.full_like(cache, float('nan'))
                        restored_expected = torch.full_like(expected_cache, float('nan'))
                        for old, new in zip(physical, changed):
                            restored[new].copy_(expected_cache[old].to('musa'))
                            restored_expected[new] = expected_cache[old]
                        cache, expected_cache, physical = restored, restored_expected, changed
                        table = torch.tensor([physical], device='musa', dtype=torch.int32)
                        layer = native.NativeV41PagedAttention(cache, window, dim**-.5)
                empty = layer.forward(query[:0], current[:0], table, len(history), sink, None, None)
                assert empty.shape == query[:0].shape
                torch.testing.assert_close(cache.cpu(), expected_cache, rtol=0, atol=0, equal_nan=True)
                record['checks'].append({'window': window, 'compressed_count': compressed_count,
                                         'zero_query': True, 'no_cache_mutation': True})
    record.update(status='PASS', count=len(record['checks']))
    args.output.write_text(json.dumps(record, indent=2))
    logger.info('Native paged attention checks passed: %s', record['count'])


if __name__ == '__main__':
    main()
