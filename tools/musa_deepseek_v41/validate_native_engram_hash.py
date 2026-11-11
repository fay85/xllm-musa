# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Native C1 Engram hash/state against Python and independent scalar history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401

from scripts.logger import logger


def _oracle(tokens: list[int], begin: int, count: int, token_map: torch.Tensor,
            multipliers: torch.Tensor, primes: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    layers, grams = multipliers.shape
    heads = primes.shape[-1]
    result = torch.empty(count, layers, (grams-1)*heads, dtype=torch.int32)
    mapping, factors = token_map.tolist(), multipliers.tolist()
    prime_values, offset_values = primes.tolist(), offsets.tolist()
    for row in range(count):
        for layer in range(layers):
            rolling, blocked = 0, False
            for shift in range(grams):
                previous = begin+row-shift
                token = tokens[previous] if previous >= 0 else 0
                blocked = blocked or previous < 0 or token in (129264, 129265)
                value = mapping[0] if blocked else mapping[max(0, min(token, len(mapping)-1))]
                rolling ^= (value*factors[layer][shift]) & ((1 << 64)-1)
                signed = rolling-(1 << 64) if rolling >= (1 << 63) else rolling
                if shift:
                    for head in range(heads):
                        column = (shift-1)*heads+head
                        result[row, layer, column] = signed % prime_values[layer][shift-1][head]+offset_values[layer][column]
    return result


def _slots(positions: torch.Tensor, table: torch.Tensor, block_size: int) -> torch.Tensor:
    return table[0, (positions//block_size) % table.shape[1]].long()*block_size+positions % block_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime'] = native
    from xllm.python import initialize_runtime
    initialize_runtime()
    from xllm.python.models.deepseek_v41_engram import (
        EngramLayout, EngramHashState, compute_hash_multipliers, image_sentinel_mask,
    )
    layout = EngramLayout([1, 14], [5000, 5000], 4, 4, 32, 64, 16, 0)
    mapping = torch.arange(64) % 16
    factors = compute_hash_multipliers(layout.layer_ids, 4, 16)
    primes = torch.tensor(layout.primes, dtype=torch.int64)
    offsets = layout.offsets
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {'status': 'RUNNING', 'checks': [],
        'scope': 'native C1 eager hash and receiver-local history; two synthetic Engram layers; scalar CPU oracle plus current Python; host bootstrap/remap simulation not network PD'}
    torch.manual_seed(111158)
    with torch.inference_mode():
        for block_size in (8, 128):
            for explicit in (False, True):
                reference = EngramHashState(layout, mapping, factors).to('musa')
                implementation = native.NativeV41EngramHash(
                    mapping.to('musa'), factors.to('musa'), primes.to('musa'), offsets.to('musa'), 0)
                swa = torch.zeros(8, block_size, 32, device='musa', dtype=torch.bfloat16)
                reference.ensure_cache(swa, block_size)
                history = torch.zeros(8*block_size, device='musa', dtype=torch.int32)
                implementation.bind_history(history)
                table = torch.tensor([[5, 2]], device='musa', dtype=torch.int32)
                chunks = [1, 3, 34, 1, 129, 3]
                tokens = torch.randint(0, 64, (sum(chunks),))
                tokens[5] = 129264
                tokens[32] = 129265
                tokens[128] = 129264
                previous = 0
                for step, count in enumerate(chunks):
                    positions = torch.arange(previous, previous+count, device='musa')
                    ids = tokens[previous:previous+count].to('musa')
                    lookback = None
                    if explicit and previous:
                        past = tokens[max(0, previous-3):previous].flip(0)
                        lookback = torch.full((1, 3), -1, device='musa', dtype=torch.int64)
                        lookback[0, :len(past)] = past.to('musa')
                        if step % 2:
                            lookback[0, 1] = -1
                    expected = reference(ids, positions, torch.tensor([0, count], device='musa'),
                        image_sentinel_mask(ids), lookback, _slots(positions, table, block_size),
                        table, block_size, max_chunk_len=count)
                    actual, mask = implementation.forward(ids, positions, table, block_size, previous, lookback)
                    oracle = _oracle(tokens.tolist(), previous, count, mapping, factors, primes, offsets)
                    try:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(actual.cpu(), oracle, rtol=0, atol=0)
                        assert torch.equal(mask, ~image_sentinel_mask(ids))
                        assert torch.equal(history, reference._cache)
                    except AssertionError:
                        result.update(status='FAIL', failure={'start': previous, 'count': count,
                            'block_size': block_size, 'explicit': explicit})
                        args.output.write_text(json.dumps(result, indent=2))
                        torch.save({'native': actual.cpu(), 'python': expected.cpu(), 'scalar': oracle,
                            'native_history': history.cpu(), 'python_history': reference._cache.cpu()},
                            args.output.with_suffix('.failure.pt'))
                        raise
                    result['checks'].append({'block_size': block_size, 'explicit': explicit,
                        'start': previous, 'count': count, 'hash_mask_and_physical_history': 'bitwise'})
                    args.output.write_text(json.dumps(result, indent=2))
                    previous += count
                    if step == 1:
                        # Fresh receiver, remapped SWA physical pages. Restore
                        # only the required original-token tail via PD pairs.
                        table = torch.tensor([[3, 6]], device='musa', dtype=torch.int32)
                        swa = torch.zeros_like(swa)
                        reference.ensure_cache(swa, block_size)
                        history = torch.zeros_like(history)
                        implementation.bind_history(history)
                        tail = torch.arange(max(0, previous-3), previous, device='musa')
                        pairs = torch.stack((_slots(tail, table, block_size),
                                             tokens[max(0, previous-3):previous].to('musa')), dim=1)
                        reference.restore_history(pairs)
                        implementation.restore_history(pairs)
                        assert torch.equal(history, reference._cache)
                        result['checks'].append({'block_size': block_size, 'explicit': explicit,
                            'fresh_receiver_tail_bootstrap_physical_remap': True})
                empty, mask = implementation.forward(ids[:0], positions[:0], table, block_size, previous, None)
                assert empty.shape == (0, 2, 12) and mask.numel() == 0
                assert torch.equal(history, reference._cache)
                result['checks'].append({'block_size': block_size, 'explicit': explicit, 'empty_inert': True})
    result.update(status='PASS', count=len(result['checks']))
    args.output.write_text(json.dumps(result, indent=2))
    logger.info('Native Engram hash checks passed: %s', result['count'])


if __name__ == '__main__':
    main()
