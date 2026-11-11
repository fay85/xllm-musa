# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Original compressor pooling against complete chronological projection history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
import torch.nn.functional as F

from scripts.logger import logger


def _norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True)+1e-20)*weight.float()).bfloat16()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    torch.manual_seed(111154)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'RUNNING', 'scope': 'native C1 compressor ratio1/2 projection/pool/norm/ring32; chronological GPU oracle plus independent CPU physical writes; not full model',
              'checks': []}
    with torch.inference_mode():
        for hidden, dim, block_size, chunks in [(64, 32, 8, [1, 1, 3, 34, 1, 65, 1]),
                                               (5120, 512, 128, [1, 256, 3, 1, 129, 1])]:
            for ratio in (1, 2):
                dtype = torch.float32 if ratio == 2 else torch.bfloat16
                weight = torch.randn(dim, hidden, device='musa', dtype=dtype)*.01
                norm_weight = (torch.randn(dim, device='musa')*.01+1).bfloat16()
                gate = torch.randn_like(weight)*.1 if ratio == 2 else None
                kv_state = torch.full((8, 32, dim), float('nan'), device='musa') if ratio == 2 else None
                score_state = torch.full_like(kv_state, float('nan')) if ratio == 2 else None
                expected_kv = kv_state.cpu() if kv_state is not None else None
                expected_score = score_state.cpu() if score_state is not None else None
                physical = [5, 2]
                table = torch.tensor([physical], device='musa', dtype=torch.int32)
                layer = native.NativeV41Compressor(weight, norm_weight, gate, kv_state, score_state, ratio, block_size, 1e-20)
                history_kv = torch.empty(0, dim, device='musa', dtype=dtype)
                history_score = torch.empty_like(history_kv)
                previous = 0
                for step, count in enumerate(chunks):
                    value = torch.randn(count, hidden, device='musa', dtype=torch.bfloat16)
                    value_copy = value.clone()
                    raw = F.linear(value.float() if ratio == 2 else value, weight)
                    scores = F.linear(value.float(), gate) if ratio == 2 else torch.zeros_like(raw)
                    history_kv = torch.cat((history_kv, raw))
                    history_score = torch.cat((history_score, scores))
                    begin, end = previous//ratio, (previous+count)//ratio
                    starts = torch.arange(begin, end, device='musa', dtype=torch.int64)*ratio
                    if ratio == 1:
                        expected = _norm(raw, norm_weight)
                    elif end > begin:
                        chronological = history_kv[begin*ratio:end*ratio].reshape(end-begin, ratio, dim)
                        chronological_scores = history_score[begin*ratio:end*ratio].reshape(end-begin, ratio, dim)
                        pooled = (chronological*chronological_scores.softmax(dim=1)).sum(1).bfloat16()
                        expected = _norm(pooled, norm_weight)
                    else:
                        expected = torch.empty(0, dim, device='musa', dtype=torch.bfloat16)
                    actual, actual_starts = layer.forward(value, previous, table)
                    try:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(actual_starts, starts, rtol=0, atol=0)
                    except AssertionError:
                        result.update(status='FAIL', failure={'hidden': hidden, 'ratio': ratio, 'start': previous, 'count': count})
                        args.output.write_text(json.dumps(result, indent=2))
                        torch.save({'actual': actual.cpu(), 'expected': expected.cpu(), 'value': value.cpu(),
                                    'kv_state': None if kv_state is None else kv_state.cpu(),
                                    'score_state': None if score_state is None else score_state.cpu()}, args.output.with_suffix('.failure.pt'))
                        raise
                    if ratio == 2:
                        raw_cpu, score_cpu = raw.cpu(), scores.cpu()
                        for row in range(max(0, count-32), count):
                            absolute = previous+row
                            if (absolute+1) % ratio:
                                block = physical[(absolute//block_size) % len(physical)]
                                expected_kv[block, absolute % 32] = raw_cpu[row]
                                expected_score[block, absolute % 32] = score_cpu[row]
                        torch.testing.assert_close(kv_state.cpu(), expected_kv, rtol=0, atol=0, equal_nan=True)
                        torch.testing.assert_close(score_state.cpu(), expected_score, rtol=0, atol=0, equal_nan=True)
                    assert torch.equal(value, value_copy)
                    result['checks'].append({'hidden': hidden, 'dim': dim, 'ratio': ratio,
                                              'start': previous, 'rows': count, 'committed_groups': end-begin,
                                              'latent_and_physical_state': 'bitwise', 'ownership': True})
                    args.output.write_text(json.dumps(result, indent=2))
                    previous += count
                    if ratio == 2 and step == 1:
                        changed = [3, 6]
                        restored_kv = torch.full_like(kv_state, float('nan'))
                        restored_score = torch.full_like(score_state, float('nan'))
                        new_kv, new_score = torch.full_like(expected_kv, float('nan')), torch.full_like(expected_score, float('nan'))
                        for old, new in zip(physical, changed):
                            restored_kv[new].copy_(expected_kv[old].to('musa'))
                            restored_score[new].copy_(expected_score[old].to('musa'))
                            new_kv[new], new_score[new] = expected_kv[old], expected_score[old]
                        physical, kv_state, score_state = changed, restored_kv, restored_score
                        expected_kv, expected_score = new_kv, new_score
                        table = torch.tensor([physical], device='musa', dtype=torch.int32)
                        layer = native.NativeV41Compressor(weight, norm_weight, gate, kv_state, score_state, ratio, block_size, 1e-20)
                empty, starts = layer.forward(value[:0], previous, table)
                assert empty.shape == (0, dim) and starts.numel() == 0
                if ratio == 2:
                    torch.testing.assert_close(kv_state.cpu(), expected_kv, rtol=0, atol=0, equal_nan=True)
                    torch.testing.assert_close(score_state.cpu(), expected_score, rtol=0, atol=0, equal_nan=True)
                result['checks'].append({'hidden': hidden, 'ratio': ratio, 'zero_input_preserves_state': True})
    result.update(status='PASS', count=len(result['checks']))
    args.output.write_text(json.dumps(result, indent=2))
    logger.info('Native compressor checks passed: %s', result['count'])


if __name__ == '__main__':
    main()
