# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise production C1 GPU position and visible-cache inputs under real graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import v41_decode_graph_inputs_test_module as native

    if torch.musa.device_count() != 1:
        raise ValueError('Require one verified idle GPU')
    torch.set_num_threads(1)
    torch.manual_seed(1116)
    cases = []
    stream = torch.musa.Stream()
    with torch.inference_mode(), torch.musa.stream(stream):
        for width in (128, 512):
            for parity in (0, 1):
                value = torch.randn(1, width, device='musa')
                score = torch.randn_like(value)
                table = torch.tensor([[3, 1, 0, 2]], device='musa', dtype=torch.int64)
                states = [torch.randn(4, 32, width, device='musa') for _ in range(2)]
                position = torch.tensor([parity], device='musa', dtype=torch.int64)
                for _ in range(2):
                    native.pair_pool_decode(value, score, table, *states, position,
                                            bool(parity), 128)
                stream.synchronize()
                graph = torch.musa.MUSAGraph()
                with torch.musa.graph(graph, stream=stream):
                    output, starts = native.pair_pool_decode(
                        value, score, table, *states, position, bool(parity), 128)
                for base in (0, 2, 30, 32, 126, 128, 254, 256):
                    actual_position = base + parity
                    position.fill_(actual_position)
                    value.copy_(torch.randn_like(value))
                    score.copy_(torch.randn_like(score) * 8)
                    table.copy_(torch.tensor([[2, 0, 3, 1]] if base % 4 == 0
                                             else [[3, 1, 0, 2]],
                                             device='musa', dtype=torch.int64))
                    initial = [torch.randn_like(state) for state in states]
                    for target, source in zip(states, initial):
                        target.copy_(source)
                    oracle_states = [state.clone() for state in initial]
                    expected, expected_starts = native.pair_pool(
                        value, score, table, *oracle_states, actual_position, 128)
                    graph.replay()
                    stream.synchronize()
                    torch.testing.assert_close(output.cpu(), expected.cpu(), atol=0, rtol=0)
                    torch.testing.assert_close(starts.cpu(), expected_starts.cpu(), atol=0, rtol=0)
                    for got, want in zip(states, oracle_states):
                        torch.testing.assert_close(got.cpu(), want.cpu(), atol=0, rtol=0)
                    cases.append({'kind': 'pair_pool_graph', 'width': width,
                                  'position': actual_position, 'parity': parity,
                                  'outputs_and_state_exact': True})
        for width in (128, 512):
            table = torch.tensor([[3, 1, 4, 2]], device='musa', dtype=torch.int64)
            storage = torch.full((5, 128, width), float('nan'), device='musa',
                                 dtype=torch.bfloat16)
            visible = torch.tensor([0], device='musa', dtype=torch.int64)
            cache = native.TokenCache(storage)
            for _ in range(2):
                cache.gather_visible(table, visible, 512)
            stream.synchronize()
            graph = torch.musa.MUSAGraph()
            with torch.musa.graph(graph, stream=stream):
                output = cache.gather_visible(table, visible, 512)
            for count in (0, 1, 127, 128, 129, 255, 256, 257, 512, 1, 0):
                storage.fill_(float('nan'))
                table.copy_(torch.tensor([[2, 4, 1, 3]], device='musa', dtype=torch.int64))
                rows = torch.arange(count, device='musa', dtype=torch.int64)
                values = torch.randn(count, width, device='musa', dtype=torch.bfloat16)
                cache.commit(values, rows, table)
                table[:, (count + 127) // 128:] = -1
                visible.fill_(count)
                expected = cache.gather(table, count, 512)
                graph.replay()
                stream.synchronize()
                torch.testing.assert_close(output.cpu(), expected.cpu(), atol=0, rtol=0)
                if not bool(torch.isfinite(output).all().item()):
                    raise AssertionError('Unallocated NaN cache was exposed')
                cases.append({'kind': 'visible_cache_graph', 'width': width,
                              'visible': count, 'capacity': 512,
                              'unallocated_nan_masked': True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'PASS_REAL_GRAPH_DYNAMIC_POSITION_AND_VISIBILITY',
              'cases': cases, 'count': len(cases),
              'scope': 'Production kernels/cache class, changed inputs and page IDs under replay; not full-model accuracy'}
    args.output.write_text(json.dumps(result, indent=2))
    logger.info('Graph input checks passed: %s cases', len(cases))


if __name__ == '__main__':
    main()
