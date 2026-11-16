# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare small-row expert-parallel routing to the unchanged native gate."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch_musa
import v41_parallel_gate_test_module as native


def _gate(logits: torch.Tensor, bias: torch.Tensor, topk: int,
          normalize: bool, parallel: bool) -> tuple[torch.Tensor, torch.Tensor]:
    os.environ['XLLM_DSV41_PARALLEL_GATE_TOPK'] = '1' if parallel else '0'
    return native.gate(logits, bias, topk, normalize, 1.5)


def _equal(actual: tuple[torch.Tensor, torch.Tensor],
           reference: tuple[torch.Tensor, torch.Tensor]) -> None:
    assert torch.equal(actual[1], reference[1])
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0,
                               equal_nan=True)


def _graph_time(logits: torch.Tensor, bias: torch.Tensor,
                parallel: bool) -> dict[str, Any]:
    os.environ['XLLM_DSV41_PARALLEL_GATE_TOPK'] = '1' if parallel else '0'
    for _ in range(10):
        native.gate(logits, bias, 6, True, 1.5)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        outputs = [native.gate(logits, bias, 6, True, 1.5) for _ in range(40)]
    for _ in range(5):
        graph.replay()
    torch.musa.synchronize()
    samples = []
    for _ in range(5):
        begin, end = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
        begin.record()
        for _ in range(50):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / (50 * 40))
    _equal(outputs[-1], _gate(logits, bias, 6, True, False))
    # Destroy before any subsequent profiler; do not leave a reset Python graph
    # alive in torch_musa's graph WeakSet (E26 lifecycle failure).
    del graph, outputs
    gc.collect()
    return {'rows': logits.size(0), 'parallel': parallel,
            'graph_calls': 40, 'replays_per_window': 50,
            'microseconds_per_gate': samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(29)
    torch.musa.set_device(0)
    checks = []
    graphs = []
    timings = []
    stream = torch.musa.Stream()
    with torch.inference_mode():
        for alternate in [False, True]:
            selected = stream if alternate else torch.musa.default_stream()
            with torch.musa.stream(selected):
                for rows in [1, 4, 6, 7]:
                    for experts in [128, 256, 384, 512, 513]:
                        for topk in [1, 6]:
                            for per_token in [False, True]:
                                bias_shape = (rows, experts) if per_token else (experts,)
                                logits = torch.randn(rows, experts, device='musa', dtype=torch.float32)
                                bias = torch.randn(bias_shape, device='musa', dtype=torch.float32) * .1
                                for case in ['normal', 'ties', 'large_bias', 'extreme']:
                                    x, b = logits.clone(), bias.clone()
                                    if case == 'ties':
                                        x.zero_()
                                        b.zero_()
                                    elif case == 'large_bias':
                                        x.fill_(-40.)
                                        b.mul_(1e4)
                                    elif case == 'extreme':
                                        x[:, :8] = torch.tensor([float('nan'), float('inf'), -float('inf'), -100., -20., 20., 100., 0.], device='musa')
                                    normalize = case != 'large_bias'
                                    expected = _gate(x, b, topk, normalize, False)
                                    actual = _gate(x, b, topk, normalize, True)
                                    _equal(actual, expected)
                                    checks.append({'rows':rows,'experts':experts,'topk':topk,
                                                   'per_token_bias':per_token,'case':case,
                                                   'alternate_stream':alternate})
                selected.synchronize()
        with torch.musa.stream(stream):
            for rows in [1, 4, 6]:
                for per_token in [False, True]:
                    logits = torch.randn(rows, 384, device='musa')
                    bias = torch.randn((rows, 384) if per_token else (384,), device='musa') * .01
                    for _ in range(3):
                        _gate(logits, bias, 6, True, True)
                    stream.synchronize()
                    graph = torch.musa.MUSAGraph()
                    with torch.musa.graph(graph):
                        actual = _gate(logits, bias, 6, True, True)
                    for multiplier in [2., -1., 0.]:
                        logits.mul_(multiplier)
                        bias.add_(.01)
                        graph.replay()
                        stream.synchronize()
                        _equal(actual, _gate(logits, bias, 6, True, False))
                        graphs.append({'rows':rows,'per_token_bias':per_token,'update':multiplier})
                    del graph, actual
                    gc.collect()
            (args.output/'checks.json').write_text(json.dumps({'status':'PASS','exact_output_checks':checks,'graph_checks':graphs},indent=2))
            for rows in [1, 4, 6]:
                logits = torch.randn(rows, 384, device='musa')
                bias = torch.randn(384, device='musa') * .01
                for label, parallel in [('off',False),('on',True),('reverse_off',False)]:
                    result = _graph_time(logits, bias, parallel)
                    result['label'] = label
                    timings.append(result)
                    (args.output/'timings.json').write_text(json.dumps(timings,indent=2))
    torch.musa.synchronize()
    (args.output/'result.json').write_text(json.dumps({'status':'PASS','exact_checks':len(checks),'dynamic_graph_checks':len(graphs),'timings':timings,'scope':'Synthetic gate only, no model promotion or accuracy claim'},indent=2))


if __name__ == '__main__':
    main()
