# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate exact 4x4 HC metadata and independent graph replay timing."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
from typing import Callable, Any
import torch
import torch_musa
import v41_hc_metadata_warp_test_module as native


def _reference(x: torch.Tensor, s: torch.Tensor, b: torch.Tensor,
               iterations: int, eps: float) -> tuple[torch.Tensor, ...]:
    return native.reference(x, s, b, 4, iterations, eps)


def _equal(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> None:
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


def _timing(fn: Callable[..., Any], x: torch.Tensor, s: torch.Tensor,
            b: torch.Tensor) -> list[float]:
    for _ in range(4):
        fn(x, s, b, 20, 1e-6)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        outputs = [fn(x, s, b, 20, 1e-6) for _ in range(80)]
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
        samples.append(begin.elapsed_time(end) * 1000 / (50 * 80))
    _equal(outputs[-1], _reference(x, s, b, 20, 1e-6))
    del graph, outputs
    gc.collect()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(30)
    torch.musa.set_device(0)
    stream = torch.musa.Stream()
    checks, graphs, timings = [], [], []
    with torch.inference_mode():
        for alternate in [False, True]:
            chosen = stream if alternate else torch.musa.default_stream()
            with torch.musa.stream(chosen):
                for rows in [0, 1, 4, 6, 9]:
                    for iterations in [1, 2, 20]:
                        for eps in [0., 1e-6, 1e-3]:
                            for magnitude in [0., 1., 20.]:
                                x = torch.randn(rows, 24, device='musa') * magnitude
                                scale = torch.randn(3, device='musa')
                                base = torch.randn(24, device='musa')
                                expected = _reference(x, scale, base, iterations, eps)
                                actual = native.candidate(x, scale, base, iterations, eps)
                                case = {'rows':rows,'iterations':iterations,'eps':eps,'magnitude':magnitude,'alternate':alternate}
                                (args.output/'current_case.json').write_text(json.dumps(case))
                                _equal(actual, expected)
                                checks.append(case)
                chosen.synchronize()
        with torch.musa.stream(stream):
            for rows in [1,4,6]:
                x = torch.randn(rows,24,device='musa')
                scale = torch.randn(3,device='musa')
                base = torch.randn(24,device='musa')
                for _ in range(3):
                    native.candidate(x,scale,base,20,1e-6)
                stream.synchronize()
                graph = torch.musa.MUSAGraph()
                with torch.musa.graph(graph):
                    actual = native.candidate(x,scale,base,20,1e-6)
                for iteration in range(3):
                    x.copy_(torch.randn_like(x));scale.copy_(torch.randn_like(scale));base.copy_(torch.randn_like(base))
                    graph.replay();stream.synchronize()
                    _equal(actual,_reference(x,scale,base,20,1e-6));graphs.append({'rows':rows,'update':iteration})
                del graph,actual
                gc.collect()
            (args.output/'checks.json').write_text(json.dumps({'status':'PASS','cases':checks,'graphs':graphs},indent=2))
            for rows in [1,4,6]:
                x=torch.randn(rows,24,device='musa');scale=torch.randn(3,device='musa');base=torch.randn(24,device='musa')
                for label,fn in [('off',_reference),('on',native.candidate),('reverse_off',_reference)]:
                    timings.append({'rows':rows,'label':label,'microseconds_per_metadata':_timing(fn,x,scale,base),'graph_calls':80,'replays_per_window':50})
                    (args.output/'timings.json').write_text(json.dumps(timings,indent=2))
    (args.output/'result.json').write_text(json.dumps({'status':'PASS','exact_checks':len(checks),'graph_checks':len(graphs),'timings':timings,'scope':'Metadata-only synthetic inputs; no model change or promotion'},indent=2))


if __name__ == '__main__':
    main()
