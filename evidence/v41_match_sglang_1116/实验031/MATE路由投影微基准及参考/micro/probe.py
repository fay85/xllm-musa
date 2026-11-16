from pathlib import Path
import gc
import hashlib
import json
import statistics
import torch
import torch_musa
import mate
from mate.deep_gemm import bf16_gemm_nt

root = Path('/expt/diagnostics/e31_gate_bf16_fp32_v1')
torch.set_num_threads(2)
torch.manual_seed(3101)
torch.musa.set_device(0)
checks, graphs, timings = [], [], []
weight = torch.randn(384, 5120, device='musa', dtype=torch.bfloat16) / 64
weight_fp32 = weight.float()

def save():
    (root / 'partial.json').write_text(json.dumps({'checks': checks, 'graphs': graphs, 'timings': timings}, indent=2))

def reference(x):
    return torch.nn.functional.linear(x.float(), weight_fp32)

def candidate(x):
    output = torch.empty((x.shape[0], 384), dtype=torch.float32, device=x.device)
    bf16_gemm_nt(x.contiguous(), weight, output)
    return output

def compare(actual, expected, case):
    torch.musa.synchronize()
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all().item()
    delta = (actual - expected).abs()
    rms = ((delta.square().mean() / expected.square().mean().clamp_min(1e-30)).sqrt()).item()
    assert rms < 1e-4, (case, rms, delta.max().item())
    return dict(case, relative_rms=rms, max_abs=delta.max().item(), top8_same=torch.equal(actual.topk(8).indices, expected.topk(8).indices))

def timing(fn, x):
    for _ in range(3):
        fn(x)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        outputs = [fn(x) for _ in range(40)]
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(5):
        begin = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        begin.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / (40 * 20))
    del graph, outputs
    gc.collect()
    return {'microseconds': samples, 'median_us': statistics.median(samples), 'calls_per_graph': 40, 'replays_per_window': 20}

with torch.inference_mode():
    alternate = torch.musa.Stream()
    for use_alternate in [False, True]:
        stream = alternate if use_alternate else torch.musa.current_stream()
        stream.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(stream):
            for rows in [1, 4, 6]:
                for magnitude in [0., 1e-5, 1.]:
                    case = {'rows': rows, 'magnitude': magnitude, 'alternate_stream': use_alternate}
                    (root / 'current_case.json').write_text(json.dumps(case))
                    x = torch.randn(rows, 5120, dtype=torch.bfloat16, device='musa') * magnitude
                    checks.append(compare(candidate(x), reference(x), case))
                    save()
    with torch.musa.stream(alternate):
        for rows in [1, 4, 6]:
            x = torch.randn(rows, 5120, dtype=torch.bfloat16, device='musa')
            for _ in range(3):
                candidate(x)
            alternate.synchronize()
            graph = torch.musa.MUSAGraph()
            with torch.musa.graph(graph):
                output = candidate(x)
            for update in range(3):
                x.copy_(torch.randn_like(x))
                graph.replay()
                graphs.append(compare(output, reference(x), {'rows': rows, 'update': update}))
                save()
            del graph, output
            gc.collect()
            for label, fn in [('off', reference), ('on', candidate), ('reverse_off', reference)]:
                timings.append(dict(rows=rows, label=label, **timing(fn, x)))
                save()

result = {'status': 'PASS', 'checks': checks, 'graphs': graphs, 'timings': timings,
          'mate_file': mate.__file__, 'mate_version': getattr(mate, '__version__', 'unknown'),
          'scope': 'BF16 operands, FP32 output; standalone MATE projection, not native C++ integration or model performance'}
(root / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'status': result['status'], 'checks': len(checks), 'graphs': len(graphs), 'timings': timings}), flush=True)
