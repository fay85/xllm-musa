# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import json, datetime, argparse
import torch, torch_musa
import v41_q4_projection_test_module as ext
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
r = args.output.parent
torch.manual_seed(1805)
torch.set_num_threads(2)
checks = []
graphs = []

def _error(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return ((a - b).square().mean() / b.square().mean().clamp_min(1e-12)).sqrt().item()
for n in [288, 512]:
    w = torch.randn(n, 5120, device='musa', dtype=torch.bfloat16) / 64
    for quant in [False, True]:
        off = ext.Dense(w, quant, False)
        on = ext.Dense(w, quant, True)
        for m in [0, 1, 4, 6, 9]:
            for nondefault in [False, True]:
                s = torch.musa.Stream() if nondefault else torch.musa.current_stream()
                s.wait_stream(torch.musa.current_stream())
                with torch.musa.stream(s):
                    x = torch.randn(m, 5120, device='musa', dtype=torch.bfloat16)
                    a = off.forward(x)
                    b = on.forward(x)
                s.synchronize()
                er = _error(a, b) if m else 0
                assert er < 0.005, (n, quant, m, er)
                checks.append({'kind': 'dense', 'n': n, 'qdq': quant, 'm': m, 'stream': nondefault, 'rms': er})
        s = torch.musa.Stream()
        s.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(s):
            x = torch.randn(4, 5120, device='musa', dtype=torch.bfloat16)
            on.forward(x)
            s.synchronize()
            g = torch.musa.MUSAGraph()
            with torch.musa.graph(g, stream=s):
                y = on.forward(x)
            for i in range(3):
                x.copy_(torch.randn_like(x))
                g.replay()
                ref = off.forward(x)
                s.synchronize()
                er = _error(y, ref)
                assert er < 0.005
                graphs.append({'kind': 'dense', 'n': n, 'qdq': quant, 'repeat': i, 'rms': er})
gate = torch.randn(288, 5120, device='musa', dtype=torch.bfloat16) / 64
up = torch.randn_like(gate) / 64
down = torch.randn(5120, 288, device='musa', dtype=torch.bfloat16) / 16
for quant in [False, True]:
    off = ext.Shared(gate, up, down, 10.0, quant, True, False)
    on = ext.Shared(gate, up, down, 10.0, quant, True, True)
    for m in [0, 1, 4, 6, 9]:
        for nondefault in [False, True]:
            s = torch.musa.Stream() if nondefault else torch.musa.current_stream()
            s.wait_stream(torch.musa.current_stream())
            with torch.musa.stream(s):
                x = torch.randn(m, 5120, device='musa', dtype=torch.bfloat16)
                a = off.forward(x)
                b = on.forward(x)
            s.synchronize()
            er = _error(a, b) if m else 0
            assert er < 0.01, (quant, m, er)
            checks.append({'kind': 'shared', 'qdq': quant, 'm': m, 'stream': nondefault, 'rms': er})
    s = torch.musa.Stream()
    s.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(s):
        x = torch.randn(4, 5120, device='musa', dtype=torch.bfloat16)
        on.forward(x)
        s.synchronize()
        g = torch.musa.MUSAGraph()
        with torch.musa.graph(g, stream=s):
            y = on.forward(x)
        for i in range(3):
            x.copy_(torch.randn_like(x))
            g.replay()
            ref = off.forward(x)
            s.synchronize()
            er = _error(y, ref)
            assert er < 0.01
            graphs.append({'kind': 'shared', 'qdq': quant, 'repeat': i, 'rms': er})
x = torch.randn(4, 5120, device='musa', dtype=torch.bfloat16)
w = torch.randn(512, 5120, device='musa', dtype=torch.bfloat16)
assert ext.supported(x, w)
assert not ext.supported(x[:1], w)
assert not ext.supported(x[:, ::2], w)
assert not ext.supported(x, w[:256])
assert not ext.supported(x.cpu(), w.cpu())
args.output.write_text(json.dumps({'status': 'PASS', 'checks': checks, 'graph_replays': graphs, 'scope': 'Production C++ Dense/Shared APIs, all FP8 QDQ retained, no whole model precision claim', 'at': datetime.datetime.now().astimezone().isoformat()}, indent=2))
print(json.dumps({'status': 'PASS', 'checks': len(checks), 'graph_replays': len(graphs)}))
