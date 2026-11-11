# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Compare native C++ V4.1 mHC preparation with the Python fused GPU path."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401

from scripts.logger import logger
from xllm.python import initialize_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose one verified idle GPU')
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime'] = native
    initialize_runtime()
    from xllm.python.models.deepseek_v41 import DeepseekV41Config, DeepseekV41HyperConnection
    for part in ('PRE', 'POST', 'METADATA'):
        os.environ[f'XLLM_DSV41_MUSA_HC_{part}'] = 'v4_fused'
    torch.manual_seed(111012)
    checks = []

    def run(name: str, residual: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor) -> None:
        hidden = residual.shape[-1]
        cfg = DeepseekV41Config(hidden_size=hidden, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
        py = DeepseekV41HyperConnection(cfg, torch.bfloat16, torch.device('musa:0'))
        cpp = native.NativeV41HyperConnection(fn, scale, base, 20, cfg.rms_norm_eps, cfg.hc_eps)
        carried = torch.rand(residual.shape[0], 4, device='musa', dtype=torch.float32) + 0.125
        next_pre, post, comb = py.hc_mixes(residual, fn, scale, base)
        expected = (py.hc_pre(residual, carried), next_pre, post, comb)
        actual = cpp.prepare(residual, carried)
        value = torch.randn(residual.shape[0], hidden, device='musa', dtype=torch.bfloat16)
        expected = (*expected, py.hc_post(value, residual, post, comb))
        actual = (*actual, cpp.combine(value, residual, post, comb))
        for part, a, z in zip(('input', 'next_pre', 'post', 'comb', 'output'), actual, expected):
            assert a.device.type == 'musa'
            tol = {'rtol': 1/128, 'atol': 1e-3} if a.dtype == torch.bfloat16 else {'rtol': 2e-5, 'atol': 1e-6}
            torch.testing.assert_close(a.cpu(), z.cpu(), **tol)
            error = (a.float() - z.float()).abs().max().item() if a.numel() else 0.0
            checks.append({'case': name, 'part': part, 'max_abs_error': error, 'bitwise_equal': torch.equal(a, z), 'status': 'PASS', **tol})

    with torch.inference_mode():
        for tokens in (0, 1, 17, 129, 256):
            for hidden in (64, 5120):
                residual = torch.randn(tokens, 4, hidden * 2, device='musa', dtype=torch.bfloat16)[..., ::2]
                fn = torch.randn(24, hidden * 4, device='musa') * 0.02
                run(f'random_T{tokens}_H{hidden}', residual, fn, torch.tensor([0.5,1.,1.5],device='musa'), torch.randn(24,device='musa') * .1)
        if args.checkpoint:
            from safetensors import safe_open
            index = json.loads((args.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
            for layer in (0,1,14,39):
                for part in ('attn','ffn'):
                    weights = []
                    for suffix in ('fn','scale','base'):
                        key = f'layers.{layer}.hc_{part}_{suffix}'
                        with safe_open(args.checkpoint/index[key], framework='pt', device='cpu') as f:
                            weights.append(f.get_tensor(key).to('musa'))
                    run(f'checkpoint_layer{layer}_{part}', torch.randn(17,4,5120,device='musa',dtype=torch.bfloat16), *weights)
        stream = torch.musa.Stream()
        with torch.musa.stream(stream):
            run('nondefault_stream',torch.randn(3,4,64,device='musa',dtype=torch.bfloat16), torch.randn(24,256,device='musa')*.02, torch.ones(3,device='musa'),torch.zeros(24,device='musa'))
        stream.synchronize()
    result={'status':'PASS','checks':checks,'scope':'Native C++ mHC sublayer vs Python fused on one GPU; not native full model or P-D numerical acceptance'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    logger.info('Native C++ mHC checks passed: %s',len(checks))


if __name__ == '__main__':
    main()
