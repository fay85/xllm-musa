# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Native C++ V4.1 attention projection with independent QDQ/RoPE references."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
import torch.nn.functional as F

from scripts.logger import logger
from tools.musa_deepseek_v41.validate_fp4 import _quant_oracle


def _quant(value: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return value
    prepared = _quant_oracle(value.reshape(-1, value.shape[-1]).cpu())[0]
    return prepared.to(value.device).reshape_as(value)


def _norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    f = value.float()
    return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + epsilon) * weight.float()).to(value.dtype)


def _rope(value: torch.Tensor, positions: torch.Tensor,
          table: torch.Tensor, inverse: bool) -> torch.Tensor:
    width = table.shape[1]
    rows = table[positions.long()]
    cosine, sine = rows.chunk(2, dim=-1)
    shape = (len(value),) + (1,) * (value.ndim - 2) + (width // 2,)
    cosine, sine = cosine.reshape(shape), sine.reshape(shape)
    if inverse:
        sine = -sine
    work_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    pairs = value[..., -width:].to(work_dtype).unflatten(-1, (-1, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even*cosine-odd*sine, even*sine+odd*cosine), dim=-1).flatten(-2)
    return torch.cat((value[..., :-width].to(work_dtype), rotated), dim=-1).to(value.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime'] = native
    os.environ['XLLM_DSV41_MUSA_ROPE'] = 'v4_fused'
    from xllm.python.kernels_musa.rotary import select_rotary
    select_rotary.cache_clear()
    fused_rope = select_rotary()
    torch.manual_seed(111152)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'RUNNING', 'scope': 'native Q/KV/norm/RoPE and grouped O projections; synthetic weights; excludes cache/indexer/TP collective/full model',
              'checks': []}
    with torch.inference_mode():
        for hidden_dim, latent, heads, head_dim, groups, out_rank, rope_width in [
                (64, 32, 4, 32, 2, 16, 16), (5120, 1280, 8, 512, 2, 512, 64)]:
            shapes = [(latent, hidden_dim), (latent,), (heads*head_dim, latent),
                      (head_dim, hidden_dim), (head_dim,),
                      (groups*out_rank, heads*head_dim//groups), (hidden_dim, groups*out_rank)]
            weights = [torch.randn(shape, device='musa', dtype=torch.bfloat16)*.01 for shape in shapes]
            weights[1].add_(1); weights[4].add_(1)
            angles = torch.arange(600, device='musa')[:, None].float() * torch.linspace(.01, .5, rope_width//2, device='musa')
            table = torch.cat((angles.cos(), angles.sin()), dim=-1)
            for enabled in (False, True):
                layer = native.NativeV41AttentionProjection(weights, heads, groups, out_rank, 1e-20, enabled)
                for count in (0, 1, 8, 17, 129, 256):
                    positions = torch.arange(count, device='musa', dtype=torch.int64)*2+1
                    hidden = torch.randn(count, hidden_dim, device='musa', dtype=torch.bfloat16)
                    saved = hidden.clone()
                    prepared = _quant(hidden, enabled)
                    qr = _norm(F.linear(prepared, weights[0]), weights[1], 1e-20)
                    key = _norm(F.linear(prepared, weights[3]), weights[4], 1e-20)
                    query = F.linear(_quant(qr, enabled), weights[2]).reshape(count, heads, head_dim)
                    expected = (fused_rope(query, positions, table, False),
                                _quant(fused_rope(key, positions, table, False), enabled), qr)
                    actual = layer.prepare(hidden, positions, table)
                    for output, reference in zip(actual, expected):
                        torch.testing.assert_close(output, reference, rtol=0, atol=0)
                    # Separate Torch ops are a different RoPE rounding schedule.
                    # Keep the difference visible; language parity uses the SAME
                    # fused kernel, plus the existing independent FP64 bound.
                    separate = _rope(query, positions, table, False)
                    different = (actual[0] != separate)
                    if bool(different.any()):
                        cpu_reference = _rope(query.cpu().double(), positions.cpu(), table.cpu().double(), False).bfloat16()
                        torch.testing.assert_close(actual[0].cpu(), cpu_reference, rtol=1/128, atol=1e-3)
                        rows_different = different.nonzero()
                        comparison = {'rows': count, 'hidden': hidden_dim, 'original_qdq': enabled,
                                      'different': int(different.sum()),
                                      'max_abs': float((actual[0].float()-separate.float()).abs().max()),
                                      'cpp_equals_python_fused': True,
                                      'fp64_rtol': 1/128, 'fp64_atol': 1e-3}
                        record.setdefault('separate_torch_rope_diagnostics', []).append(comparison)
                        torch.save({'query_before_rope': query.cpu(), 'positions': positions.cpu(),
                                    'table': table.cpu(), 'cpp': actual[0].cpu(),
                                    'python_fused': expected[0].cpu(), 'separate_torch': separate.cpu(),
                                    'fp64_rounded': cpu_reference, 'different_indices': rows_different.cpu()},
                                   args.output.with_name('rope_rounding_'+str(hidden_dim)+'_'+str(count)+'_'+str(enabled)+'.pt'))
                    attention = torch.randn(count, heads, head_dim, device='musa', dtype=torch.bfloat16)
                    saved_attention = attention.clone()
                    value = _quant(fused_rope(attention, positions, table, True), enabled).reshape(count, groups, heads*head_dim//groups)
                    low = torch.einsum('tgd,grd->tgr', value, weights[5].reshape(groups, out_rank, -1))
                    output = F.linear(_quant(low.reshape(count, groups*out_rank), enabled), weights[6])
                    projected = layer.project_output(attention, positions, table)
                    try:
                        torch.testing.assert_close(projected, output, rtol=0, atol=0)
                    except AssertionError:
                        record.update(status='FAIL', failure={'hidden': hidden_dim, 'rows': count, 'quantized': enabled})
                        args.output.write_text(json.dumps(record, indent=2))
                        torch.save({'actual': projected.cpu(), 'expected': output.cpu()}, args.output.with_suffix('.failure.pt'))
                        raise
                    assert torch.equal(hidden, saved) and torch.equal(attention, saved_attention)
                    record['checks'].append({'hidden': hidden_dim, 'latent': latent, 'heads': heads,
                                             'head_dim': head_dim, 'rows': count, 'original_qdq': enabled,
                                             'query_kv_latent_output': 'bitwise', 'ownership': True})
                    args.output.write_text(json.dumps(record, indent=2))
        stream = torch.musa.Stream()
        with torch.musa.stream(stream):
            positions = torch.tensor([127], device='musa')
            hidden = torch.randn(1, hidden_dim, device='musa', dtype=torch.bfloat16)
            q, kv, qr = layer.prepare(hidden, positions, table)
            result = layer.project_output(q, positions, table)
        stream.synchronize()
        assert torch.isfinite(result).all() and result.device.type == 'musa'
        record['checks'].append({'non_default_stream': True, 'device': 'musa'})
    record.update(status='PASS', count=len(record['checks']))
    args.output.write_text(json.dumps(record, indent=2))
    logger.info('Native attention projection checks passed: %s', record['count'])


if __name__ == '__main__':
    main()
