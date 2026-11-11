# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Independent CPU-oracle checks for V4.1 attention FP4 quantization.

This validates a primitive only. Cache allocation/encoding and P/D transport
must be adapted together before the attention backend can select this path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401

from scripts.logger import logger


def _oracle(
    x: torch.Tensor, group_size: int, e4m3_scale: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = x.cpu().float().reshape(-1, group_size)
    maximum = value.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        raw = maximum.clamp_min(6 * 2.**-9) * (1. / 6.)
        scale_values = raw.to(torch.float8_e4m3fn)
        scale = scale_values.float()
        scale_codes = scale_values.view(torch.uint8)
    else:
        raw = maximum.clamp_min(6 * 2.**-126) * (1. / 6.)
        exponent = raw.double().log2().ceil()
        scale = torch.pow(2., exponent).float()
        scale_codes = (exponent + 127).to(torch.uint8)
    normalized = (value / scale).clamp(-6, 6)
    magnitude = normalized.abs()
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    distances = (magnitude.unsqueeze(-1) - levels).abs()
    minimum = distances.amin(-1, keepdim=True)
    # Prefer even codes only when they are among the nearest values.
    codes = torch.arange(8)
    priority = codes + (codes % 2) * 8
    nearest = torch.where(distances == minimum, priority, 100)
    selected = nearest.argmin(-1).to(torch.uint8)
    signed = selected | (torch.signbit(normalized).to(torch.uint8) << 3)
    dequant = levels[selected.long()] * torch.where(
        torch.signbit(normalized), -1., 1.) * scale
    packed = signed[:, ::2] | (signed[:, 1::2] << 4)
    return (
        dequant.reshape_as(x).bfloat16(),
        packed.reshape(x.shape[0], x.shape[1] // 2),
        scale_codes.reshape(x.shape[0], x.shape[1] // group_size),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    torch.manual_seed(111114)
    result = {'status': 'RUNNING', 'checks': [], 'scope': 'quantization primitive only; cache and full model not integrated'}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def _save() -> None:
        args.output.write_text(json.dumps(result, indent=2))

    def _check(name: str, x: torch.Tensor, group: int, fp8: bool) -> None:
        outputs = native.musa_v41_attention_fp4_quant(x.to('musa'), group, fp8)
        expected = _oracle(x, group, fp8)
        for index, (actual, reference) in enumerate(zip(outputs, expected)):
            # Include negative-zero BF16 storage in the exact check.
            dtype = torch.int16 if index == 0 else torch.uint8
            try:
                torch.testing.assert_close(actual.cpu().view(dtype), reference.view(dtype), rtol=0, atol=0)
            except AssertionError:
                actual_cpu = [t.cpu() for t in outputs]
                torch.save({'input': x.cpu(), 'actual': actual_cpu, 'expected': expected,
                            'group': group, 'e4m3_scale': fp8}, args.output.with_suffix('.failure.pt'))
                mismatch = (actual.cpu().view(dtype) != reference.view(dtype)).nonzero()
                examples = []
                for point in mismatch[:8]:
                    row, column = point.tolist()
                    feature = column if index == 0 else column * (2 if index == 1 else group)
                    group_index = feature // group
                    scale_code = actual_cpu[2][row, group_index].reshape(1)
                    scale_value = (scale_code.view(torch.float8_e4m3fn).float()[0]
                                   if fp8 else 2.**(int(scale_code[0])-127))
                    examples.append({'tensor_index': index, 'index': [row, column],
                                     'input': float(x.cpu()[row, feature]),
                                     'scale': float(scale_value),
                                     'normalized_fp32': float(x.cpu()[row, feature].float()/scale_value),
                                     'normalized_fp64': float(x.cpu()[row, feature].double()/float(scale_value)),
                                     'actual_qdq': float(actual_cpu[0][row, feature]),
                                     'expected_qdq': float(expected[0][row, feature]),
                                     'actual_scale_code': int(actual_cpu[2][row, group_index]),
                                     'expected_scale_code': int(expected[2][row, group_index])})
                args.output.with_suffix('.diagnostic.json').write_text(json.dumps(examples, indent=2))
                raise
        result['checks'].append({'name': name, 'group': group, 'e4m3_scale': fp8, 'status': 'PASS', 'exact_qdq_packed_scale_bytes': True})
        _save()

    try:
        with torch.inference_mode():
            for group, fp8 in [(32, False), (16, True), (16, False), (32, True)]:
                for rows in (0, 1, 3, 17, 256):
                    for width in (32, 128, 512, 576):
                        x = torch.randn(rows, width, dtype=torch.bfloat16)
                        factor = torch.pow(2., torch.randint(-16, 9, (rows, width // group)).float())
                        x = (x.float() * factor.repeat_interleave(group, -1)).bfloat16()
                        _check(f'random-{rows}-{width}', x, group, fp8)
                _check('all-zero', torch.zeros(3, 128, dtype=torch.bfloat16), group, fp8)
                for sign in (1., -1.):
                    # Anchor amax=6 to force scale=1. Exercise each exact
                    # midpoint and its adjacent BF16 values independently.
                    mid = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.bfloat16)
                    for direction in (-float('inf'), 0., float('inf')):
                        vals = mid if direction == 0 else torch.nextafter(mid, torch.full_like(mid, direction))
                        x = torch.zeros(7, group, dtype=torch.bfloat16)
                        x[:, 0] = vals * sign
                        x[:, 1] = 6.
                        _check(f'midpoints-{sign}-{direction}', x, group, fp8)
                if fp8:
                    for scale in (3., 5., 60.):
                        midpoint = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
                        x = torch.zeros(14, group, dtype=torch.bfloat16)
                        x[:, 0] = torch.cat((midpoint, -midpoint)) * scale
                        x[:, 1] = 6. * scale
                        _check(f'non-power2-scale-midpoints-{scale}', x, group, fp8)
                # Force a non-default producer and consumer stream.
                stream = torch.musa.Stream()
                x = torch.randn(17, 128, dtype=torch.bfloat16, device='musa')
                stream.wait_stream(torch.musa.current_stream())
                with torch.musa.stream(stream):
                    outputs = native.musa_v41_attention_fp4_quant((x + 1).contiguous(), group, fp8)
                stream.synchronize()
                expected = _oracle((x + 1).cpu(), group, fp8)
                for actual, reference in zip(outputs, expected):
                    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
                result['checks'].append({'name': 'nondefault-stream', 'group': group, 'e4m3_scale': fp8, 'status': 'PASS'})
                _save()
        result.update(status='PASS', count=len(result['checks']))
        _save()
        logger.info('Attention FP4 primitive checks passed: %s', result['count'])
    except Exception:
        result['status'] = 'FAIL'
        _save()
        raise


if __name__ == '__main__':
    main()
