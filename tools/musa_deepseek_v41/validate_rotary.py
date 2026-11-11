# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Validate the real MUSA rotary binding without a full serving benchmark."""
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
from xllm.python.kernels_musa.rotary import select_rotary


def _reference(x: torch.Tensor, positions: torch.Tensor, table: torch.Tensor, inverse: bool) -> torch.Tensor:
    half = table.shape[-1] // 2
    rows = table[positions].double()
    cos, sin = rows[:, :half], rows[:, half:]
    if inverse:
        sin = -sin
    for _ in range(x.ndim - 2):
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    tail = x[..., -2 * half:].double()
    even, odd = tail[..., ::2], tail[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)
    return torch.cat((x[..., :-2 * half], rotated), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module
    sys.modules['xllm_runtime'] = musa_python_kernel_test_module
    os.environ['XLLM_DSV41_MUSA_ROPE'] = 'v4_fused'
    select_rotary.cache_clear()
    op = select_rotary()
    torch.manual_seed(111011)
    checks = []
    with torch.inference_mode():
        for tokens in (0, 1, 17, 256, 2048):
            for shape in ((tokens, 512), (tokens, 8, 512), (tokens, 2, 4, 512)):
                for dtype in (torch.bfloat16, torch.float32):
                    # Non-contiguous hidden tests the real packing in the wrapper.
                    x = torch.randn(*shape[:-1], 1024, dtype=dtype, device='musa')[..., ::2]
                    positions = torch.arange(tokens, device='musa') % 37
                    angle = torch.randn(37, 32)
                    table = torch.cat((angle.cos(), angle.sin()), dim=-1).to('musa')
                    for inverse in (False, True):
                        actual = op(x, positions, table, inverse)
                        expected = _reference(x.cpu(), positions.cpu(), table.cpu(), inverse)
                        assert actual.device.type == 'musa'
                        tolerance = {'rtol': 1 / 128, 'atol': 1e-3} if dtype == torch.bfloat16 else {'rtol': 2e-5, 'atol': 1e-6}
                        torch.testing.assert_close(actual.cpu(), expected, **tolerance)
                        torch.testing.assert_close(actual[..., :-64].cpu(), x[..., :-64].cpu(), rtol=0, atol=0)
                        checks.append({'shape': shape, 'dtype': str(dtype), 'inverse': inverse, 'status': 'PASS', **tolerance})
        stream = torch.musa.Stream()
        with torch.musa.stream(stream):
            x = torch.randn(17, 4, 512, device='musa', dtype=torch.bfloat16)
            pos = torch.arange(17, device='musa')
            angle = torch.randn(17, 32, device='musa')
            table = torch.cat((angle.cos(), angle.sin()), dim=-1)
            actual = op(x + 1, pos, table, True)
        stream.synchronize()
        torch.testing.assert_close(actual.cpu(), _reference((x + 1).cpu(), pos.cpu(), table.cpu(), True), rtol=1 / 128, atol=1e-3)
        checks.append({'nondefault_stream': True, 'status': 'PASS'})
    result = {'status': 'PASS', 'checks': checks, 'scope': '61 GPU primitive checks against CPU FP64 oracle; full-model and C++ route verification still required'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    logger.info('RoPE numerical checks passed: %s', len(checks))


if __name__ == '__main__':
    main()
