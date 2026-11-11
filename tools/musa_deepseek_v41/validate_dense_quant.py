# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Original dense input QDQ and parameter-preserving GPU integration checks."""
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
from tools.musa_deepseek_v41.validate_fp4 import _quant_oracle
from xllm.python.kernels_musa.dense_quant import select_original_fp8_dense


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime'] = native
    os.environ['XLLM_DSV41_MUSA_DENSE_QUANT'] = 'original_qdq'
    torch.manual_seed(111120)
    record = {'status': 'RUNNING', 'scope': 'BF16 dense GEMM with original activation QDQ; not native FP8 GEMM/full model', 'checks': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for k, n in [(32, 19), (128, 70), (5120, 512), (512, 5120)]:
            original = torch.nn.Linear(k, n, bias=False, device='musa', dtype=torch.bfloat16)
            original.weight.uniform_(-.01, .01)
            module = select_original_fp8_dense(original)
            assert module.weight is original.weight
            for shape in [(0, k), (1, k), (17, k), (2, 3, k)]:
                x = torch.randn(shape, dtype=torch.bfloat16)
                qdq, _ = _quant_oracle(x.reshape(-1, k))
                expected = torch.nn.functional.linear(qdq.reshape_as(x).to('musa'), original.weight)
                actual = module(x.to('musa'))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert actual.device.type == 'musa' and actual.shape == (*shape[:-1], n)
                if x.numel():
                    assert not torch.equal(x, qdq.reshape_as(x)), 'Quantization test must exercise changed input values'
                record['checks'].append({'k': k, 'n': n, 'input_shape': shape, 'status': 'PASS', 'oracle_qdq_then_same_gpu_gemm': 'bitwise'})
                args.output.write_text(json.dumps(record, indent=2))
        for mode in ['wrong-mode']:
            os.environ['XLLM_DSV41_MUSA_DENSE_QUANT'] = mode
            try:
                select_original_fp8_dense(original)
            except ValueError:
                record['checks'].append({'rejected_mode': mode, 'status': 'PASS'})
            else:
                raise AssertionError('Invalid mode was accepted')
    record.update(status='PASS', count=len(record['checks']))
    args.output.write_text(json.dumps(record, indent=2))
    logger.info('Original FP8 dense QDQ checks passed: %s', record['count'])


if __name__ == '__main__':
    main()
