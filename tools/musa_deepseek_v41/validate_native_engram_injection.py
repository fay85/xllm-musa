# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Native C++ Engram projection plus injection against an independent oracle."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401

from scripts.logger import logger
from tools.musa_deepseek_v41.validate_fp4 import _quant_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-test', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    source = args.reference_test.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    if source_sha != 'eb1dc12562aa02360b1796b3222db58aa83c2db116fad4225d40ed03ba3ba081':
        raise ValueError('Independent Engram oracle changed')
    reference_node = next(node for node in ast.parse(source).body
                          if isinstance(node, ast.FunctionDef) and node.name == 'reference')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[reference_node], type_ignores=[]),
                 str(args.reference_test), 'exec'), scope)
    reference = scope['reference']
    import musa_python_kernel_test_module as native
    torch.manual_seed(111151)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'RUNNING', 'oracle_sha256': source_sha,
              'scope': 'native projection+gate at original D5120; synthetic weights/embeddings; excludes hash/table/all-gather/full model',
              'checks': []}
    with torch.inference_mode():
        weight = torch.empty(25600, 6144, device='musa', dtype=torch.bfloat16).uniform_(-.01, .01)
        qw = torch.randn(4, 5120, device='musa', dtype=torch.bfloat16)
        kw = torch.randn_like(qw)
        for quantized in (False, True):
            layer = native.NativeV41EngramInjection(weight, qw, kw, quantized)
            for rows in (0, 1, 8, 17, 128, 256):
                # CPU tensors are test inputs and quantization oracle only.
                embedding_cpu = torch.randn(rows, 6144, dtype=torch.bfloat16)
                embedding = embedding_cpu.to('musa')
                hidden = torch.randn(rows, 4, 5120, device='musa', dtype=torch.bfloat16)
                qdq = _quant_oracle(embedding_cpu)[0].to('musa') if quantized else embedding
                kv = torch.nn.functional.linear(qdq, weight)
                mask = torch.arange(rows, device='musa') % 2 == 0
                expected = reference(hidden, kv, qw, kw)
                expected = torch.where(mask[:, None, None], expected, hidden)
                original_hidden = hidden.clone()
                original_embedding = embedding.clone()
                actual = layer.forward(hidden, embedding, mask)
                try:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                except AssertionError:
                    record.update(status='FAIL', failure={'rows': rows, 'quantized': quantized})
                    args.output.write_text(json.dumps(record, indent=2))
                    torch.save({'hidden': hidden.cpu(), 'embedding': embedding.cpu(),
                                'actual': actual.cpu(), 'expected': expected.cpu()},
                               args.output.with_suffix('.failure.pt'))
                    raise
                assert torch.equal(hidden, original_hidden)
                assert torch.equal(embedding, original_embedding)
                assert actual.device.type == 'musa'
                record['checks'].append({'rows': rows, 'original_fp8_activation': quantized,
                                         'exact': True, 'masked_passthrough': True,
                                         'input_ownership': True})
                args.output.write_text(json.dumps(record, indent=2))
        stream = torch.musa.Stream()
        with torch.musa.stream(stream):
            hidden = torch.randn(1, 4, 5120, device='musa', dtype=torch.bfloat16)
            embedding = torch.randn(1, 6144, device='musa', dtype=torch.bfloat16)
            layer = native.NativeV41EngramInjection(weight, qw, kw, False)
            actual = layer.forward(hidden, embedding, None)
            expected = reference(hidden, torch.nn.functional.linear(embedding, weight), qw, kw)
        stream.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        record['checks'].append({'non_default_stream': True, 'exact': True})
    record.update(status='PASS', count=len(record['checks']))
    args.output.write_text(json.dumps(record, indent=2))
    logger.info('Native Engram injection checks passed: %s', record['count'])


if __name__ == '__main__':
    main()
