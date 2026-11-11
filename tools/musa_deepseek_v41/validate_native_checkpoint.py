# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Native original-FP8 loader and unchanged packed-FP4 EP slices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401

from scripts.logger import logger


def _bytes(value: torch.Tensor) -> bytes:
    return value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--real-dense', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime'] = native
    from xllm.python import initialize_runtime
    initialize_runtime()
    from xllm.python.models.deepseek_v41 import dequant_fp8_block
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {'status': 'RUNNING', 'checks': [],
        'scope': 'native CPU offline FP8/UE8M0 dequant then TP slicing and GPU copy; exact packed-FP4 EP slices; one original1017 mapped matrix; not fullmodel loading'}
    torch.manual_seed(111159)
    with torch.inference_mode():
        for n, k in [(17, 33), (32, 32), (64, 128)]:
            payload = (torch.arange(n*k) % 256).byte().reshape(n, k)
            for exponent in (0, 1, 120, 127, 254):
                scales = torch.full((-(n//-32), -(k//-32)), exponent, dtype=torch.uint8)
                reference = dequant_fp8_block(payload, scales)
                for typed in (False, True):
                    weight = payload.view(torch.float8_e4m3fn) if typed else payload
                    value = native.native_v41_checkpoint_dequant_fp8(weight, scales)
                    torch.testing.assert_close(value, reference, rtol=0, atol=0, equal_nan=True)
                    finite = ~torch.isnan(reference)
                    assert torch.equal(value.view(torch.int16)[finite], reference.view(torch.int16)[finite])
                    result['checks'].append({'n': n, 'k': k, 'ue8m0_byte': exponent,
                        'float8_storage': typed, 'finite_payloads': 'bitwise incl signed zero', 'nan': 'equal_nan'})
        tensors = {'embed.weight': torch.randn(64, 64).bfloat16(),
                   'dense.weight': torch.randn(64, 128).to(torch.float8_e4m3fn),
                   'dense.scale': torch.randint(120, 130, (2, 4), dtype=torch.uint8),
                   'norm': torch.randn(64), 'expert_count': torch.tensor(8)}
        for index in range(8):
            tensors[f'experts.{index}.w1.weight'] = torch.randint(0, 256, (32, 32), dtype=torch.uint8)
            tensors[f'experts.{index}.w1.scale'] = torch.randint(120, 130, (32, 2), dtype=torch.uint8)
        source_sha = {name: hashlib.sha256(_bytes(value)).hexdigest() for name, value in tensors.items()}
        expected = dequant_fp8_block(tensors['dense.weight'], tensors['dense.scale'])
        for world in (1, 2, 4):
            for rank in range(world):
                loaded = native.native_v41_checkpoint_fixture_load(tensors, rank, world)
                for key, reference in {
                    'embedding': tensors['embed.weight'].chunk(world, dim=1)[rank],
                    'dense': expected, 'column': expected.chunk(world, dim=0)[rank],
                    'row': expected.chunk(world, dim=1)[rank], 'norm': tensors['norm'],
                    'expert': torch.stack([tensors[f'experts.{i}.w1.weight'] for i in range(rank*(8//world), (rank+1)*(8//world))]),
                    'scale': torch.stack([tensors[f'experts.{i}.w1.scale'] for i in range(rank*(8//world), (rank+1)*(8//world))]),
                }.items():
                    assert loaded[key].device.type == 'musa'
                    torch.testing.assert_close(loaded[key].cpu(), reference, rtol=0, atol=0)
                result['checks'].append({'world': world, 'rank': rank,
                    'dense_TP_after_whole_block_dequant_and_packed_EP': 'bitwise'})
        assert source_sha == {name: hashlib.sha256(_bytes(value)).hexdigest() for name, value in tensors.items()}
        # The exact original tensor bytes are in a new two-tensor Safetensors
        # envelope; parse via the same native mmap reader as serving.
        manifest = json.loads((args.real_dense/'manifest.json').read_text())
        path = args.real_dense/'selected_original_tensors.safetensors'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['asset_sha256']
        import struct
        raw = path.read_bytes()
        header_size = struct.unpack('<Q', raw[:8])[0]
        metadata = json.loads(raw[8:8+header_size])
        decoded = {}
        for suffix in ('weight', 'scale'):
            name = manifest['prefix']+'.'+suffix
            entry = metadata[name]
            a, b = entry['data_offsets']
            payload = bytearray(raw[8+header_size+a:8+header_size+b])
            assert hashlib.sha256(payload).hexdigest() == manifest['tensors'][name]['sha256']
            value = torch.frombuffer(payload, dtype=torch.uint8).reshape(entry['shape'])
            decoded[suffix] = value
        reference = dequant_fp8_block(decoded['weight'], decoded['scale'])
        actual = native.native_v41_checkpoint_mapped_dense(str(path), manifest['prefix'])
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
        result['checks'].append({'original_revision': manifest['model_revision'],
            'matrix': manifest['prefix'], 'shape': list(actual.shape),
            'asset_sha256': manifest['asset_sha256'], 'native_mmap_reader_and_decode': 'bitwise'})
    result.update(status='PASS', count=len(result['checks']))
    args.output.write_text(json.dumps(result, indent=2))
    logger.info('Native checkpoint checks passed: %s', result['count'])


if __name__ == '__main__':
    main()
