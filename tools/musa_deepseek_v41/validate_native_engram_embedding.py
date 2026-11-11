# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Native original FP8 Engram lookup/sharding; collective simulated explicitly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401

from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime'] = native
    from xllm.python import initialize_runtime
    initialize_runtime()
    from xllm.python.models.deepseek_v41_engram import EngramEmbedding
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {'status': 'RUNNING', 'checks': [],
        'scope': 'native original device FP8 E4M3 + UE8M0/group32 lookup and whole-head TP slices; one GPU simulates shards and concatenation; no actual collective/network or fullmodel proof'}
    torch.manual_seed(111160)
    sizes = [17+2*i for i in range(24)]
    total = sum(sizes)
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    with torch.inference_mode():
        for dim in (32, 512):
            payload = (torch.arange(total*dim) % 256).byte().reshape(total, dim)
            payload[(payload & 127) == 127] = 0
            scales = torch.randint(120, 130, (total, dim//32), dtype=torch.uint8)
            full_reference = EngramEmbedding(total, dim, sizes, 1, 0, device=torch.device('musa'))
            full_reference.weight.copy_(payload.to('musa'))
            full_reference.weight_scale.copy_(scales.to('musa'))
            for count in (0, 1, 17, 129):
                ids = torch.stack([torch.randint(starts[i], starts[i]+sizes[i], (count,))
                                   for i in range(len(sizes))], dim=1).to('musa')
                if count:
                    ids[0, 0] = total+123
                if count > 1:
                    ids[1, 1] = -1
                expected = full_reference.lookup(ids)
                # Independent CPU decode, not the same GPU wrapper.
                cpu_ids = ids.cpu()
                owned = (cpu_ids >= 0) & (cpu_ids < total)
                selected = cpu_ids.clamp(0, total-1).flatten()
                values = payload[selected].contiguous().view(torch.float8_e4m3fn).float()
                factors = torch.pow(2., scales[selected].int()-127).repeat_interleave(32, dim=-1)
                cpu_reference = (values*factors).bfloat16().reshape(count, len(sizes), dim)
                cpu_reference = torch.where(owned[..., None], cpu_reference, 0)
                torch.testing.assert_close(expected.cpu(), cpu_reference, rtol=0, atol=0)
                for world in (1, 2, 4, 8):
                    pieces = []
                    for rank in range(world):
                        reference = EngramEmbedding(total, dim, sizes, world, rank, device=torch.device('musa'))
                        begin, rows = reference.vocab_start, reference.part_num_embeddings
                        reference.weight.copy_(payload[begin:begin+rows].to('musa'))
                        reference.weight_scale.copy_(scales[begin:begin+rows].to('musa'))
                        implementation = native.NativeV41EngramEmbedding(reference.weight,
                            reference.weight_scale, begin, reference.head_start, reference.part_n_hash_cols)
                        actual = implementation.local(ids)
                        torch.testing.assert_close(actual, reference.local_block(ids), rtol=0, atol=0)
                        if world == 1:
                            torch.testing.assert_close(implementation.lookup(ids), expected, rtol=0, atol=0)
                        pieces.append(actual)
                        result['checks'].append({'dim': dim, 'rows': count, 'tp': world,
                            'rank': rank, 'local_original_FP8_lookup': 'bitwise'})
                    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=0, atol=0)
                    result['checks'].append({'dim': dim, 'rows': count, 'tp': world,
                        'ordered_head_concatenation': 'bitwise; simulated, not MCCL'})
                args.output.write_text(json.dumps(result, indent=2))
    result.update(status='PASS', count=len(result['checks']))
    args.output.write_text(json.dumps(result, indent=2))
    logger.info('Native Engram embedding checks passed: %s', result['count'])


if __name__ == '__main__':
    main()
