# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Paired A/B/A Engram gate microbenchmark, never a serving performance claim."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
from scripts.logger import logger


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-test',type=Path,required=True)
    args=parser.parse_args()
    if torch.musa.device_count()!=1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    source=args.reference_test.read_bytes()
    assert hashlib.sha256(source).hexdigest()=='eb1dc12562aa02360b1796b3222db58aa83c2db116fad4225d40ed03ba3ba081'
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='reference')
    scope={'torch':torch};exec(compile(ast.Module(body=[function],type_ignores=[]),str(args.reference_test),'exec'),scope)
    reference=scope['reference']
    import musa_python_kernel_test_module as native
    torch.manual_seed(111150)
    result={'status':'RUNNING','scope':'gate only; synthetic BF16 hidden/projectedKV/weights; no lookup/projection/MCCL/PD/server',
            'warmup_per_phase':20,'iterations_per_phase':100,'order':'reference_A/fused_B/reference_A2',
            'gpu_time':'MUSA events over100 calls, including producer/host scheduling gaps',
            'host_time':'wall time of same100 calls including final synchronize','cases':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for rows in (1,8,128,256):
            x=torch.randn(rows,4,5120,device='musa',dtype=torch.bfloat16)
            kv=torch.randn(rows,25600,device='musa',dtype=torch.bfloat16)
            qw=torch.randn(4,5120,device='musa',dtype=torch.bfloat16);kw=torch.randn_like(qw)
            torch.testing.assert_close(native.musa_v41_engram_gate(x,kv,qw,kw,None),reference(x,kv,qw,kw),rtol=0,atol=0)
            measurements=[]
            for name,operation in [('reference_A',reference),('fused_B',native.musa_v41_engram_gate),('reference_A2',reference)]:
                for _ in range(20):operation(x,kv,qw,kw)
                torch.musa.synchronize()
                start=torch.musa.Event(enable_timing=True);end=torch.musa.Event(enable_timing=True)
                before=time.perf_counter();start.record()
                for _ in range(100):operation(x,kv,qw,kw)
                end.record();torch.musa.synchronize();elapsed=time.perf_counter()-before
                measurements.append({'phase':name,'event_ms_per_call':start.elapsed_time(end)/100,'wall_ms_per_call':elapsed*1000/100})
            baseline=(measurements[0]['event_ms_per_call']+measurements[2]['event_ms_per_call'])/2
            actual=measurements[1]['event_ms_per_call']
            result['cases'].append({'rows':rows,'shape':[rows,4,5120],'measurements':measurements,
                                    'reduction_pct_vs_two_references':100*(1-actual/baseline),'speedup':baseline/actual,
                                    'fused_faster_than_both_references':actual<min(measurements[0]['event_ms_per_call'],measurements[2]['event_ms_per_call'])})
            args.output.write_text(json.dumps(result,indent=2))
    result['status']='PASS';args.output.write_text(json.dumps(result,indent=2))
    logger.info('Engram paired local microbenchmark completed')


if __name__=='__main__':
    main()
