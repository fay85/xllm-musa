# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct native shape regression and timing for two independent packing candidates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
import torch
import torch_musa
import musa_python_kernel_test_module as native
sys.path.insert(0,'/work/repos/xllm-musa')
from scripts.logger import logger


def _error(a: torch.Tensor,b: torch.Tensor) -> dict:
    x=a.float();y=b.float();delta=x-y;rms=float((delta.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-8)).item());maximum=float(delta.abs().max().item());assert rms<0.02,{'rms':rms,'max_abs':maximum}
    return {'relative_rms':rms,'max_abs':maximum,'exact':torch.equal(a,b)}


def _time(function: object) -> float:
    for _ in range(3):function()
    torch.musa.synchronize();start=torch.musa.Event(enable_timing=True);end=torch.musa.Event(enable_timing=True);start.record()
    for _ in range(30):function()
    end.record();end.synchronize();return start.elapsed_time(end)/30


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert torch.musa.device_count()==1;torch.manual_seed(111215);torch.set_num_threads(1);device='musa:0';dtype=torch.bfloat16;rows=[];timings={}
    with torch.inference_mode():
      for intermediate in [256,2048]:
       weights=[(torch.randn(intermediate,5120)*0.02).to(device=device,dtype=dtype),(torch.randn(intermediate,5120)*0.02).to(device=device,dtype=dtype),(torch.randn(5120,intermediate)*0.02).to(device=device,dtype=dtype)]
       old=native.NativeV41SharedMlp(*weights,10.,True,False);new=native.NativeV41SharedMlp(*weights,10.,True,True)
       for count in [1,17]:
        for magnitude in [0.,1.,50.]:
         x=(torch.randn(count,5120)*magnitude).to(device=device,dtype=dtype);a=old.forward(x);b=new.forward(x);error=_error(b,a);rows.append({'kind':'shared_gate_up','intermediate':intermediate,'rows':count,'magnitude':magnitude,**error})
       x=torch.randn(1,5120,device=device,dtype=dtype);samples=[]
       for repeat in range(4):
        funcs={'old':lambda:old.forward(x),'packed':lambda:new.forward(x)};labels=list(funcs) if repeat%2==0 else list(reversed(funcs));samples.append({k:_time(funcs[k]) for k in labels})
       timings['shared_'+str(intermediate)]={'samples':samples,'old_ms':statistics.median(r['old'] for r in samples),'packed_ms':statistics.median(r['packed'] for r in samples)}
      weights=[(torch.randn(1536,5120)*0.02).to(device=device,dtype=dtype),torch.ones(1536,device=device,dtype=dtype),(torch.randn(4096,1536)*0.02).to(device=device,dtype=dtype),(torch.randn(512,5120)*0.02).to(device=device,dtype=dtype),torch.ones(512,device=device,dtype=dtype),(torch.randn(1024,4096)*0.02).to(device=device,dtype=dtype),(torch.randn(5120,1024)*0.02).to(device=device,dtype=dtype)]
      old=native.NativeV41AttentionProjection(weights,8,1,1024,1e-6,True,False);new=native.NativeV41AttentionProjection(weights,8,1,1024,1e-6,True,True)
      angle=torch.randn(2048,32,device=device);table=torch.cat([angle.cos(),angle.sin()],-1).float().contiguous()
      for count in [1,17]:
       for magnitude in [0.,1.,50.]:
        x=(torch.randn(count,5120)*magnitude).to(device=device,dtype=dtype);positions=torch.arange(count,device=device,dtype=torch.int64)+257;a=old.prepare(x,positions,table);b=new.prepare(x,positions,table);errors=[_error(y,z) for y,z in zip(b,a)];rows.append({'kind':'q_a_kv_packed','rows':count,'magnitude':magnitude,'output_errors_query_key_qr':errors})
      x=torch.randn(1,5120,device=device,dtype=dtype);positions=torch.tensor([257],device=device);samples=[]
      for repeat in range(4):
       funcs={'old':lambda:old.prepare(x,positions,table),'packed':lambda:new.prepare(x,positions,table)};labels=list(funcs) if repeat%2==0 else list(reversed(funcs));samples.append({k:_time(funcs[k]) for k in labels})
      timings['q_a_kv']={'samples':samples,'old_ms':statistics.median(r['old'] for r in samples),'packed_ms':statistics.median(r['packed'] for r in samples)}
      result={'status':'PASS_PACKED_NATIVE_PROJECTION_BASIC_CHECKS','cases':rows,'timings':timings,'scope':'GPU event interval includes host enqueue gaps; not full model or a model accuracy validation','accuracy_scope':'仅通过基础功能检查，未完成模型精度验收'};args.output.write_text(json.dumps(result,indent=2));logger.info('Packed projection basic checks finished, cases=%d',len(rows))


if __name__=='__main__':main()
