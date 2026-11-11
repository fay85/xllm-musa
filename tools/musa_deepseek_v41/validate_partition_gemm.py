# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Validate original FP4 group32 linear through native MATE BF16 K32 partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
import torch.nn.functional as F

from scripts.logger import logger
from tools.musa_deepseek_v41.validate_fp4 import _quant_oracle, _weight_oracle


def _quant_codes(x: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    value=x.cpu().float().reshape(x.shape[0],x.shape[1]//32,32)
    raw=value.abs().amax(-1).clamp_min(1e-4)*(1./448.)
    scales=torch.pow(2.,raw.double().log2().ceil()).float()
    q=(value/scales.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn)
    return q.permute(1,0,2).contiguous(),scales.T.unsqueeze(-1).contiguous()


def _read_expert(root: Path) -> tuple[list[torch.Tensor],list[torch.Tensor],dict]:
    manifest=json.loads((root/'manifest.json').read_text());raw=(root/'expert.raw').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['sha256']:
        raise RuntimeError('Real expert blob checksum mismatch')
    tensors={}
    for name,info in manifest['tensors'].items():
        data=raw[info['offset']:info['offset']+info['bytes']]
        if hashlib.sha256(data).hexdigest()!=info['sha256']:
            raise RuntimeError(f'Real expert tensor checksum mismatch: {name}')
        tensors[name]=torch.frombuffer(bytearray(data),dtype=torch.uint8).reshape(info['shape']).clone().to('musa')
    prefix='layers.14.ffn.experts.0.'
    return [tensors[prefix+n+'.weight'] for n in ('w1','w3','w2')],[tensors[prefix+n+'.scale'] for n in ('w1','w3','w2')],manifest


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bmm-so',type=Path,required=True)
    p.add_argument('--real-expert',type=Path,required=True)
    args=p.parse_args()
    if torch.musa.device_count()!=1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    os.environ['XLLM_TILELANG_LIB']='/usr/local/lib/python3.10/dist-packages/tilelang/lib/libtilelang.so'
    os.environ['XLLM_DSV41_MATE_BMM_SO']=str(args.bmm_so)
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime']=native
    from xllm.python import initialize_runtime
    initialize_runtime()
    from xllm.python.kernels_musa.fp4 import select_fp4_linear
    from xllm.python.models.deepseek_v41 import dequant_fp4_group,_swiglu_clamped,_PackedFP4Linear
    torch.manual_seed(111113)
    budget=512*1024*1024
    result={'status':'RUNNING','checks':[],'benchmarks':[], 'bmm_so_sha256':hashlib.sha256(args.bmm_so.read_bytes()).hexdigest(), 'scope':'BF16 mantissa K32 operator and expert layer; not native FP4/FP8 GEMM or full-model correctness/performance'}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    def _save() -> None:
        args.output.write_text(json.dumps(result,indent=2))
    def _passed(name: str,**fields: object) -> None:
        result['checks'].append({'name':name,'status':'PASS',**fields});_save()
    def _linear(x: torch.Tensor,w: torch.Tensor,s: torch.Tensor) -> torch.Tensor:
        return native.musa_v41_partition_linear(x.contiguous(),w.contiguous(),s.contiguous(),str(args.bmm_so),budget)
    def _expected_linear(x: torch.Tensor,w: torch.Tensor,s: torch.Tensor) -> torch.Tensor:
        q=_quant_oracle(x.cpu())[0]
        return (q.double()@_weight_oracle(w,s).double().T).bfloat16()
    try:
        with torch.inference_mode():
            for rows in (0,1,17,256):
                for inner in (32,64,2304,5120):
                    x=torch.randn(rows,inner,dtype=torch.bfloat16)
                    factor=torch.pow(2.,torch.randint(-15,10,(rows,inner//32)).float())
                    x=(x.float()*factor.repeat_interleave(32,-1)).bfloat16()
                    q,s=native.musa_v41_block32_batch_quant(x.to('musa'))
                    eq,es=_quant_codes(x)
                    torch.testing.assert_close(q.cpu().view(torch.uint8),eq.view(torch.uint8),rtol=0,atol=0)
                    torch.testing.assert_close(s.cpu(),es,rtol=0,atol=0)
                    _passed(f'quant-layout-{rows}-{inner}',exact_codes_and_scales=True)
            packed=torch.tensor([i|(i<<4) for i in range(16)],dtype=torch.uint8).reshape(16,1).repeat(1,32)
            sc=torch.tensor([0,1,2,100,120,121,122,123,124,125,126,127,128,129,130,131],dtype=torch.uint8).reshape(16,1).repeat(1,2)
            b,bs=native.musa_v41_block32_batch_unpack(packed.to('musa'),sc.to('musa'))
            table=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])
            eb=table[:,None].repeat(1,64).reshape(16,2,32).permute(1,0,2).contiguous().to(torch.float8_e4m3fn)
            es=torch.pow(2.,sc.double()-127).float().T.unsqueeze(-1).contiguous()
            torch.testing.assert_close(b.cpu().view(torch.uint8),eb.view(torch.uint8),rtol=0,atol=0)
            torch.testing.assert_close(bs.cpu(),es,rtol=0,atol=0)
            _passed('all-E2M1-values-UE8M0-and-layout',exact_codes_and_scales=True)
            for groups in (1,2,72,160):
                part=torch.randn(groups,17,65,dtype=torch.float32,device='musa')
                actual=native.musa_v41_block32_batch_reduce(part)
                expected=part.cpu().double().sum(0).bfloat16()
                torch.testing.assert_close(actual.cpu(),expected,rtol=1/128,atol=1e-4)
                _passed(f'partial-reduce-{groups}')
            part=torch.tensor([2.**24,1.,-2.**24],device='musa').reshape(3,1,1)
            torch.testing.assert_close(native.musa_v41_block32_batch_reduce(part).cpu(),torch.ones(1,1,dtype=torch.bfloat16),rtol=0,atol=0)
            _passed('partial-reduce-cancellation',exact=True)
            for m,n,k in [(0,65,64),(17,65,5120)]:
                part=torch.randn(k//32,m,n,device='musa')
                sa=torch.pow(2.,torch.randint(-5,2,(k//32,m,1),device='musa').float())
                sb=torch.pow(2.,torch.randint(-5,2,(k//32,n,1),device='musa').float())
                actual=native.musa_v41_block32_batch_reduce_scaled(part,sa,sb)
                expected=(part.cpu().double()*sa.cpu().double()*sb.cpu().double().transpose(-1,-2)).sum(0).bfloat16()
                torch.testing.assert_close(actual.cpu(),expected,rtol=1/128,atol=1e-4)
                _passed(f'scaled-partial-reduce-{m}-{n}-{k}')
            for m,n,k in [(0,65,64),(1,65,64),(4,128,256),(17,128,256),(256,128,256),(1,2304,5120),(4,2304,5120),(17,2304,5120),(256,2304,5120),(1,5120,2304),(17,5120,2304)]:
                x=torch.randn(m,k,dtype=torch.bfloat16,device='musa')
                w=torch.randint(0,256,(n,k//2),dtype=torch.uint8,device='musa')
                s=torch.randint(120,124,(n,k//32),dtype=torch.uint8,device='musa')
                actual=_linear(x,w,s);expected=_expected_linear(x,w,s)
                torch.testing.assert_close(actual.cpu(),expected,rtol=1/128,atol=1e-3)
                _passed(f'linear-{m}-{n}-{k}',max_abs=float((actual.cpu().float()-expected.float()).abs().max()) if actual.numel() else 0.,partial_bytes=m*n*(k//32)*4)
            for m,n,k,chunk_budget in [(257,65,64,128*1024*1024),(17,128,256,3*128*8*4)]:
                x=torch.randn(m,k,dtype=torch.bfloat16,device='musa')
                w=torch.randint(0,256,(n,k//2),dtype=torch.uint8,device='musa')
                s=torch.randint(120,124,(n,k//32),dtype=torch.uint8,device='musa')
                actual=native.musa_v41_partition_linear(x,w,s,str(args.bmm_so),chunk_budget)
                torch.testing.assert_close(actual.cpu(),_expected_linear(x,w,s),rtol=1/128,atol=1e-3)
                _passed(f'chunked-linear-{m}-{n}-{k}',partial_budget=chunk_budget)
            ws,ss,manifest=_read_expert(args.real_expert);result['real_expert']=manifest;_save()
            os.environ['XLLM_DSV41_MUSA_FP4_LINEAR']='partition32'
            op=select_fp4_linear(torch.device('musa:0'),torch.bfloat16,True,dequant_fp4_group)
            for rows in (0,1,4,17):
                x=(torch.randn(rows,5120,dtype=torch.bfloat16,device='musa')*.5).contiguous()
                rw=torch.linspace(.137,.913,rows,dtype=torch.float32,device='musa')
                gate=op(x,ws[0],ss[0]);up=op(x,ws[1],ss[1])
                py=_swiglu_clamped(x,gate,up,_PackedFP4Linear(op,ws[2],ss[2]),10.,rw)
                cpp=native.NativeV41PartitionRoutedMlp(ws[0],ss[0],ws[1],ss[1],ws[2],ss[2],10.,str(args.bmm_so),budget).forward(x,rw)
                torch.testing.assert_close(cpp.cpu(),py.cpu(),rtol=0,atol=0)
                g=_expected_linear(x,ws[0],ss[0]).float().clamp_max(10.)
                u=_expected_linear(x,ws[1],ss[1]).float().clamp(-10.,10.)
                hidden=(F.silu(g)*u*rw.cpu().unsqueeze(-1)).bfloat16()
                expected=_expected_linear(hidden,ws[2],ss[2])
                try:
                    torch.testing.assert_close(cpp.cpu(),expected,rtol=1/64,atol=5e-4)
                except AssertionError:
                    eg=_expected_linear(x,ws[0],ss[0]);eu=_expected_linear(x,ws[1],ss[1])
                    gpu_hidden=(F.silu(gate.float().clamp_max(10.))*up.float().clamp(-10.,10.)*rw.unsqueeze(-1)).bfloat16()
                    same_gpu_input_cpu_hidden=(F.silu(gate.cpu().float().clamp_max(10.))*up.cpu().float().clamp(-10.,10.)*rw.cpu().unsqueeze(-1)).bfloat16()
                    qx,_=native.musa_v41_block32_qdq(x)
                    dense_g=F.linear(qx,dequant_fp4_group(ws[0],ss[0]));dense_u=F.linear(qx,dequant_fp4_group(ws[1],ss[1]))
                    def _difference(a: torch.Tensor,b: torch.Tensor) -> dict:
                        a=a.cpu();b=b.cpu();d=(a.float()-b.float()).abs();indices=(a!=b).nonzero();examples=[]
                        for ix in indices[:12]:
                            loc=tuple(ix.tolist());examples.append({'index':list(loc),'actual':float(a[loc]),'expected':float(b[loc])})
                        return {'different':int(indices.shape[0]),'max_abs':float(d.max()) if d.numel() else 0,'examples':examples}
                    diag={'gate_partition_vs_cpu':_difference(gate,eg),'up_partition_vs_cpu':_difference(up,eu),'gate_dense_vs_cpu':_difference(dense_g,eg),'up_dense_vs_cpu':_difference(dense_u,eu),'hidden_vs_cpu':_difference(gpu_hidden,hidden),'hidden_same_input_cpu':_difference(gpu_hidden,same_gpu_input_cpu_hidden)}
                    down_with_cpu_hidden=_linear(hidden.to('musa'),ws[2],ss[2])
                    diag['down_given_cpu_hidden']=_difference(down_with_cpu_hidden,expected)
                    qh,sh=native.musa_v41_block32_batch_quant(gpu_hidden);eqh,esh=_quant_codes(hidden)
                    diag['down_input_qcodes']=_difference(qh.float(),eqh.float());diag['down_input_scales']=_difference(sh,esh)
                    torch.save({'x':x.cpu(),'route_weight':rw.cpu(),'gate':gate.cpu(),'up':up.cpu(),'expected_gate':eg,'expected_up':eu,'gpu_hidden':gpu_hidden.cpu(),'cpu_hidden':hidden,'actual':cpp.cpu(),'expected':expected},args.output.with_suffix('.failure.pt'))
                    args.output.with_suffix('.diagnostic.json').write_text(json.dumps(diag,indent=2));raise
                _passed(f'real1017-layer14-expert0-mlp-{rows}',python_cpp_exact=True)
            stream=torch.musa.Stream();stream.wait_stream(torch.musa.current_stream())
            x=torch.randn(4,5120,dtype=torch.bfloat16,device='musa')
            with torch.musa.stream(stream):actual=_linear(x+1,ws[0],ss[0])
            stream.synchronize();torch.testing.assert_close(actual.cpu(),_expected_linear(x+1,ws[0],ss[0]),rtol=1/128,atol=1e-3)
            _passed('nondefault-stream-native-ffi')
            # A bounded local comparison; no serving/QPS/TTFT claim.
            for rows in (4,17):
                x=torch.randn(rows,5120,dtype=torch.bfloat16,device='musa')
                def _reference() -> torch.Tensor:
                    q,_=native.musa_v41_block32_qdq(x)
                    return F.linear(q,dequant_fp4_group(ws[0],ss[0]))
                def _partition() -> torch.Tensor:
                    return _linear(x,ws[0],ss[0])
                for label,fn in [('bf16_reference',_reference),('partition32',_partition)]:
                    for _ in range(3):fn()
                    torch.musa.synchronize();samples=[]
                    baseline_bytes=torch.musa.memory_allocated()
                    torch.musa.reset_peak_memory_stats()
                    for _ in range(5):
                        start=time.perf_counter()
                        for _ in range(5):fn()
                        torch.musa.synchronize();samples.append((time.perf_counter()-start)*1000/5)
                    result['benchmarks'].append({'rows':rows,'n':2304,'k':5120,'mode':label,'wall_ms_per_call':samples,'median_ms':statistics.median(samples),'peak_allocated':torch.musa.max_memory_allocated(),'baseline_allocated':baseline_bytes,'peak_increment':torch.musa.max_memory_allocated()-baseline_bytes,'scope':'one expert w1 only including current weight preparation; no server/communication'})
                    _save()
        result.update(status='PASS',count=len(result['checks']));_save();logger.info('Partition32 checks passed: %s',len(result['checks']))
    except Exception:
        result['status']='FAIL';_save();raise


if __name__=='__main__':
    main()
