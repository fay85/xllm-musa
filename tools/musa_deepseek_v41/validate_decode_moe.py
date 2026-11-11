# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Validate fixed-launch GPU routed experts and native C++ local EP slice."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
import torch.nn.functional as F

from scripts.logger import logger
from tools.musa_deepseek_v41.validate_fp4 import _quant_oracle, _weight_oracle


def _oracle(x: torch.Tensor, ids: torch.Tensor, routing: torch.Tensor,
            weights: list[torch.Tensor], scales: list[torch.Tensor],
            first: int, quantize: bool) -> torch.Tensor:
    x=x.cpu();ids=ids.cpu();routing=routing.cpu()
    decoded=[_weight_oracle(w.flatten(0,1),s.flatten(0,1)).reshape(w.shape[0],w.shape[1],w.shape[2]*2) for w,s in zip(weights,scales)]
    q=_quant_oracle(x)[0] if quantize else x
    result=torch.zeros_like(x,dtype=torch.float32)
    for token in range(x.shape[0]):
        # Stable ascending expert order is part of the existing combine path.
        for route in sorted(range(ids.shape[1]),key=lambda k:int(ids[token,k])):
            local=int(ids[token,route])-first
            if local<0 or local>=weights[0].shape[0]:
                continue
            g=(q[token].double()@decoded[0][local].double().T).bfloat16().float().clamp_max(10.)
            u=(q[token].double()@decoded[1][local].double().T).bfloat16().float().clamp(-10.,10.)
            h=(F.silu(g)*u*routing[token,route]).bfloat16().reshape(1,-1)
            h=_quant_oracle(h)[0] if quantize else h
            result[token]+=(h.double()@decoded[2][local].double().T).bfloat16().float().flatten()
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--numeric-fixture',type=Path,default=Path(__file__).resolve().parents[2]/'tests/data/deepseek_v41_fp4/bf16_halfway_minimal.pt')
    args=parser.parse_args()
    if torch.musa.device_count()!=1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime']=native
    from xllm.python import initialize_runtime
    initialize_runtime()
    torch.manual_seed(111107)
    result={'status':'RUNNING','checks':[],'scope':'GPU local expert decode and sublayers; not full model or cross-host PD'}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    def save() -> None:
        args.output.write_text(json.dumps(result,indent=2))

    def check(label: str,x: torch.Tensor,ids: torch.Tensor,routing: torch.Tensor,
              ws: list[torch.Tensor],ss: list[torch.Tensor],first: int,quantize: bool) -> None:
        call_args=(x,ids,routing,ws[0],ss[0],ws[1],ss[1],ws[2],ss[2],first,10.,quantize)
        actual=native.musa_v41_decode_moe(*call_args)
        cpp=native.NativeV41PackedRoutedExperts(ws[0],ss[0],ws[1],ss[1],ws[2],ss[2],first,10.,quantize)
        cpp_out=cpp.forward(x,ids,routing)
        torch.testing.assert_close(cpp_out.cpu(),actual.cpu(),rtol=0,atol=0)
        expected=_oracle(x,ids,routing,ws,ss,first,quantize)
        try:
            torch.testing.assert_close(actual.cpu(),expected,rtol=1/64,atol=5e-4)
        except AssertionError:
            result['status']='FAIL';result['failed_case']=label
            torch.save({'x':x.cpu(),'ids':ids.cpu(),'weights':routing.cpu(),'actual':actual.cpu(),'expected':expected},args.output.with_suffix('.failure.pt'));save();raise
        result['checks'].append({'name':label,'status':'PASS','shape':list(x.shape),'topk':ids.shape[1],'first_expert':first,'block32':quantize,'cpp_bitwise_equal':True,'rtol':1/64,'atol':5e-4})
        save()
    try:
        with torch.inference_mode():
            fixture=torch.load(args.numeric_fixture,weights_only=True)
            x=fixture['x'].to('musa');packed=fixture['packed'].to('musa');scales=fixture['scales'].to('musa')
            actual=native.musa_v41_packed_fp4_gemv(x,packed,scales)
            expected=(fixture['x'].double()@_weight_oracle(fixture['packed'],fixture['scales']).double().T).bfloat16()
            torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
            result['checks'].append({'name':'original-failed-up147-halfway','status':'PASS','exact_bf16':True,'actual':actual.cpu().tolist(),'expected':expected.tolist()});save()
            for hidden,inter,experts in [(64,32,3),(128,64,4),(5120,2304,2)]:
                ws=[torch.randint(0,256,shape,dtype=torch.uint8,device='musa') for shape in ((experts,inter,hidden//2),(experts,inter,hidden//2),(experts,hidden,inter//2))]
                ss=[torch.randint(120,123,(v.shape[0],v.shape[1],v.shape[2]//16),dtype=torch.uint8,device='musa') for v in ws]
                for rows in ((1,) if hidden==5120 else (0,1,8)):
                    for k in ((6,) if hidden==5120 else (1,6,16,32)):
                        x=torch.randn(rows,hidden,dtype=torch.bfloat16,device='musa')
                        ids=torch.randint(6,7+experts+2,(rows,k),device='musa')
                        rw=torch.rand(rows,k,dtype=torch.float32,device='musa')
                        for quantize in (False,True):check(f'random-{hidden}-{rows}-{k}-{quantize}',x,ids,rw,ws,ss,7,quantize)
            # All routes nonlocal, duplicate local experts, zeros, out-of-order IDs.
            ws=[torch.randint(0,256,shape,dtype=torch.uint8,device='musa') for shape in ((3,32,32),(3,32,32),(3,64,16))]
            ss=[torch.full((v.shape[0],v.shape[1],v.shape[2]//16),121,dtype=torch.uint8,device='musa') for v in ws]
            x=torch.randn(1,64,dtype=torch.bfloat16,device='musa')
            for label,values in [('empty',[0,1,100,101,102,103]),('duplicate',[9,7,8,7,9,8]),('mixed',[9,100,7,8,-1,7])]:
                ids=torch.tensor([values],device='musa');rw=torch.tensor([[.1,.2,0.,.4,.15,.15]],device='musa')
                for q in (False,True):check(label+str(q),x,ids,rw,ws,ss,7,q)
            stream=torch.musa.Stream();stream.wait_stream(torch.musa.current_stream())
            ids=torch.tensor([[9,7,8,100,101,-1]],device='musa');rw=torch.rand(1,6,device='musa')
            with torch.musa.stream(stream):
                actual=native.musa_v41_decode_moe((x+1).contiguous(),ids,rw,ws[0],ss[0],ws[1],ss[1],ws[2],ss[2],7,10.,True)
            stream.synchronize()
            torch.testing.assert_close(actual.cpu(),_oracle(x+1,ids,rw,ws,ss,7,True),rtol=1/64,atol=5e-4)
            result['checks'].append({'name':'nondefault-stream','status':'PASS'});save()
            # Actual model forward: the GPU route must avoid the host segment helper.
            from xllm.python.models.deepseek_v41 import DeepseekV41MoE,DeepseekV41Config
            cfg=DeepseekV41Config(hidden_size=64,moe_intermediate_size=32,n_routed_experts=384,n_activated_experts=6,routed_scaling_factor=1.5,swiglu_limit=10.)
            os.environ['XLLM_DSV41_MUSA_GATE']='v41_fused';os.environ['XLLM_DSV41_MUSA_SHARED_SWIGLU']='v4_fused'
            for mode in ('packed_bf16','block32'):
                os.environ['XLLM_DSV41_MUSA_FP4_LINEAR']=mode
                os.environ['XLLM_DSV41_MUSA_FP4_MOE']='reference'
                baseline=DeepseekV41MoE(cfg,0,torch.bfloat16,torch.device('musa:0'),packed_fp4=True)
                baseline.gate.weight.zero_();baseline.e_score_correction_bias.copy_(torch.arange(384,device='musa',dtype=torch.float32))
                for name in ('w1','w2','w3'):
                    a=getattr(baseline,'experts_'+name);b=getattr(baseline,'experts_'+name+'_scale')
                    a.copy_(torch.randint(0,256,a.shape,device='musa',dtype=torch.uint8));b.fill_(121)
                    getattr(baseline,'shared_'+name).weight.fill_(.003)
                os.environ['XLLM_DSV41_MUSA_FP4_MOE']='v41_decode'
                model=DeepseekV41MoE(cfg,0,torch.bfloat16,torch.device('musa:0'),packed_fp4=True)
                model.load_state_dict(baseline.state_dict())
                for rows in (0,1,8):
                    x=torch.randn(rows,64,dtype=torch.bfloat16,device='musa')
                    expected=baseline(x)
                    with patch.object(DeepseekV41MoE,'_local_experts',side_effect=AssertionError('host segments called')), patch.object(torch.Tensor,'cpu',side_effect=AssertionError('unexpected D2H')), patch.object(torch.Tensor,'tolist',side_effect=AssertionError('unexpected host list')):
                        actual=model(x)
                    torch.testing.assert_close(actual.cpu(),expected.cpu(),rtol=1/64,atol=5e-4)
                    result['checks'].append({'name':f'model-no-host-segments-{mode}-{rows}','status':'PASS'});save()
                x=torch.randn(17,64,dtype=torch.bfloat16,device='musa')
                torch.testing.assert_close(model(x).cpu(),baseline(x).cpu(),rtol=0,atol=0)
                result['checks'].append({'name':f'large-M-retains-explicit-path-{mode}','status':'PASS'});save()
        result.update(status='PASS',count=len(result['checks']));save()
        logger.info('GPU local decode expert checks passed: %s',len(result['checks']))
    except Exception:
        result['status']='FAIL';save();raise


if __name__=='__main__':
    main()
