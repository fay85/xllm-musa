# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Native C++ model components: original dense rounding and CSA2 sink math."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
import torch.nn.functional as F

from scripts.logger import logger


def _attention_reference(q: torch.Tensor, kv: torch.Tensor, ids: torch.Tensor,
                         sink: torch.Tensor, scale: float, budget: int) -> torch.Tensor:
    result=torch.empty_like(q)
    tile=min(256,max(1,budget//(2*ids.shape[1]*q.shape[-1]*4)))
    for start in range(0,len(q),tile):
        pick=ids[start:start+tile].long()
        valid=(pick>=0)&(pick<len(kv))
        selected=kv[pick.clamp(0,len(kv)-1)].float()
        selected=torch.where(valid.unsqueeze(-1),selected,0)
        score=torch.einsum('shd,snd->shn',q[start:start+tile].float(),selected)*scale
        score=score.masked_fill(~valid.unsqueeze(1),float('-inf'))
        maximum=score.amax(-1).clamp_min(-1e30)
        probability=(score-maximum.unsqueeze(-1)).exp()
        denom=probability.sum(-1)+(sink-maximum).exp()
        value=torch.einsum('shn,snd->shd',probability,selected)/denom.unsqueeze(-1)
        result[start:start+tile]=value.to(q.dtype)
    return result


def _attention_cpu(q: torch.Tensor, kv: torch.Tensor, ids: torch.Tensor,
                   sink: torch.Tensor, scale: float) -> torch.Tensor:
    # Independent FP64 softmax includes a zero-valued sink key explicitly.
    q,kv,ids,sink=q.cpu().double(),kv.cpu().double(),ids.cpu().long(),sink.cpu().double()
    rows=[]
    for position in range(len(q)):
        valid=(ids[position]>=0)&(ids[position]<len(kv))
        selected=kv[ids[position][valid]]
        score=q[position]@selected.T*scale
        score=torch.cat((score,sink[:,None]),dim=-1)
        weights=score.softmax(-1)[:,:len(selected)]
        rows.append(weights@selected)
    return torch.stack(rows) if rows else torch.empty(q.shape,dtype=torch.float64)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if torch.musa.device_count()!=1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    torch.manual_seed(111130)
    record={'status':'RUNNING','scope':'native C++ components only, not full native CausalLM/PD',
            'checks':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def _save() -> None:
        args.output.write_text(json.dumps(record,indent=2))

    with torch.inference_mode():
        for k,n in [(32,19),(128,70),(5120,512),(512,5120)]:
            weight=torch.randn(n,k,device='musa',dtype=torch.bfloat16)*.01
            for quantized in (False,True):
                module=native.NativeV41Dense(weight,quantized)
                for shape in [(0,k),(1,k),(17,k),(2,3,k)]:
                    value=torch.randn(shape,device='musa',dtype=torch.bfloat16)
                    prepared=(native.musa_v41_block32_qdq(value.reshape(-1,k).contiguous())[0].reshape_as(value)
                              if quantized else value)
                    expected=F.linear(prepared,weight)
                    actual=module.forward(value)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    record['checks'].append({'kind':'dense','shape':shape,'n':n,'quantized':quantized,'exact':True})
                    _save()
        for hidden,inter in [(64,32),(5120,2304)]:
            gate=torch.randn(inter,hidden,device='musa',dtype=torch.bfloat16)*.01
            up=torch.randn_like(gate)*.01
            down=torch.randn(hidden,inter,device='musa',dtype=torch.bfloat16)*.01
            for quantized in (False,True):
                module=native.NativeV41SharedMlp(gate,up,down,10.,quantized)
                for rows in (0,1,8,17):
                    value=torch.randn(rows,hidden,device='musa',dtype=torch.bfloat16)
                    prepared=native.musa_v41_block32_qdq(value)[0] if quantized else value
                    packed=torch.cat((F.linear(prepared,gate),F.linear(prepared,up)),-1)
                    activation=native.musa_shared_swiglu(packed.contiguous(),10.)
                    if quantized:
                        activation=native.musa_v41_block32_qdq(activation)[0]
                    expected=F.linear(activation,down)
                    actual=module.forward(value)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    record['checks'].append({'kind':'shared_mlp','hidden':hidden,'inter':inter,'rows':rows,'quantized':quantized,'exact':True})
                    _save()
        for heads,dim,width,length in [(2,64,16,32),(8,512,640,1024)]:
            for rows in (0,1,3,17,257):
                q=torch.randn(rows,heads,dim,device='musa',dtype=torch.bfloat16)*.1
                kv=torch.randn(length,dim,device='musa',dtype=torch.bfloat16)*.1
                ids=torch.randint(1,length,(rows,width),device='musa')
                ids[:,::5]=-1
                ids[:,1::7]=length+2
                if rows:
                    ids[0,:]=-1
                kv[0]=float('nan')
                kv[-1]=float('nan')
                # Out-of-range picks clamp to these poisoned rows, but valid
                # picks must not read them. Masking only scores is insufficient.
                ids=torch.where(ids==length-1,torch.ones_like(ids),ids)
                sink=torch.linspace(-2,2,heads,device='musa')
                budget=32*1024*1024;scale=dim**-.5
                module=native.NativeV41SparseAttention(scale,budget)
                actual=module.forward(q,kv,ids,sink)
                expected=_attention_reference(q,kv,ids,sink,scale,budget)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                assert bool(torch.isfinite(actual).all())
                if rows:
                    assert torch.equal(actual[0],torch.zeros_like(actual[0]))
                if 0<rows<=3:
                    oracle=_attention_cpu(q,kv,ids,sink,scale)
                    torch.testing.assert_close(actual.cpu().double(),oracle,rtol=0,atol=.002)
                record['checks'].append({'kind':'sparse_attention','rows':rows,'heads':heads,'dim':dim,
                                         'pick_width':width,'all_invalid_row':True,'poisoned_padding':True,'exact':True})
                _save()
        # Verify execution follows the caller's non-default MUSA stream.
        stream=torch.musa.Stream();stream.wait_stream(torch.musa.current_stream())
        q=torch.randn(3,2,64,device='musa',dtype=torch.bfloat16);kv=torch.randn(32,64,device='musa',dtype=torch.bfloat16)
        ids=torch.arange(16,device='musa').repeat(3,1);sink=torch.zeros(2,device='musa')
        # Both inputs are produced on the default stream before the new wait.
        stream.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(stream):
            transformed=(q+1).contiguous();actual=native.NativeV41SparseAttention(.125,32*1024*1024).forward(transformed,kv,ids,sink)
        stream.synchronize()
        expected=_attention_reference(q+1,kv,ids,sink,.125,32*1024*1024)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        record['checks'].append({'kind':'nondefault_stream','exact':True})
    record.update(status='PASS',count=len(record['checks']))
    _save()
    logger.info('Native C++ attention/dense checks passed: %s',record['count'])


if __name__=='__main__':
    main()
