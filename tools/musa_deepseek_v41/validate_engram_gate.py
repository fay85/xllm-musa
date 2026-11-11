# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Original Engram semantics against the frozen delivered SGLang oracle."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401

from scripts.logger import logger
from xllm.python.kernels_musa.engram import original_engram_gate


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-test',type=Path,required=True)
    parser.add_argument('--real-weights',type=Path,required=True)
    args=parser.parse_args()
    if torch.musa.device_count()!=1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    source=args.reference_test.read_bytes()
    reference_sha=hashlib.sha256(source).hexdigest()
    if reference_sha!='eb1dc12562aa02360b1796b3222db58aa83c2db116fad4225d40ed03ba3ba081':
        raise ValueError('Delivered independent oracle source changed')
    tree=ast.parse(source)
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='reference')
    space={'torch':torch}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(args.reference_test),'exec'),space)
    reference=space['reference']
    import musa_python_kernel_test_module as native
    torch.manual_seed(111140)
    result={'status':'RUNNING','oracle_source_sha256':reference_sha,'scope':'Engram gate only; lookup/projection/PD and fullmodel separate',
            'checks':[],'tolerance':'BF16/FP32 exact versus frozen delivered GPU oracle; independent CPU FP64 rounded toBF16 atol0.015625'}
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def _check(label: str, tensors: tuple[torch.Tensor,...], mask: torch.Tensor | None) -> None:
        x,kv,qw,kw=tensors
        originals=[t.clone() for t in tensors]
        expected=reference(x,kv,qw,kw)
        if mask is not None:
            expected=torch.where(mask[:,None,None],expected,x)
        semantic=original_engram_gate(x,kv,qw,kw,mask,1e-20)
        torch.testing.assert_close(semantic,expected,rtol=0,atol=0)
        actual=native.musa_v41_engram_gate(x,kv,qw,kw,mask)
        try:
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        except AssertionError:
            diff=(actual.float()-expected.float()).abs()
            result.update(status='FAIL',failure={'label':label,'x_dtype':str(x.dtype),'weight_dtype':str(qw.dtype),
                                               'max_abs':float(diff.max()),'different':int((actual!=expected).sum())})
            args.output.write_text(json.dumps(result,indent=2))
            torch.save({'inputs':[t.cpu() for t in tensors],'mask':None if mask is None else mask.cpu(),
                        'actual':actual.cpu(),'expected':expected.cpu()},args.output.with_suffix('.failure.pt'))
            raise
        for before,after in zip(originals,tensors):
            assert torch.equal(before,after),'Gate mutated its input'
        if len(x):
            assert actual.data_ptr() not in [t.data_ptr() for t in tensors]
        if len(x)==1 and x.dtype==torch.bfloat16:
            # CPU FP64 independently normalizes each copy and includes its
            # signed square root; it is an oracle only, never a serving path.
            h=x.cpu().double();key=kv[:,:20480].cpu().double().reshape(-1,4,5120);value=kv[:,20480:].cpu().double()
            weight=qw.cpu().double()*kw.cpu().double()
            dot=(h*weight*key).sum(-1)*torch.rsqrt(h.square().mean(-1)+1e-20)*torch.rsqrt(key.square().mean(-1)+1e-20)*(5120**-.5)
            gate=torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(),dot))
            if mask is not None:gate=gate.masked_fill(~mask.cpu()[:,None],0)
            oracle=h+gate[:,:,None]*value[:,None,:]
            # Compare rounded BF16 output to rounded high-precision output.
            torch.testing.assert_close(actual.cpu(),oracle.bfloat16(),rtol=0,atol=.015625)
        result['checks'].append({'label':label,'rows':len(x),'dtype':str(x.dtype),'weight_dtype':str(qw.dtype),
                                 'mask':mask is not None,'exact':True,'ownership':True})
        args.output.write_text(json.dumps(result,indent=2))

    with torch.inference_mode():
        for dtype in (torch.bfloat16,torch.float32):
            for weight_dtype in (torch.bfloat16,torch.float32):
                for rows in (0,1,8,17,128,256,512):
                    tensors=(torch.randn(rows,4,5120,device='musa',dtype=dtype),
                             torch.randn(rows,25600,device='musa',dtype=dtype),
                             torch.randn(4,5120,device='musa',dtype=weight_dtype),
                             torch.randn(4,5120,device='musa',dtype=weight_dtype))
                    _check('random',tensors,None)
                    if rows in (1,17):
                        mask=torch.arange(rows,device='musa')%2==1
                        _check('token-mask',tensors,mask)
                    if rows==1:
                        tensors[0].zero_();_check('zero',tensors,None)
                        tensors[0].fill_(1);tensors[1].fill_(1);tensors[2].fill_(1);tensors[3].fill_(1);tensors[3][:,1::2]=-1
                        _check('cancellation',tensors,None)
                        tensors[0].fill_(-0.0);_check('negative-zero',tensors,None)
        manifest=json.loads((args.real_weights/'manifest.json').read_text())
        for layer in (1,14):
            weights=[]
            for name in ('q_weight','k_weight'):
                item=manifest['tensors'][f'layers.{layer}.engram.{name}'];raw=(args.real_weights/item['filename']).read_bytes()
                assert hashlib.sha256(raw).hexdigest()==item['sha256']
                dtype={'BF16':torch.bfloat16,'F32':torch.float32}[item['dtype']]
                weights.append(torch.frombuffer(bytearray(raw),dtype=dtype).reshape(item['shape']).to('musa'))
            for rows in (1,8,128):
                tensors=(torch.randn(rows,4,5120,device='musa',dtype=torch.bfloat16),
                         torch.randn(rows,25600,device='musa',dtype=torch.bfloat16),*weights)
                _check(f'real1017-layer{layer}-weights',tensors,None)
    result.update(status='PASS',count=len(result['checks']))
    args.output.write_text(json.dumps(result,indent=2))
    logger.info('Engram gate checks passed: %s',result['count'])


if __name__=='__main__':
    main()
