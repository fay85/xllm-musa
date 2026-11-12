# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Existing native MHC reduction: short checks at actual small-batch boundaries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import torch
import torch_musa
from safetensors.torch import load_file

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--library-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('XLLM_DSV41_MHC_SMALL_BATCH_FUSION')!='1':
        raise ValueError('Require explicit small-batch policy before native import')
    import musa_python_kernel_test_module as native
    torch.set_num_threads(1)
    torch.musa.set_device(0)
    source=load_file(str(args.fixture),device='cpu')
    records=[]
    with torch.inference_mode():
        for part in ['attn','ffn']:
            fn,scale,base=[source[f'layers.0.hc_{part}_{term}'].to(device='musa',dtype=torch.float32) for term in ['fn','scale','base']]
            reference=native.NativeV41HyperConnection(fn,scale,base,20,1e-6,1e-6,str(args.library_directory),False)
            target=native.NativeV41HyperConnection(fn,scale,base,20,1e-6,1e-6,str(args.library_directory),True)
            for rows in [1,2,3,4,6,8,9]:
                torch.manual_seed(111351+rows)
                residual=torch.randn(rows,4,fn.size(1)//4,dtype=torch.bfloat16).to('musa')
                carried=torch.rand(rows,4,dtype=torch.float32).to('musa')
                expected=reference.prepare(residual,carried)
                actual=target.prepare(residual,carried)
                torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
                errors=[]
                for a,b in zip(actual[1:],expected[1:]):
                    torch.testing.assert_close(a,b,rtol=3e-4,atol=1e-4)
                    errors.append(float((a-b).abs().max().item()))
                value=torch.randn(rows,fn.size(1)//4,dtype=torch.bfloat16).to('musa')
                a=target.combine(value,residual,actual[2],actual[3]);b=reference.combine(value,residual,expected[2],expected[3])
                torch.testing.assert_close(a,b,rtol=.01,atol=.02)
                records.append({'part':part,'rows':rows,'metadata_max_abs':errors,'carried_pre_exact':True,'merged_max_abs':float((a-b).abs().max().item())})
    torch.musa.synchronize()
    args.output.write_text(json.dumps({'status':'PASS_SMALL_BATCH_MHC_NATIVE_LAYER_BOUNDARIES','cases':records,'accuracy_scope':'Short directly-related native GPU layer checks only; not model precision acceptance'},indent=2))

if __name__=='__main__':
    main()
