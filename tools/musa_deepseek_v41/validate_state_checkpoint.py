# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU paged-cache transaction; CPU copies are test oracles only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
import torch_musa
import musa_python_kernel_test_module as native
sys.path.insert(0,'/work/repos/xllm-musa')
from scripts.logger import logger


def bits(value: torch.Tensor) -> torch.Tensor:
    return value.view(torch.int16 if value.dtype==torch.bfloat16 else torch.int32).cpu()


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert torch.musa.device_count()==1
    torch.manual_seed(111225)
    torch.set_num_threads(1)
    device='musa:0'
    cases=[]
    with torch.inference_mode():
        for geometry in [(8,4,16),(8,128,512)]:
          for index_dtype in [torch.int32,torch.int64]:
           for nondefault in [False,True]:
            for ring in [False,True]:
             for first,count in [(0,0),(0,1),(3,5),(7,17),(31,5),(127,5),(255,5)]:
              if not ring and first+count>geometry[1]*3:continue
              stream=torch.musa.Stream() if nondefault else torch.musa.current_stream()
              with torch.musa.stream(stream):
                cache=torch.randn(geometry,device=device,dtype=torch.bfloat16)
                cache.view(-1)[:3]=torch.tensor([float('nan'),-0.,0.],device=device,dtype=torch.bfloat16)
                table=torch.tensor([[5,-1,2]],device=device,dtype=index_dtype)
                original=bits(cache)
                # Independent test oracle uses the same explicit scheduler map.
                logical=torch.arange(first,first+count,device=device,dtype=torch.int64)
                columns=logical//geometry[1]
                if ring:columns=columns%3
                slots=table[0].long()[columns]*geometry[1]+logical%geometry[1]
                slots=slots[(slots>=0)&(slots<geometry[0]*geometry[1])].unique()
                recurrent=torch.randn(6,32,512,device=device,dtype=torch.bfloat16)
                score=torch.randn(6,32,512,device=device)
                history=torch.arange(768,device=device,dtype=torch.int32)
                references=[bits(x) for x in [recurrent,score,history]]
                transaction=native.NativeV41StateCheckpoint(16*1024*1024)
                transaction.add_paged(cache,table,first,count,ring)
                for tensor in [recurrent,score,history]:transaction.add_full(tensor)
                saved=transaction.saved_bytes()
                if slots.numel():cache.view(-1,geometry[2]).index_fill_(0,slots,19)
                recurrent.fill_(21);score.fill_(22);history.fill_(23)
                # Untouched block 7 is another request: rollback must preserve it.
                cache[7].fill_(91)
                original[7]=bits(cache[7])
                transaction.restore()
                assert torch.equal(bits(cache),original)
                assert all(torch.equal(bits(x),ref) for x,ref in zip([recurrent,score,history],references))
                expected_bytes=slots.numel()*geometry[2]*2+sum(x.numel()*x.element_size() for x in [recurrent,score,history])
                assert saved==expected_bytes
                cases.append({'shape':geometry,'index_dtype':str(index_dtype),'nondefault':nondefault,'ring':ring,'first':first,'count':count,'saved_bytes':saved,'cache_recurrent_history_bitwise_restored':True,'other_request_block_preserved':True})
    result={'status':'PASS_BOUNDED_GPU_PAGED_AND_FULL_STATE_ROLLBACK','cases':cases,'scope':'Only production C++ transaction primitive; not DSpark inference, scheduler-length rollback, draft block attention, or cross-request prefix migration. All model/cache math in GPU; CPU copies are validation references.','accuracy_scope':'仅通过基础功能检查，未完成模型精度验收'}
    args.output.write_text(json.dumps(result,indent=2))
    logger.info('Cache rollback cases=%d, all bitwise',len(cases))


if __name__=='__main__':main()
