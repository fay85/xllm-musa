# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark block visibility and accepted-context-only cache writes on MUSA."""
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


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sparse-so',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(111232)
    assert torch.musa.device_count()==1
    device='musa:0'
    cases=[]
    with torch.inference_mode():
        oracle=native.NativeV41SparseAttention(512**-.5,32*1024*1024)
        for nondefault in [False,True]:
            stream=torch.musa.Stream() if nondefault else torch.musa.current_stream()
            with torch.musa.stream(stream):
                for table_dtype in [torch.int32,torch.int64]:
                    table=torch.tensor([[5,2]],device=device,dtype=table_dtype)
                    for previous in [0,1,127,128,129,257,511]:
                        for width in [1,5]:
                            cache=(torch.randn(7,128,512)*.2).to(device=device,dtype=torch.bfloat16)
                            before=cache.cpu().clone()
                            query=(torch.randn(width,8,512)*.2).to(device=device,dtype=torch.bfloat16)
                            current=(torch.randn(width,512)*.2).to(device=device,dtype=torch.bfloat16)
                            sink=torch.linspace(-2.,2.,8,device=device)
                            # Independent chronological logical-to-physical map.
                            logical=list(range(max(0,previous-128),previous))
                            physical=[([5,2][(position//128)%2]*128+position%128) for position in logical]
                            history=cache.view(-1,512).index_select(0,torch.tensor(physical,device=device,dtype=torch.int64))
                            rows=torch.cat([history,current],0)
                            picks=torch.arange(rows.size(0),device=device,dtype=torch.int32)[None,:].expand(width,-1).contiguous()
                            expected=oracle.forward(query,rows,picks,sink)
                            actual=native.v41_paged_draft_forward(cache,query,current,table,previous,sink,args.sparse_so)
                            assert torch.equal(before.view(torch.int16),cache.cpu().view(torch.int16)), 'Proposed KV must not mutate accepted history'
                            error=(actual.float()-expected.float())
                            maximum=float(error.abs().max().item())
                            rms=float((error.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-6)).item())
                            assert maximum<.015 and rms<.04,(previous,width,table_dtype,nondefault,maximum,rms)
                            if width==5:
                                # First query must see the future proposals in this non-causal block.
                                changed=current.clone();changed[-1].add_(2)
                                altered=native.v41_paged_draft_forward(cache,query,changed,table,previous,sink,args.sparse_so)
                                assert not torch.equal(altered[0].cpu(),actual[0].cpu())
                                assert torch.equal(before.view(torch.int16),cache.cpu().view(torch.int16))
                            cases.append({'nondefault':nondefault,'table_dtype':str(table_dtype),'previous_context':previous,'query_block_width':width,'left_history_rows':len(logical),'noncausal_entire_block_visible':True,'cache_bitwise_unchanged':True,'relative_rms':rms,'max_abs':maximum})
        commits=[]
        for dtype in [torch.int32,torch.int64]:
            for count in [0,1,6]:
                for nondefault in [False,True]:
                    stream=torch.musa.Stream() if nondefault else torch.musa.current_stream()
                    with torch.musa.stream(stream):
                        cache=torch.randn(7,128,512).to(device=device,dtype=torch.bfloat16)
                        expected=cache.cpu().clone()
                        raw=([0] if count==1 else [0,0,-1,130,511,130])[:count]
                        slots=torch.tensor(raw,device=device,dtype=dtype)
                        values=torch.randn(count,512).to(device=device,dtype=torch.bfloat16)
                        if count:values[-1,0]=float('nan');values[-1,1]=-0.
                        cpu=values.cpu()
                        for row,slot in enumerate(raw):
                            if slot>=0:expected.view(-1,512)[slot].copy_(cpu[row])
                        native.v41_draft_commit_context(cache,slots,values)
                        assert torch.equal(cache.cpu().view(torch.int16),expected.view(torch.int16))
                        commits.append({'dtype':str(dtype),'count':count,'nondefault':nondefault,'latest_duplicate_wins':True,'padding_skipped':True,'NaN_and_signed_zero_bits_preserved':True})
    result={'status':'PASS_DSPARK_NONCAUSAL_BLOCK_AND_CONTEXT_COMMIT','attention_cases':cases,'commit_cases':commits,'attention_implementation':'Actual MATE sparse library; independent FP32 GPU oracle with logical ring mapping','scope':'One required DSpark primitive, not integrated draft/target serving and no performance gain claimed','accuracy_scope':'仅通过基础功能检查，未完成模型精度验收'}
    args.output.write_text(json.dumps(result,indent=2))
    logger.info('DSpark noncausal cases=%d accepted-context commit cases=%d',len(cases),len(commits))


if __name__=='__main__':
    main()
