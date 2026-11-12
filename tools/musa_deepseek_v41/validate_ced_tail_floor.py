# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify bounded replay never attends stale history and commits exact tail slots."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--sparse-library',required=True);args=parser.parse_args();assert torch.musa.device_count()==1;torch.manual_seed(111217);records=[];device='musa:0';width=512;window=128;block_size=128
    with torch.inference_mode():
      for original_rows,context in [(129,129),(256,256),(512,512),(256,512),(512,1024)]:
       rows=min(window,original_rows);start=context-rows
       for with_compressed in [False,True]:
        for nondefault in [False,True]:
         stream=torch.musa.Stream() if nondefault else torch.musa.current_stream()
         with torch.musa.stream(stream):
          # Every historical cache row is NaN: a stale history read must not
          # influence any output when visibility is floored at tail start.
          cache=torch.full((5,block_size,width),float('nan'),device=device,dtype=torch.bfloat16);before=cache.clone();table=torch.tensor([[3,1]],device=device,dtype=torch.int32);position=torch.arange(start,context,device=device,dtype=torch.int64);current=torch.randn(rows,width,device=device,dtype=torch.bfloat16);query=torch.randn(rows,8,width,device=device,dtype=torch.bfloat16);sink=torch.randn(8,device=device)
          compressed=torch.randn(17,width,device=device,dtype=torch.bfloat16) if with_compressed else None
          extra=torch.tensor([[-1,0,5,16,17]],device=device,dtype=torch.int32).expand(rows,-1).contiguous() if with_compressed else None
          target=native.NativeV41PagedAttention(cache,window,width**-0.5,args.sparse_library,True,False)
          actual=target.forward_with_window_floor(query,current,table,context,sink,compressed,extra,position,start)
          desired=position[:,None]-window+1+torch.arange(window,device=device);picks=torch.where(desired.ge(start),desired-start+window,-1)
          candidates=torch.cat([torch.zeros(window,width,device=device,dtype=torch.bfloat16),current],0)
          if with_compressed:
           mapped=torch.where(extra.ge(0).logical_and(extra.lt(17)),extra.long()+window+rows,-1);picks=torch.cat([picks,mapped],-1);candidates=torch.cat([candidates,compressed],0)
          reference=native.NativeV41SparseAttention(width**-0.5,32*1024**2,args.sparse_library).forward(query,candidates,picks,sink)
          assert torch.equal(actual.cpu(),reference.cpu()),('tail visibility',original_rows,context,with_compressed,nondefault)
          assert torch.isfinite(actual).all().item(),'NaN stale history leaked into replay'
          pages=table[0].index_select(0,torch.floor_divide(position,block_size).remainder(table.size(1))).long();slots=pages*block_size+position.remainder(block_size);expected=before.view(-1,width).clone();expected.index_copy_(0,slots,current)
          assert torch.allclose(cache.view(-1,width).cpu(),expected.cpu(),rtol=0,atol=0,equal_nan=True),('tail writeback',original_rows,context)
          records.append({'original_rows':original_rows,'tail_rows':rows,'context':context,'tail_start':start,'with_compressed':with_compressed,'nondefault_stream':nondefault,'output_exact_independent_floor_reference':True,'stale_NaN_history_masked':True,'tail_slot_commit_exact':True})
    result={'status':'PASS_CED_TAIL_FLOOR_AND_RING_COMMIT_BASIC_CHECKS','cases':records,'scope':'Basic visibility/cache checks, not full model equivalence or performance. Full prompt logits of skipped rows unsupported.','accuracy_scope':'仅通过基础功能检查，未完成模型精度验收'};args.output.write_text(json.dumps(result,indent=2));logger.info('CED tail floor and cache checks passed: %d cases',len(records))


if __name__=='__main__':main()
