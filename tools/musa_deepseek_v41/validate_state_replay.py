# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real ratio2 attention state: speculative write, rollback, accepted-only replay."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--sparse-so',required=True);args=parser.parse_args()
    assert torch.musa.device_count()==1
    torch.set_num_threads(1);torch.manual_seed(111235);device='musa:0';dtype=torch.bfloat16
    names=['swa','key','index','compressor_kv_state','compressor_score_state']
    def random(shape: tuple) -> torch.Tensor:
        return (torch.randn(shape)*.005).to(device=device,dtype=dtype)
    def clone(cache: object) -> object:
        result=native.NativeV41AttentionCache()
        for name in names:setattr(result,name,getattr(cache,name).clone())
        return result
    def exact(left: object,right: object) -> None:
        for name in names:
            a=getattr(left,name).cpu().contiguous().view(torch.uint8);b=getattr(right,name).cpu().contiguous().view(torch.uint8)
            assert torch.equal(a,b),('cache exact',name)
    with torch.inference_mode():
        weights=native.NativeV41AttentionWeights();weights.projection=[random((1280,5120)),torch.ones(1280,device=device,dtype=dtype),random((4096,1280)),random((512,5120)),torch.ones(512,device=device,dtype=dtype),random((1024,4096)),random((5120,1024))];weights.sink=torch.zeros(8,device=device);weights.compressor_norm=torch.ones(512,device=device,dtype=dtype);weights.compressor_kv=random((512,5120)).float();weights.compressor_gate=random((512,5120)).float();weights.indexer=[random((4096,1280)),random((32,5120)),random((128,512)),torch.ones(128,device=device,dtype=dtype)]
        cfg=native.NativeV41AttentionConfig();cfg.mode=native.NativeV41AttentionMode.FULL;cfg.layer=0;cfg.source_layer=0;cfg.ratio=2;cfg.heads=8;cfg.groups=1;cfg.output_rank=1024;cfg.window=128;cfg.index_heads=32;cfg.index_dim=128;cfg.index_topk=256;cfg.epsilon=1e-6;cfg.scale=512**-.5;cfg.original_fp8_activation=True;cfg.sparse_library_path=args.sparse_so;cfg.indexer_score_directory='/work/artifacts/native_stage_i/lib/indexer';cfg.decode_cache_read_reuse=True;cfg.decode_fused_swa_read=True
        layer=native.NativeV41Attention(cfg,weights)
        angles=torch.arange(1024,device=device).float()[:,None]*torch.pow(10000.,-torch.arange(0,64,2,device=device).float()/64.)[None,:];rope=torch.cat((angles.cos(),angles.sin()),-1).contiguous();table=torch.tensor([[1,2,3,4]],device=device,dtype=torch.int32)
        cases=[]
        for nondefault in [False,True]:
            stream=torch.musa.Stream() if nondefault else torch.musa.current_stream()
            with torch.musa.stream(stream):
                for previous in [127,128,257]:
                    base=native.NativeV41AttentionCache();base.swa=torch.zeros(7,128,512,device=device,dtype=dtype);base.key=torch.zeros_like(base.swa);base.index=torch.zeros(7,128,128,device=device,dtype=dtype);base.compressor_kv_state=torch.zeros(7,32,512,device=device);base.compressor_score_state=torch.zeros_like(base.compressor_kv_state)
                    prefill=random((previous,5120));pos=torch.arange(previous,device=device,dtype=torch.int64)
                    layer.forward(prefill,pos,rope,base,table,table,table,previous,native.NativeV41AttentionRuntime())
                    proposals=random((7,5120))
                    for accepted in [1,3,6]:
                        trial=clone(base);reference=clone(base);checkpoint=native.NativeV41StateCheckpoint(64*1024*1024)
                        checkpoint.add_paged(trial.swa,table,previous,6,True)
                        first=previous//2;count=(previous+6)//2-first
                        checkpoint.add_paged(trial.key,table,first,count,False);checkpoint.add_paged(trial.index,table,first,count,False)
                        touched_positions=torch.arange(previous,previous+6,device=device,dtype=torch.int64)
                        blocks=table[0].index_select(0,(touched_positions//128)%table.size(1)).contiguous()
                        checkpoint.add_blocks(trial.compressor_kv_state,blocks);checkpoint.add_blocks(trial.compressor_score_state,blocks)
                        layer.forward(proposals[:6],touched_positions,rope,trial,table,table,table,previous+6,native.NativeV41AttentionRuntime())
                        checkpoint.restore();exact(trial,base)
                        # The rejected suffix must not influence the next real token.
                        for offset in range(accepted+1):
                            positions=torch.tensor([previous+offset],device=device,dtype=torch.int64);hidden=proposals[offset:offset+1]
                            expected=layer.forward(hidden,positions,rope,reference,table,table,table,previous+offset+1,native.NativeV41AttentionRuntime())
                            actual=layer.forward(hidden,positions,rope,trial,table,table,table,previous+offset+1,native.NativeV41AttentionRuntime())
                            assert torch.equal(actual.cpu(),expected.cpu()),'Accepted-only replay attention output'
                        exact(trial,reference)
                        cases.append({'nondefault':nondefault,'previous':previous,'speculative_inputs':6,'accepted_input_prefix':accepted,'next_real_token_checked':True,'after_rollback_all_cache_bytes_exact':True,'after_replay_all_cache_bytes_exact':True,'next_output_exact':True,'checkpoint_bytes':checkpoint.saved_bytes(),'recurrent_pages':sorted(set(blocks.cpu().tolist()))})
    result={'status':'PASS_REAL_RATIO2_ATTENTION_ROLLBACK_AND_ACCEPTED_REPLAY','cases':cases,'scope':'Native actual MATE sparse/AOT indexer and pair compressor, page-scoped SWA/key/index/recurrent snapshot. Engram and scheduler/native target integration remain separate. No whole-model precision gate.','accuracy_scope':'仅通过基础功能检查，未完成模型精度验收'}
    args.output.write_text(json.dumps(result,indent=2));logger.info('Actual ratio2 speculative rollback/replay cases=%d',len(cases))


if __name__=='__main__':main()
