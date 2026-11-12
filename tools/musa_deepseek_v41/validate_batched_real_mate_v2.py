# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real native batch projections with disjoint request cache and CSA2 runtime."""
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
    args=parser.parse_args()
    metadata_cases=[([0,1,2],[0,1,130],2,2,[1,129]),([0,1,2,3,4],[0,1,130,388,899],4,4,[1,129,258,511]),([0,1,3],[0,1,2],2,3,None),([0,1],[0,1],1,1,None),([1,2,3],[0,1,2],2,2,None),([0,1,2],[0,2,2],2,2,None),([0,1,2],[0,2,1],2,2,None),([0,1,2],[0,1],2,2,None),([0,1,2],[0,1,2],2,3,None)]
    for query,context,requests,tokens,expected in metadata_cases:
        assert native.v41_parse_decode_batch(query,context,requests,tokens)==expected
    assert torch.musa.device_count()==1
    torch.manual_seed(111229)
    torch.set_num_threads(1)
    device='musa:0';dtype=torch.bfloat16
    def _random(shape: tuple[int, ...]) -> torch.Tensor:return (torch.randn(shape)*.005).to(device=device,dtype=dtype)
    def _cache_copy(cache: native.NativeV41AttentionCache) -> native.NativeV41AttentionCache:
        result=native.NativeV41AttentionCache()
        for name in ['swa','key','index','compressor_kv_state','compressor_score_state']:
            value=getattr(cache,name)
            setattr(result,name,value.clone() if value is not None else None)
        return result
    with torch.inference_mode():
        weights=native.NativeV41AttentionWeights()
        weights.projection=[_random((1536,5120)),torch.ones(1536,device=device,dtype=dtype),_random((4096,1536)),_random((512,5120)),torch.ones(512,device=device,dtype=dtype),_random((1024,4096)),_random((5120,1024))]
        weights.sink=torch.zeros(8,device=device)
        indexer_weights=[_random((4096,1536)),_random((32,5120)),_random((128,512)),torch.ones(128,device=device,dtype=dtype)]
        weights.compressor_norm=torch.ones(512,device=device,dtype=dtype)
        angles=torch.arange(1024,device=device).float()[:,None]*torch.pow(10000.,-torch.arange(0,64,2,device=device).float()/64.)[None,:]
        rope=torch.cat((angles.cos(),angles.sin()),-1).contiguous()
        cases=[]
        for batch in [2,4]:
          for ratio in [0,1,2]:
           for nondefault in [False,True]:
            stream=torch.musa.Stream() if nondefault else torch.musa.current_stream()
            with torch.musa.stream(stream):
              cfg=native.NativeV41AttentionConfig();cfg.sparse_library_path='/work/artifacts/native_stage_i/lib/mate/libdeepseek_v41_sparse_h8_bf16.so';cfg.indexer_score_directory='/work/artifacts/native_stage_i/lib/indexer';cfg.decode_cache_read_reuse=True;cfg.decode_fused_swa_read=True;cfg.layer=0;cfg.source_layer=-1 if ratio==0 else 0;cfg.mode=native.NativeV41AttentionMode.SWA if ratio==0 else native.NativeV41AttentionMode.FULL;cfg.ratio=ratio;cfg.heads=8;cfg.groups=1;cfg.output_rank=1024;cfg.window=128;cfg.index_heads=32;cfg.index_dim=128;cfg.index_topk=256;cfg.epsilon=1e-6;cfg.scale=512**-.5;cfg.original_fp8_activation=True
              weights.compressor_kv=_random((512,5120)).float() if ratio==2 else _random((512,5120));weights.compressor_gate=_random((512,5120)).float() if ratio==2 else None
              weights.indexer=[] if ratio==0 else indexer_weights
              layer=native.NativeV41Attention(cfg,weights)
              contexts=[1,129,258,511][:batch];positions=torch.tensor([n-1 for n in contexts],device=device,dtype=torch.int64);hidden=_random((batch,5120));blocks=batch*4+4
              initial=native.NativeV41AttentionCache();initial.swa=torch.randn(blocks,128,512,device=device,dtype=dtype);initial.key=torch.randn(blocks,128,512,device=device,dtype=dtype);initial.index=torch.randn(blocks,128,128,device=device,dtype=dtype);initial.compressor_kv_state=torch.zeros(blocks,32,512,device=device) if ratio==2 else None;initial.compressor_score_state=torch.zeros(blocks,32,512,device=device) if ratio==2 else None
              # Every request owns disjoint physical pages and distinct lengths.
              table=torch.tensor([[1+4*r,2+4*r,3+4*r,4+4*r] for r in range(batch)],device=device,dtype=torch.int32)
              old=_cache_copy(initial);new=_cache_copy(initial);ref_runtime=[native.NativeV41AttentionRuntime() for _ in range(batch)];batch_runtime=[native.NativeV41AttentionRuntime() for _ in range(batch)]
              expected=[]
              for row in range(batch):expected.append(layer.forward(hidden[row:row+1],positions[row:row+1],rope,old,table[row:row+1],table[row:row+1],table[row:row+1],contexts[row],ref_runtime[row]))
              actual,batch_runtime=native.v41_attention_batched_with_runtime(layer,hidden,positions,rope,new,table,table,table,contexts,batch_runtime)
              expected=torch.cat(expected,0);difference=(actual.float()-expected.float());rms=float((difference.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-6)).item());maximum=float(difference.abs().max().item());assert rms<.05,('batched projection numerical bound',batch,ratio,nondefault,rms,maximum)
              # Unowned physical block0 is a sentinel for another request.
              assert torch.equal(new.swa[0].cpu().view(torch.int16),initial.swa[0].cpu().view(torch.int16))
              assert torch.equal(new.key[0].cpu().view(torch.int16),initial.key[0].cpu().view(torch.int16))
              assert torch.equal(new.index[0].cpu().view(torch.int16),initial.index[0].cpu().view(torch.int16))
              if ratio:
               for row in range(batch):
                assert batch_runtime[row].source_layer==0
                assert torch.equal(batch_runtime[row].key_table.cpu(),table[row:row+1].cpu())
               # Published key/index/compressor uses each q1 request unchanged.
               for name in ['key','index','compressor_kv_state','compressor_score_state']:
                left,right=getattr(old,name),getattr(new,name)
                if left is None:assert right is None
                else:assert torch.equal(left.cpu(),right.cpu()),('state isolation',batch,ratio,name)
              cases.append({'batch':batch,'ratio':ratio,'nondefault':nondefault,'contexts':contexts,'output_relative_rms':rms,'output_max_abs':maximum,'request_tables_and_CSA2_runtime_isolated':True,'source_compressed_state_exact':bool(ratio),'unowned_block0_unchanged':True})
    result={'status':'PASS_NATIVE_BATCHED_MATE_ATTENTION_AND_REQUEST_ISOLATION','cases':cases,'metadata_contract_cases':len(metadata_cases),'scope':'B2/B4 native C++ batch projections plus actual MATE sparse attention and AOT indexer, with accepted fused SWA/cache-read flags. Request-specific state isolation and 9 metadata contract checks. Full-model native adapter and formal serving remain pending.','accuracy_scope':'仅通过基础功能检查，未完成模型精度验收'}
    args.output.write_text(json.dumps(result,indent=2));logger.info('Batched attention isolation cases=%d',len(cases))


if __name__=='__main__':main()
