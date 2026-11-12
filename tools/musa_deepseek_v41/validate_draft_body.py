# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Three native DSpark layers, context injection and shifted residual carry."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
import torch_musa
from safetensors.torch import load_file
import musa_python_kernel_test_module as native
sys.path.insert(0,'/work/repos/xllm-musa')
from scripts.logger import logger


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',required=True)
    parser.add_argument('--mhc-fixture',required=True)
    parser.add_argument('--gemm-so',required=True)
    parser.add_argument('--mhc-directory',required=True)
    parser.add_argument('--sparse-so',required=True)
    args=parser.parse_args()
    assert torch.musa.device_count()==1
    torch.set_num_threads(1)
    torch.manual_seed(111233)
    device='musa:0';dtype=torch.bfloat16
    def random(shape: tuple) -> torch.Tensor:
        return (torch.randn(shape)*.005).to(device=device,dtype=dtype)
    with torch.inference_mode():
        loaded=native.v41_load_w4a8_fixture(args.fixture,torch.empty(0,device=device,dtype=dtype))
        hc=load_file(args.mhc_fixture,device='cpu')
        hcweights=[hc['layers.0.'+name].to(device) for name in ['hc_attn_fn','hc_attn_scale','hc_attn_base','hc_ffn_fn','hc_ffn_scale','hc_ffn_base']]
        configs=[];weights=[];ref_layers=[];ref_attention=[];ref_projection=[];ref_hc=[]
        for layer in range(3):
            aw=native.NativeV41AttentionWeights();aw.projection=[random((1536,5120)),torch.ones(1536,device=device,dtype=dtype),random((4096,1536)),random((512,5120)),torch.ones(512,device=device,dtype=dtype),random((1024,4096)),random((5120,1024))];aw.sink=torch.zeros(8,device=device)
            mw=native.NativeV41MoeWeights();mw.shared=[random((2048,5120)),random((2048,5120)),random((5120,2048))];empty=torch.empty(0,device=device,dtype=dtype);mw.expert=[loaded[0],loaded[2],empty];mw.scale=[loaded[1],loaded[3],empty];mw.gate=random((2,5120));mw.bias=torch.zeros(2,device=device);mw.vision_bias=None
            dw=native.NativeV41DecoderWeights();dw.attention=aw;dw.moe=mw;dw.hyper_connection=hcweights;dw.input_norm=torch.ones(5120,device=device,dtype=dtype);dw.post_attention_norm=dw.input_norm.clone();dw.engram=[]
            ac=native.NativeV41AttentionConfig();ac.mode=native.NativeV41AttentionMode.SWA;ac.layer=layer;ac.source_layer=-1;ac.ratio=0;ac.heads=8;ac.groups=1;ac.output_rank=1024;ac.window=128;ac.epsilon=1e-6;ac.scale=512**-.5;ac.original_fp8_activation=True;ac.sparse_library_path=args.sparse_so
            mc=native.NativeV41MoeConfig();mc.expert_format=native.NativeV41ExpertFormat.W4AFP8_MT;mc.dense=False;mc.original_fp8_activation=True;mc.first_expert=0;mc.topk=2;mc.normalize=True;mc.scale=1.5;mc.limit=10.;mc.w4a8_so_path=args.gemm_so
            cfg=native.NativeV41DecoderConfig();cfg.attention=ac;cfg.moe=mc;cfg.sinkhorn_iterations=20;cfg.hc_epsilon=1e-6;cfg.mhc_prenorm_directory=args.mhc_directory;cfg.decode_ffn_tail_graph=False
            configs.append(cfg);weights.append(dw);ref_layers.append(native.NativeV41Decoder(cfg,dw));ref_attention.append(native.NativeV41Attention(ac,aw));ref_projection.append(native.NativeV41AttentionProjection(aw.projection,8,1,1024,1e-6,True));ref_hc.append(native.NativeV41HyperConnection(*hcweights[:3],20,1e-6,1e-6,args.mhc_directory))
        embedding=random((256,5120));head=random((256,5120));main_weight=random((5120,15360));norm=torch.ones(5120,device=device,dtype=dtype)
        angles=torch.arange(1024,device=device).float()[:,None]*torch.pow(10000.,-torch.arange(0,64,2,device=device).float()/64.)[None,:];rope=torch.cat((angles.cos(),angles.sin()),-1).contiguous()
        model=native.NativeV41DraftModel(configs,weights,embedding,main_weight,norm,norm,head,rope)
        projection=native.NativeV41Dense(main_weight,True)
        table=torch.tensor([[5,2]],device=device,dtype=torch.int32)
        cases=[]
        for nondefault in [False,True]:
            stream=torch.musa.Stream() if nondefault else torch.musa.current_stream()
            with torch.musa.stream(stream):
                for previous in [0,129,257]:
                    caches=[random((7,128,512)) for _ in range(3)]
                    reference=[c.clone() for c in caches]
                    count=min(previous,4)
                    captured=random((count,15360));context_positions=torch.arange(previous-count,previous,device=device,dtype=torch.int64)
                    slots=(table[0].index_select(0,(context_positions//128)%2).long()*128+context_positions%128)
                    projected=model.write_context_kv(captured,context_positions,slots,caches)
                    expected_projected=native.v41_rms_norm(projection.forward(captured),norm,1e-6)
                    assert torch.equal(projected.cpu(),expected_projected.cpu())
                    for layer in range(3):
                        if count:
                            _,kv,_=ref_projection[layer].prepare(expected_projected,context_positions,rope)
                            native.v41_draft_commit_context(reference[layer],slots,kv)
                        assert torch.equal(caches[layer].cpu(),reference[layer].cpu())
                    before=[c.cpu().clone() for c in caches]
                    for anchor in [11,29]:
                        tokens=torch.tensor([anchor,7,7,7,7],device=device,dtype=torch.int32)
                        positions=torch.arange(previous,previous+5,device=device,dtype=torch.int64)
                        actual=model.forward(tokens,positions,caches,table,previous)
                        hidden=embedding.index_select(0,tokens.long()).unsqueeze(1).expand(-1,4,-1).contiguous()
                        pre=torch.zeros(5,4,device=device);pre[:,0]=1
                        for layer in range(3):
                            attn_input,attn_pre,attn_post,attn_comb=ref_hc[layer].prepare(hidden,pre)
                            attn_output=ref_attention[layer].forward_draft_block(native.v41_rms_norm(attn_input,norm,1e-6),positions,rope,reference[layer],table,previous)
                            hidden,pre=ref_layers[layer].forward_tail(attn_output,hidden,attn_pre,attn_post,attn_comb,tokens)
                        merged=(hidden.float()*pre.unsqueeze(-1)).sum(1).to(dtype)
                        expected=native.v41_rms_norm(merged,norm,1e-6)
                        difference=(actual.float()-expected.float());rms=float((difference.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-6)).item());maximum=float(difference.abs().max().item());assert rms<.03,(previous,anchor,nondefault,rms,maximum)
                        logits=model.logits(actual);reference_logits=torch.nn.functional.linear(actual,head);assert torch.equal(logits.cpu(),reference_logits.cpu())
                        for layer in range(3):
                            assert torch.equal(before[layer].view(torch.int16),caches[layer].cpu().view(torch.int16))
                        cases.append({'nondefault':nondefault,'previous':previous,'anchor':anchor,'block_width':5,'layers':3,'context_projection_exact':True,'cache_unchanged_by_proposals':True,'vocabulary_head_exact':True,'relative_rms':rms,'max_abs':maximum})
    result={'status':'PASS_NATIVE_THREE_LAYER_DSPARK_BODY_BASIC','cases':cases,'scope':'Single GPU slice geometry with real layer0 HC and two W4A8 expert fixture, synthetic dense projection/embedding/head. Actual MATE sparse/HC/W4A8. This is not TP8 full checkpoint or target/speculative serving validation.','accuracy_scope':'仅通过基础功能检查，未完成模型精度验收'}
    args.output.write_text(json.dumps(result,indent=2));logger.info('Native three-layer DSpark body cases=%d',len(cases))


if __name__=='__main__':main()
