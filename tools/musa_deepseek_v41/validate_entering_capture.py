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
        model=native.NativeV41TextModel(configs,weights,embedding,norm,head,rope,rope)
        table=torch.tensor([[1,2,3,4]],device=device,dtype=torch.int32)
        cases=[]
        for nondefault in [False,True]:
            stream=torch.musa.Stream() if nondefault else torch.musa.current_stream()
            with torch.musa.stream(stream):
                for width in [1,5]:
                    tokens=torch.tensor([11]+[7]*(width-1),device=device,dtype=torch.int32)
                    positions=torch.arange(width,device=device,dtype=torch.int64)
                    caches=[];reference=[]
                    for layer in range(3):
                        c=native.NativeV41AttentionCache();c.swa=random((7,128,512));c.key=torch.zeros_like(c.swa);c.index=torch.zeros(7,128,128,device=device,dtype=dtype);c.compressor_kv_state=None;c.compressor_score_state=None;caches.append(c)
                        r=native.NativeV41AttentionCache();r.swa=c.swa.clone();r.key=c.key.clone();r.index=c.index.clone();r.compressor_kv_state=None;r.compressor_score_state=None;reference.append(r)
                    actual,captured=native.v41_text_forward_with_capture(model,tokens,positions,caches,table,table,table,width,[2,0],5120)
                    hidden=embedding.index_select(0,tokens.long()).unsqueeze(1).expand(-1,4,-1).contiguous()
                    pre=torch.zeros(width,4,device=device);pre[:,0]=1
                    expected_capture={}
                    runtime=native.NativeV41AttentionRuntime()
                    for layer in range(3):
                        if layer in [0,2]:expected_capture[layer]=hidden.mean(1).clone()
                        hidden,pre=ref_layers[layer].forward(hidden,pre,positions,rope,reference[layer],table,table,table,width,runtime,tokens,None,None)
                    expected_aux=torch.cat([expected_capture[2],expected_capture[0]],1)
                    assert torch.equal(captured.cpu(),expected_aux.cpu()), 'Entering residual and caller capture-slot order must agree'
                    expected=native.v41_rms_norm((hidden.float()*pre.unsqueeze(-1)).sum(1).to(dtype),norm,1e-6)
                    difference=(actual.float()-expected.float());rms=float((difference.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-6)).item());assert rms<.03
                    for layer in range(3):assert torch.equal(caches[layer].swa.cpu(),reference[layer].swa.cpu())
                    cases.append({'width':width,'nondefault':nondefault,'ordered_capture_ids':[2,0],'entering_residual_mean_and_slot_order_exact':True,'SWA_caches_exact':True,'final_hidden_relative_rms':rms})
    result={'status':'PASS_TARGET_ENTERING_RESIDUAL_CAPTURE','cases':cases,'scope':'Three-layer actual MATE sparse/mHC/W4A8 component fixture, caller capture order and entering-vs-exiting residual checked; this is not full target/spec serving validation.','accuracy_scope':'仅通过基础功能检查，未完成模型精度验收'}
    args.output.write_text(json.dumps(result,indent=2));logger.info('Native target entering capture cases=%d',len(cases))


if __name__=='__main__':main()
