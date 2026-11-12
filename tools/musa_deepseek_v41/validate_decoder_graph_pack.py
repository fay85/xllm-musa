# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native cache-free decoder-tail graph with real HC/expert weights and changing inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Callable
import torch
import torch_musa
from safetensors.torch import load_file
import musa_python_kernel_test_module as native
sys.path.insert(0,'/work/repos/xllm-musa')
from scripts.logger import logger


def _time(function: Callable) -> float:
    for _ in range(3):function()
    torch.musa.synchronize();start=torch.musa.Event(enable_timing=True);end=torch.musa.Event(enable_timing=True);start.record()
    for _ in range(30):function()
    end.record();end.synchronize();return start.elapsed_time(end)/30


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fixture',required=True);parser.add_argument('--mhc-fixture',required=True);parser.add_argument('--gemm-so',required=True);parser.add_argument('--mhc-directory',required=True);args=parser.parse_args();assert torch.musa.device_count()==1;torch.manual_seed(111218);torch.set_num_threads(1);device='musa:0';dtype=torch.bfloat16
    with torch.inference_mode():
      bit_checks=[]
      for token_dtype in [torch.int32,torch.int64]:
       for nondefault in [False,True]:
        stream=torch.musa.Stream() if nondefault else torch.musa.current_stream()
        with torch.musa.stream(stream):
         sources=[torch.randn(1,5120,device=device,dtype=dtype),torch.randn(1,4,5120,device=device,dtype=dtype),torch.randn(1,4,device=device),torch.randn(1,4,device=device),torch.randn(1,4,4,device=device),torch.tensor([2**40+17 if token_dtype==torch.int64 else 123],device=device,dtype=token_dtype)]
         for value in sources[:-1]:value.view(-1)[:3]=torch.tensor([float('nan'),-0.,0.],device=device,dtype=value.dtype)
         targets=[torch.empty_like(value) for value in sources];native.v41_copy_decoder_graph_inputs(sources,targets)
         # Integer bit views compare NaN and signed zero without arithmetic.
         for src,dst in zip(sources,targets):
          bitdtype=torch.int16 if src.dtype==dtype else (torch.int32 if src.dtype==torch.float32 else src.dtype)
          assert torch.equal(src.view(bitdtype).cpu(),dst.view(bitdtype).cpu())
         bit_checks.append({'token_dtype':str(token_dtype),'nondefault_stream':nondefault,'six_fields_bitwise_exact':True})
      def random(shape: tuple) -> torch.Tensor:return (torch.randn(shape)*.02).to(device=device,dtype=dtype)
      aw=native.NativeV41AttentionWeights();aw.projection=[random((1536,5120)),torch.ones(1536,device=device,dtype=dtype),random((4096,1536)),random((512,5120)),torch.ones(512,device=device,dtype=dtype),random((1024,4096)),random((5120,1024))];aw.sink=torch.zeros(8,device=device)
      mw=native.NativeV41MoeWeights();mw.shared=[random((2048,5120)),random((2048,5120)),random((5120,2048))];loaded=native.v41_load_w4a8_fixture(args.fixture,torch.empty(0,device=device,dtype=dtype));empty=torch.empty(0,device=device,dtype=dtype);mw.expert=[loaded[0],loaded[2],empty];mw.scale=[loaded[1],loaded[3],empty];mw.gate=random((2,5120));mw.bias=torch.zeros(2,device=device);mw.vision_bias=None
      hc=load_file(args.mhc_fixture,device='cpu');hcweights=[hc['layers.0.'+name].to(device) for name in ['hc_attn_fn','hc_attn_scale','hc_attn_base','hc_ffn_fn','hc_ffn_scale','hc_ffn_base']]
      dw=native.NativeV41DecoderWeights();dw.attention=aw;dw.moe=mw;dw.hyper_connection=hcweights;dw.input_norm=torch.ones(5120,device=device,dtype=dtype);dw.post_attention_norm=dw.input_norm.clone();dw.engram=[]
      def config(graph: bool, fused: bool=False) -> object:
       attention=native.NativeV41AttentionConfig();attention.mode=native.NativeV41AttentionMode.SWA;attention.layer=0;attention.source_layer=-1;attention.ratio=0;attention.heads=8;attention.groups=1;attention.output_rank=1024;attention.window=128;attention.epsilon=1e-6;attention.scale=512**-.5;attention.original_fp8_activation=True
       moe=native.NativeV41MoeConfig();moe.expert_format=native.NativeV41ExpertFormat.W4AFP8_MT;moe.dense=False;moe.original_fp8_activation=True;moe.first_expert=0;moe.topk=2;moe.normalize=True;moe.scale=1.5;moe.limit=10.;moe.w4a8_so_path=args.gemm_so
       cfg=native.NativeV41DecoderConfig();cfg.attention=attention;cfg.moe=moe;cfg.sinkhorn_iterations=20;cfg.hc_epsilon=1e-6;cfg.mhc_prenorm_directory=args.mhc_directory;cfg.decode_ffn_tail_graph=graph;cfg.decode_ffn_graph_fused_inputs=fused;return cfg
      old=native.NativeV41Decoder(config(False),dw);graph=native.NativeV41Decoder(config(True),dw);packed=native.NativeV41Decoder(config(True,True),dw);attn_hc=native.NativeV41HyperConnection(*hcweights[:3],20,1e-6,1e-6,args.mhc_directory);records=[];first_capture_ms=None
      for case,magnitude in enumerate([0.,1.,50.,.1]):
       residual=(torch.randn(1,4,5120)*magnitude).to(device=device,dtype=dtype);carried=torch.softmax(torch.randn(1,4,device=device),-1);_,pre,post,comb=attn_hc.prepare(residual,carried);attn_output=(torch.randn(1,5120)*magnitude).to(device=device,dtype=dtype);ids=torch.tensor([111+case],device=device,dtype=torch.int32);inputs=(attn_output,residual,pre,post,comb,ids)
       expected=tuple(t.cpu().clone() for t in old.forward_tail(*inputs));t=time.monotonic();actual=tuple(x.cpu().clone() for x in graph.forward_tail(*inputs));elapsed=(time.monotonic()-t)*1000
       if first_capture_ms is None:first_capture_ms=elapsed
       errors=[float((x.float()-y.float()).abs().max().item()) for x,y in zip(actual,expected)];assert all(torch.equal(x,y) for x,y in zip(actual,expected)),{'case':case,'errors':errors}
       packed_actual=tuple(x.cpu().clone() for x in packed.forward_tail(*inputs));assert all(torch.equal(x,y) for x,y in zip(packed_actual,expected)),('packed Graph',case)
       records.append({'packed_outputs_exact':True,'case':case,'magnitude':magnitude,'outputs_exact':True,'max_abs_outputs':errors,'call_including_cpu_ms':elapsed})
      samples=[]
      for repeat in range(4):
       funcs={'eager':lambda:old.forward_tail(*inputs),'graph_including_6_input_copies':lambda:graph.forward_tail(*inputs),'graph_including_fused_input_copy':lambda:packed.forward_tail(*inputs)};labels=list(funcs) if repeat%2==0 else list(reversed(funcs));samples.append({label:_time(funcs[label]) for label in labels})
      result={'status':'PASS_CACHE_FREE_DECODER_TAIL_GRAPH_BASIC_MATH','cases':records,'first_capture_call_ms':first_capture_ms,'micro_samples_ms':samples,'eager_ms':statistics.median(r['eager'] for r in samples),'packed_graph_ms':statistics.median(r['graph_including_fused_input_copy'] for r in samples),'packed_copy_bit_checks':bit_checks,'graph_ms':statistics.median(r['graph_including_6_input_copies'] for r in samples),'scope':'Single-rank real HC/two-expert math; TP/EP skipped here. Mixed-dtype IPC Graph already separately checked on8ranks. Full integration still needs owned 1P1D basic and formal tests.','peak_allocated_bytes':torch.musa.max_memory_allocated(),'accuracy_scope':'仅通过基础功能检查，未完成模型精度验收'};args.output.write_text(json.dumps(result,indent=2));logger.info('Decoder tail math checked: eager %.6f Graph %.6f ms, excluding real TP/EP in this probe',result['eager_ms'],result['graph_ms'])


if __name__=='__main__':main()
