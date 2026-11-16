# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""DP capacity prerequisite: true checkpoint FP8 prequantization in SG438.
Research microbenchmark only; serving remains native C++ and unchanged.
"""
from pathlib import Path
import json,sys,hashlib,statistics,gc
import torch,torch_musa,triton
import musa_python_kernel_test_module as native
from sglang.kernels.ops.quantization.musa_decode_quant_gemm import _prequantized_decode_gemm_kernel
from sglang.kernels.ops.quantization.fp8_kernel import _reduce_block_fp8_split_k,sglang_per_token_group_quant_fp8
sys.path.insert(0,'/expt/repos/xllm-musa/tools/musa_deepseek_v41')
from validate_dense_block32_aot import _tensor
r=Path('/expt/diagnostics/e40_prequant_real_weights_v1');torch.manual_seed(4017);torch.set_num_threads(2);assert torch.musa.device_count()==1
model=Path('/work/artifacts/model_view_w4afp8_mt_1017');index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];checks=[];bench=[]
def save():
 (r/'partial.json').write_text(json.dumps({'checks':checks,'benchmarks':bench},indent=2))
def load(prefix):
 w=_tensor(model,index,prefix+'.weight');exp=_tensor(model,index,prefix+'.scale').int();assert not exp.eq(255).any();s=torch.where(exp.eq(0),0x00400000,exp<<23).contiguous().view(torch.float32);return w,s

def prequant(x,w,s,splits):
 m,k=x.shape;n=w.shape[0];assert k%(32*splits)==0 and n%32==0
 a,ascale=sglang_per_token_group_quant_fp8(x,32,eps=1e-4,scale_ue8m0=True)
 assert a.is_contiguous() and ascale.is_contiguous() and ascale.dtype==torch.float32
 partials=torch.empty((splits,m,n),device=x.device,dtype=torch.float32);out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
 _prequantized_decode_gemm_kernel[(n//32,splits)](a,w,ascale,s,partials,m,n,k,splits,num_warps=4,num_stages=1)
 _reduce_block_fp8_split_k[(triton.cdiv(m*n,256),)](partials,out,m*n,splits,256)
 return out

def error(a,b):
 assert torch.isfinite(b).all().item();return ((a.float()-b.float()).square().mean()/a.float().square().mean().clamp_min(1e-30)).sqrt().item()

def graph_time(fn,nodes=30):
 stream=torch.musa.Stream();stream.wait_stream(torch.musa.current_stream())
 with torch.musa.stream(stream):
  for _ in range(3):fn()
  stream.synchronize();graph=torch.musa.MUSAGraph()
  with torch.musa.graph(graph,stream=stream):
   for _ in range(nodes):out=fn()
  for _ in range(3):graph.replay()
  samples=[]
  for _ in range(5):
   start=torch.musa.Event(enable_timing=True);end=torch.musa.Event(enable_timing=True);start.record()
   for _ in range(3):graph.replay()
   end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000/(nodes*3))
 return {'nodes':nodes,'samples_us':samples,'median_us':statistics.median(samples)}

specs=[('qa',['layers.0.attn.wq_a'],16),('kv',['layers.0.attn.wkv'],16),('qb_h64',['layers.0.attn.wq_b'],4),('ob_h64',['layers.0.attn.wo_b'],4),('shared_gate_up',['layers.0.ffn.shared_experts.w1','layers.0.ffn.shared_experts.w3'],8),('shared_down',['layers.0.ffn.shared_experts.w2'],8)]
with torch.inference_mode():
 for name,prefixes,split in specs:
  parts=[load(p) for p in prefixes];wc=torch.cat([p[0] for p in parts],0).contiguous() if len(parts)>1 else parts[0][0];sc=torch.cat([p[1] for p in parts],0).contiguous() if len(parts)>1 else parts[0][1]
  w=wc.to('musa');s=sc.to('musa');carrier=(w.float()*s.repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16();ref=native.NativeV41Dense(carrier,True);n,k=w.shape
  for m in [1,4,6]:
   actual_split=4 if name=='shared_gate_up' and m==6 else split
   for tiny in [False,True]:
    for nondefault in [False,True]:
     stream=torch.musa.Stream() if nondefault else torch.musa.current_stream();stream.wait_stream(torch.musa.current_stream())
     with torch.musa.stream(stream):
      x=torch.randn(m,k,device='musa',dtype=torch.bfloat16)*(1e-6 if tiny else 1.);a=ref.forward(x);b=prequant(x,w,s,actual_split)
     stream.synchronize();er=error(a,b);assert er<.01,(name,m,tiny,nondefault,er);checks.append({'name':name,'rows':m,'tiny':tiny,'nondefault_stream':nondefault,'rms':er,'split':actual_split});save()
  for m in [4,6]:
   actual_split=4 if name=='shared_gate_up' and m==6 else split;x=torch.randn(m,k,device='musa',dtype=torch.bfloat16)
   z={'name':name,'rows':m,'weight_shape':[n,k],'split':actual_split,'weight_bytes_bf16':carrier.numel()*carrier.element_size(),'weight_bytes_fp8_scales':w.numel()+s.numel()*s.element_size(),'control':graph_time(lambda:ref.forward(x)),'prequant':graph_time(lambda:prequant(x,w,s,actual_split)),'control_reverse':graph_time(lambda:ref.forward(x))};bench.append(z);save();print(json.dumps(z),flush=True)
  del ref,carrier,w,s,wc,sc,parts,x,a,b;gc.collect();torch.musa.empty_cache()
result={'status':'PASS','checks':checks,'benchmarks':bench,'environment':{'torch':torch.__version__,'torch_musa':torch_musa.__version__,'triton':triton.__version__,'native_module':native.__file__,'native_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()},'scope':'Real layer0 weights, full unsharded shapes, graph event microbenchmark; activation minimum fixed1e-4 to match native; no model integration, not end-to-end or accuracy acceptance'};(r/'result.json').write_text(json.dumps(result,indent=2));print('PASS',len(checks),flush=True)
