from pathlib import Path
import json,gc,hashlib
import torch,torch_musa
import v41_prequant_projection_test_module as new
import musa_python_kernel_test_module as old
import sys
sys.path.insert(0,'/expt/repos/xllm-musa/tools/musa_deepseek_v41')
from validate_dense_block32_aot import _tensor
r=Path('/expt/diagnostics/e42_model_projection_checks_v1');model=Path('/work/artifacts/model_view_w4afp8_mt_1017');index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];aot='/expt/artifacts/e42_prequant_aot_v1';torch.set_num_threads(2);torch.manual_seed(4217);assert torch.musa.device_count()==1
records=[];graph_checks=[];shared_checks=[];cpu_checks=[];timings=[]
def save():
 (r/'partial.json').write_text(json.dumps({'cpu':cpu_checks,'projection':records,'graph':graph_checks,'shared':shared_checks,'timings':timings},indent=2))
def rms(a,b):
 assert torch.isfinite(b).all().item()
 return ((a.float()-b.float()).square().mean()/a.float().square().mean().clamp_min(1e-30)).sqrt().item() if a.numel() else 0.
def load(prefix,dim,rank=0,size=1):
 w=_tensor(model,index,prefix+'.weight');sc=_tensor(model,index,prefix+'.scale');ww,ss=new.prepare_fp8(w,sc,rank,size,dim);ex=sc.view(torch.uint8).int();expected=torch.where(ex.eq(0),0x00400000,ex<<23).contiguous().view(torch.float32);want=w
 if dim is not None:
  width=w.shape[dim]//size;want=want.narrow(dim,rank*width,width);expected=expected.narrow(dim,rank*width//32,width//32)
 assert torch.equal(want.contiguous().view(torch.uint8),ww.view(torch.uint8));assert torch.equal(expected,ss);cpu_checks.append({'prefix':prefix,'dim':dim,'rank':rank,'size':size,'shape':list(ww.shape),'bytes':ww.numel()+ss.numel()*4,'exact':True});save();return ww.to('musa'),ss.to('musa')
def carrier(w,s):return (w.float()*s.repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16()
with torch.inference_mode():
 cases=[('layers.0.attn.wq_a',None,0,1),('layers.0.attn.wkv',None,0,1)]
 for prefix,dim in [('layers.0.attn.wq_b',0),('layers.0.attn.wo_b',1),('layers.0.ffn.shared_experts.w1',0),('layers.0.ffn.shared_experts.w3',0),('layers.0.ffn.shared_experts.w2',1)]:
  cases.append((prefix,None,0,1));cases.extend((prefix,dim,rank,8) for rank in [0,3,7])
 for prefix,dim,rank,size in cases:
  w,s=load(prefix,dim,rank,size);ref=old.NativeV41Dense(carrier(w,s),True);n,k=w.shape
  stream=torch.musa.Stream();stream.wait_stream(torch.musa.current_stream())
  with torch.musa.stream(stream):
   for m in [0,1,4,6,17,32,33,65,128]:
    x=torch.randn(m,k,device='musa',dtype=torch.bfloat16);a=ref.forward(x) if m else torch.empty((0,n),device='musa',dtype=torch.bfloat16);b=new.dense_forward(x,w,s,aot);stream.synchronize();e=rms(a,b);assert e<.01,(prefix,dim,rank,m,e);records.append({'prefix':prefix,'dim':dim,'rank':rank,'rows':m,'rms':e});save()
   x=torch.randn(4,k,device='musa',dtype=torch.bfloat16)
   for _ in range(2):new.dense_forward(x,w,s,aot)
   stream.synchronize();g=torch.musa.MUSAGraph()
   with torch.musa.graph(g,stream=stream):out=new.dense_forward(x,w,s,aot)
   for amp in [1.,1e-6]:
    x.copy_(torch.randn_like(x)*amp);g.replay();a=ref.forward(x);stream.synchronize();e=rms(a,out);assert e<.01;graph_checks.append({'prefix':prefix,'dim':dim,'rank':rank,'amplitude':amp,'rms':e});save()
  del g,out,ref,w,s,x,a,b;gc.collect();torch.musa.empty_cache()
 for size,rank in [(1,0),(8,0),(8,3),(8,7)]:
  params=[load('layers.0.ffn.shared_experts.'+p,(1 if p=='w2' else 0) if size>1 else None,rank,size) for p in ['w1','w3','w2']];ws=[p[0] for p in params];ss=[p[1] for p in params];cs=[carrier(w,s) for w,s in params];ref=old.NativeV41SharedMlp(*cs,10.,True,False)
  for m in [1,4,6,32,65,128]:
   x=torch.randn(m,5120,device='musa',dtype=torch.bfloat16)*.2;a=ref.forward(x);b=new.shared_forward(x,ws,ss,aot);torch.musa.synchronize();e=rms(a,b);assert e<.02,(size,rank,m,e);shared_checks.append({'size':size,'rank':rank,'rows':m,'rms':e});save()
  del ref,params,ws,ss,cs,x,a,b;gc.collect();torch.musa.empty_cache()
z={'status':'PASS','cpu':cpu_checks,'projection':records,'graph':graph_checks,'shared':shared_checks,'module':new.__file__,'module_sha256':hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest(),'scope':'FP8 checkpoint aligned CPU sharding, native Dense and shared GPU/graph; not full-model acceptance'};(r/'result.json').write_text(json.dumps(z,indent=2));print(json.dumps({k:len(z[k]) for k in ['cpu','projection','graph','shared']}),flush=True)
