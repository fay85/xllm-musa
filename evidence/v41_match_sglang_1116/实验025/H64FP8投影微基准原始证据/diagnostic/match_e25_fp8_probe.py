from pathlib import Path
import json,statistics,torch,torch_musa
import v41_q4_projection_test_module as ext
from match_e25_fp8_reference import fused_decode_quant_gemm
r=Path('/expt/diagnostics/e25_h64_fp8_projection_v1');torch.manual_seed(1901);torch.set_num_threads(2);checks=[];bench=[]
def save():
 (r/'partial.json').write_text(json.dumps({'checks':checks,'benchmarks':bench},indent=2))
def graph_time(fn,nodes=50):
 s=torch.musa.Stream();s.wait_stream(torch.musa.current_stream())
 with torch.musa.stream(s):
  for _ in range(3):fn()
  s.synchronize();g=torch.musa.MUSAGraph()
  with torch.musa.graph(g,stream=s):
   for _ in range(nodes):y=fn()
  for _ in range(3):g.replay()
  samples=[]
  for _ in range(3):
   a=torch.musa.Event(enable_timing=True);b=torch.musa.Event(enable_timing=True);a.record()
   for _ in range(3):g.replay()
   b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/(nodes*3))
 return {'nodes':nodes,'us':samples,'median_us':statistics.median(samples)}
for name,n,k in [('qb_h64',32768,1280),('ob_h64',5120,8192)]:
 wf=(torch.randn(n,k,device='musa')*16).to(torch.float8_e4m3fn)
 scale=torch.pow(2.,torch.randint(-9,-5,(n//32,k//32),device='musa')).float()
 w=(wf.float()*scale.repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16()
 off=ext.Dense(w,True,False);champ=ext.Dense(w,True,True)
 for m in [1,4,6]:
  for tiny in [False,True]:
   for otherstream in [False,True]:
    s=torch.musa.Stream() if otherstream else torch.musa.current_stream();s.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(s):
     x=torch.randn(m,k,device='musa',dtype=torch.bfloat16)*(1e-6 if tiny else 1.)
     a=off.forward(x);b=fused_decode_quant_gemm(x,wf,scale)
    s.synchronize();er=((a.float()-b.float()).square().mean()/a.float().square().mean().clamp_min(1e-30)).sqrt().item()
    assert torch.isfinite(b).all().item() and er<.01,(name,m,tiny,otherstream,er)
    checks.append({'projection':name,'m':m,'tiny':tiny,'nondefault_stream':otherstream,'rms':er});save()
 x=torch.randn(4,k,device='musa',dtype=torch.bfloat16)
 splits_list=[1,2,4,8] if k==1280 else [4,8,16,32]
 timings={};reference=off.forward(x)
 for splits in splits_list:
  observed=fused_decode_quant_gemm(x,wf,scale,split_count=splits)
  er=((reference.float()-observed.float()).square().mean()/reference.float().square().mean().clamp_min(1e-30)).sqrt().item()
  assert torch.isfinite(observed).all().item() and er<.01,(name,splits,er)
  timings[str(splits)]={'rms':er,**graph_time(lambda:fused_decode_quant_gemm(x,wf,scale,split_count=splits))}
 z={'projection':name,'m':4,'n':n,'k':k,'original':graph_time(lambda:off.forward(x)),'fp8_splits':timings,'original_reverse':graph_time(lambda:off.forward(x))};bench.append(z);save();print(json.dumps(z),flush=True)
(r/'result.json').write_text(json.dumps({'status':'PASS','checks':checks,'benchmarks':bench,'scope':'Isolated SG FP8 block32 projection, activation minimum preserved1e-4; no model deployment yet'},indent=2));print('PASS',len(checks),flush=True)
