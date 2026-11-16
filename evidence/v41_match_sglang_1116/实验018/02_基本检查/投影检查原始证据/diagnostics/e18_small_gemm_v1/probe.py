from pathlib import Path
import torch,torch_musa,json,datetime,statistics
import v41_small_gemm_test_module as ext
r=Path('/expt/diagnostics/e18_small_gemm_v1');torch.manual_seed(1802);torch.set_num_threads(2)
shapes=[('qa',5120,1024),('kv',5120,512),('qkv',5120,1536),('qb',1024,4096),('ob',1024,5120),('woa',4096,1024),('shared_up',5120,288),('shared_down',288,5120)]
checks=[];benches=[];graphs=[]
def error(y,ref):
 a=y.cpu().double();b=ref.cpu().double();return float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-10))
def bench(fn):
 s=torch.musa.Stream();s.wait_stream(torch.musa.current_stream())
 with torch.musa.stream(s):
  for _ in range(8):fn()
  s.synchronize();g=torch.musa.MUSAGraph()
  with torch.musa.graph(g,stream=s):
   for _ in range(20):out=fn()
  for _ in range(5):g.replay()
  vals=[]
  for _ in range(3):
   a,b=torch.musa.Event(enable_timing=True),torch.musa.Event(enable_timing=True);a.record()
   for _ in range(30):g.replay()
   b.record();b.synchronize();vals.append(a.elapsed_time(b)/600)
 return vals
for name,k,n in shapes:
 for m in range(1,9):
  x=torch.randn(m,k,device='musa',dtype=torch.bfloat16);w=torch.randn(n,k,device='musa',dtype=torch.bfloat16);y=torch.empty(m,n,device='musa',dtype=torch.bfloat16);assert ext.supported(x,w,y)
  ref=x.cpu().double()@w.cpu().double().T
  for nondefault in [False,True]:
   s=torch.musa.Stream() if nondefault else torch.musa.current_stream();s.wait_stream(torch.musa.current_stream())
   with torch.musa.stream(s):ext.gemm(x,w,y)
   s.synchronize();er=error(y,ref);assert er<0.004,(name,m,er);checks.append({'name':name,'m':m,'nondefault':nondefault,'rms_vs_fp64':er})
  if m in [1,4,6,8]:
   base=lambda:torch.nn.functional.linear(x,w);cand=lambda:ext.gemm(x,w,y)
   benches.append({'name':name,'m':m,'k':k,'n':n,'baseline_ms':bench(base),'candidate_ms':bench(cand),'reverse_ms':bench(base),'rms_vs_torch':error(y,base())})
  if m in [4,6]:
   s=torch.musa.Stream();s.wait_stream(torch.musa.current_stream())
   with torch.musa.stream(s):
    ext.gemm(x,w,y);s.synchronize();g=torch.musa.MUSAGraph()
    with torch.musa.graph(g,stream=s):ext.gemm(x,w,y)
    for rep in range(3):
     x.copy_(torch.randn_like(x));g.replay();s.synchronize();ref=x.cpu().double()@w.cpu().double().T;er=error(y,ref);assert er<0.004;graphs.append({'name':name,'m':m,'repeat':rep,'rms':er})
  (r/'partial.json').write_text(json.dumps({'checks':len(checks),'benches':len(benches),'graphs':len(graphs),'last':[name,m]}))
x=torch.empty(9,32,device='musa',dtype=torch.bfloat16);w=torch.empty(512,32,device='musa',dtype=torch.bfloat16);y=torch.empty(9,512,device='musa',dtype=torch.bfloat16);assert not ext.supported(x,w,y)
assert not ext.supported(x[:1].cpu(),w.cpu(),y[:1].cpu())
assert not ext.supported(x[:1].float(),w,y[:1])
assert not ext.supported(x[:1],w[:,::2],y[:1])
assert not ext.supported(x[:0],w,y[:0])
(r/'result.json').write_text(json.dumps({'status':'PASS_ISOLATED_NOT_MODEL','at':datetime.datetime.now().astimezone().isoformat(),'checks':checks,'graph_replays':graphs,'benches':benches,'unsupported_checks':5,'scope':'synthetic BF16 carriers; graph event microbench; no whole model share'},indent=2))
print(json.dumps({'status':'PASS','checks':len(checks),'graph_replays':len(graphs),'benches':len(benches)}))
