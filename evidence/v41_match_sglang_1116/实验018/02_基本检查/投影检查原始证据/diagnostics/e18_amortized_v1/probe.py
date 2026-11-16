from pathlib import Path
import torch,torch_musa,json,datetime,statistics
import v41_small_gemm_test_module as ext
r=Path('/expt/diagnostics/e18_amortized_v1');torch.manual_seed(1804);torch.set_num_threads(2)
def bench(fn,nodes,repeats):
 s=torch.musa.Stream();s.wait_stream(torch.musa.current_stream())
 with torch.musa.stream(s):
  for _ in range(5):fn()
  s.synchronize();g=torch.musa.MUSAGraph()
  with torch.musa.graph(g,stream=s):
   for _ in range(nodes):out=fn()
  for _ in range(3):g.replay()
  vals=[]
  for _ in range(3):
   a,b=torch.musa.Event(enable_timing=True),torch.musa.Event(enable_timing=True);a.record()
   for _ in range(repeats):g.replay()
   b.record();b.synchronize();vals.append(a.elapsed_time(b)/(nodes*repeats))
 return vals
records=[]
for name,k,n in [('qa',5120,1024),('kv',5120,512),('qb',1024,4096),('ob',1024,5120),('woa',4096,1024),('shared_up',5120,288),('shared_down',288,5120)]:
 m=4;x=torch.randn(m,k,device='musa',dtype=torch.bfloat16);w=torch.randn(n,k,device='musa',dtype=torch.bfloat16);out=torch.empty(m,n,device='musa',dtype=torch.bfloat16);base=lambda:torch.nn.functional.linear(x,w);cand=lambda:ext.gemm(x,w,out)
 nodeslist=[20,40,200,500] if name=='qa' else [500]
 for nodes in nodeslist:
  record={'name':name,'m':m,'k':k,'n':n,'nodes_per_replay':nodes,'repeats':3,'torch_ms':bench(base,nodes,3),'skinny_ms':bench(cand,nodes,3),'torch_reverse_ms':bench(base,nodes,3)};records.append(record);(r/'partial.json').write_text(json.dumps(records,indent=2))
(r/'result.json').write_text(json.dumps({'status':'COMPLETE_DIAGNOSTIC','at':datetime.datetime.now().astimezone().isoformat(),'records':records,'scope':'Graph repeat amortization control; not full model hotspot fraction'},indent=2));print(json.dumps({'status':'COMPLETE','records':len(records)}))
