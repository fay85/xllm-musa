from pathlib import Path
import torch,torch_musa,json,datetime,traceback
import mate
from mate.gemm import bmm
r=Path('/expt/diagnostics/e18_mate_bf16_v1');torch.manual_seed(1803);torch.set_num_threads(2)
def bench(fn):
 s=torch.musa.Stream();s.wait_stream(torch.musa.current_stream())
 with torch.musa.stream(s):
  for _ in range(8):fn()
  s.synchronize();g=torch.musa.MUSAGraph()
  with torch.musa.graph(g,stream=s):
   for _ in range(40):out=fn()
  for _ in range(5):g.replay()
  vs=[]
  for _ in range(3):
   a,b=torch.musa.Event(enable_timing=True),torch.musa.Event(enable_timing=True);a.record()
   for _ in range(30):g.replay()
   b.record();b.synchronize();vs.append(a.elapsed_time(b)/1200)
 return vs
records=[]
for name,k,n in [('qa',5120,1024),('qb',1024,4096),('woa',4096,1024),('shared_down',288,5120)]:
 m=4;x=torch.randn(1,m,k,device='musa',dtype=torch.bfloat16);w=torch.randn(1,n,k,device='musa',dtype=torch.bfloat16);out=torch.empty(1,m,n,device='musa',dtype=torch.bfloat16);ref=torch.nn.functional.linear(x[0],w[0]).unsqueeze(0);rec={'name':name,'k':k,'n':n,'m':m,'torch_ms':bench(lambda:torch.nn.functional.linear(x[0],w[0]))}
 for backend in ['mudnn','mutlass']:
  try:
   fn=lambda:bmm(x,w,out=out,backend=backend);fn();torch.musa.synchronize();err=((out.float()-ref.float()).square().mean()/ref.float().square().mean().clamp_min(1e-12)).sqrt().item();assert err<.005,err;rec[backend]={'rms':err,'graph_ms':bench(fn)}
  except Exception as ex:rec[backend]={'error':repr(ex),'traceback':traceback.format_exc()[-3000:]}
 records.append(rec);(r/'partial.json').write_text(json.dumps({'mate_file':mate.__file__,'records':records},indent=2))
(r/'result.json').write_text(json.dumps({'status':'DIAGNOSTIC_COMPLETE','mate_file':mate.__file__,'at':datetime.datetime.now().astimezone().isoformat(),'records':records},indent=2));print(json.dumps({'status':'COMPLETE','records':len(records)}))
