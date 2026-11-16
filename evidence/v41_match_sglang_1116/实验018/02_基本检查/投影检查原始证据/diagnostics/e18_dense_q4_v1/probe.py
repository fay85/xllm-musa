from pathlib import Path
import torch,torch_musa,json,os,datetime
r=Path('/expt/diagnostics/e18_dense_q4_v1');torch.manual_seed(1801);records=[]
def bench(fn):
 stream=torch.musa.Stream();stream.wait_stream(torch.musa.current_stream())
 with torch.musa.stream(stream):
  for _ in range(5):fn()
  stream.synchronize();g=torch.musa.MUSAGraph()
  with torch.musa.graph(g,stream=stream):
   for _ in range(20):out=fn()
  for _ in range(5):g.replay()
  values=[]
  for _ in range(3):
   start,end=torch.musa.Event(enable_timing=True),torch.musa.Event(enable_timing=True);start.record()
   for _ in range(30):g.replay()
   end.record();end.synchronize();values.append(start.elapsed_time(end)/600)
 return values
for m in [1,4,6,8]:
 for name,k,n1,n2 in [('qkv',5120,1024,512),('shared_tp8',5120,288,288),('shared_full',5120,2304,2304)]:
  x=torch.randn((m,k),device='musa',dtype=torch.bfloat16);a=torch.randn((n1,k),device='musa',dtype=torch.bfloat16);b=torch.randn((n2,k),device='musa',dtype=torch.bfloat16);packed=torch.cat([a,b],0).contiguous()
  sep=lambda:(torch.nn.functional.linear(x,a),torch.nn.functional.linear(x,b))
  fused=lambda:torch.nn.functional.linear(x,packed)
  ref=torch.cat(sep(),-1);out=fused();rel=(out.float()-ref.float()).square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-12)
  records.append({'m':m,'kind':name,'input':[m,k],'outputs':[n1,n2],'separate_graph_ms':bench(sep),'packed_graph_ms':bench(fused),'relative_rms':rel.item()})
 for name,k,n in [('q_b',1024,4096),('o_b',1024,5120),('wo_a',4096,1024),('shared_down_tp8',288,5120)]:
  x=torch.randn((m,k),device='musa',dtype=torch.bfloat16);a=torch.randn((n,k),device='musa',dtype=torch.bfloat16);fn=lambda:torch.nn.functional.linear(x,a);records.append({'m':m,'kind':name,'input':[m,k],'outputs':[n],'single_graph_ms':bench(fn)})
(r/'result.json').write_text(json.dumps({'status':'PASS_DIAGNOSTIC_ONLY','at':datetime.datetime.now().astimezone().isoformat(),'torch':torch.__version__,'device':str(torch.musa.get_device_properties(0)),'records':records,'scope':'Graph microbench with synthetic BF16 carriers; no communication, not whole model Amdahl fraction'},indent=2))
print(json.dumps({'status':'PASS','records':len(records)}))
