from pathlib import Path
import json,subprocess,datetime
name='xiaopang1116_v41_match_prefill_sj9';cid='68b4f202f5821bd9b8d819b2b564609145de5f141a7f4c10256bc0e5cba1c984';c=json.loads(subprocess.check_output(['docker','inspect',name]))[0];assert c['Id']==cid and c['State']['Running'];g=subprocess.run(['timeout','20','mthreads-gmi'],capture_output=True,text=True);assert g.returncode==0 and 'No running processes found' in g.stdout
r=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/builds/graph_event_probe');r.mkdir(exist_ok=False);(r/'preflight.txt').write_text(g.stdout)
s='''from pathlib import Path
import torch,torch_musa,json,time,datetime
r=Path('/expt/builds/graph_event_probe');torch.set_num_threads(1);assert torch.musa.device_count()==1
x=torch.randn(256,256,device='musa',dtype=torch.bfloat16);w=torch.randn_like(x);y=torch.empty_like(x);s=torch.musa.Stream();events=[torch.musa.Event(enable_timing=True) for _ in range(4)];graph=torch.musa.MUSAGraph();s.wait_stream(torch.musa.current_stream())
with torch.musa.stream(s):
 for e in events:e.record()
 for _ in range(3):torch.mm(x,w,out=y)
s.synchronize()
with torch.musa.graph(graph,stream=s):
 events[0].record()
 for _ in range(16):torch.mm(x,w,out=y)
 events[1].record()
 for _ in range(32):torch.mm(x,w,out=y)
 events[2].record()
 for _ in range(16):torch.mm(x,w,out=y)
 events[3].record()
rows=[]
for i in range(4):
 with torch.musa.stream(s):graph.replay()
 s.synchronize();spans=[events[j].elapsed_time(events[j+1]) for j in range(3)];total=events[0].elapsed_time(events[3]);assert all(v>0 for v in spans) and abs(sum(spans)-total)<.02
 rows.append({'trial':i,'segment_ms':spans,'total_ms':total})
torch.testing.assert_close(y,x@w)
(r/'results.json').write_text(json.dumps({'status':'PASS_GRAPH_CAPTURE_EVENT_TIMESTAMPS','rows':rows,'scope':'Single-device synthetic event-node probe only; not model profiling or optimization gain','at':datetime.datetime.now().astimezone().isoformat()},indent=2))
''';(r/'probe.py').write_text(s)
p=subprocess.run(['docker','exec','-e','MUSA_VISIBLE_DEVICES=0','-e','LD_PRELOAD=','-e','OMP_NUM_THREADS=1',name,'python3','/expt/builds/graph_event_probe/probe.py'],capture_output=True,text=True,timeout=40);(r/'probe.log').write_text(p.stdout+p.stderr);(r/'finished.json').write_text(json.dumps({'exit_code':p.returncode,'at':datetime.datetime.now().astimezone().isoformat()}));print(p.stdout,p.stderr);print((r/'results.json').read_text() if (r/'results.json').exists() else 'No results');assert p.returncode==0
