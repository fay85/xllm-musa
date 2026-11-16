from pathlib import Path
import json,torch,torch_musa
r=Path('/expt/diagnostics/e20_graph_event_probe_v1');torch.set_num_threads(2);x=torch.randn(128,128,device='musa');s=torch.musa.Stream();s.wait_stream(torch.musa.current_stream())
with torch.musa.stream(s):
 for _ in range(3):y=x@x
 s.synchronize();events=[torch.musa.Event(enable_timing=True) for _ in range(3)]
 for a in events:a.record()
 s.synchronize();g=torch.musa.MUSAGraph()
 with torch.musa.graph(g,stream=s):
  events[0].record();y=x@x;events[1].record()
  for _ in range(5):z=y@x
  events[2].record()
 vals=[]
 for _ in range(3):
  g.replay();s.synchronize();a=events[0].elapsed_time(events[1]);b=events[1].elapsed_time(events[2]);assert a>0 and b>0;vals.append({'one_gemm_ms':a,'five_gemm_ms':b})
(r/'result.json').write_text(json.dumps({'status':'PASS','measurements':vals,'scope':'Event nodes inside real MUSA Graph replay, same stream; no model change'},indent=2));print(json.dumps(vals))
