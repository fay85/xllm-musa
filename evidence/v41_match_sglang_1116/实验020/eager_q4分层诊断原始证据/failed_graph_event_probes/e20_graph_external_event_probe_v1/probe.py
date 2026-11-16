from pathlib import Path
import json,ctypes,re,torch,torch_musa
r=Path('/expt/diagnostics/e20_graph_external_event_probe_v1');torch.set_num_threads(2)
headers=Path('/usr/local/musa/include');matches=[]
for p in headers.glob('*.h'):
 for line in p.read_text(errors='replace').splitlines():
  if re.search(r'^\s*#define\s+musaEventRecordExternal\s+',line):matches.append(line)
assert any(re.search(r'0x0*1\b|\b1\b',s) for s in matches),matches
lib=ctypes.CDLL('libmusart.so');lib.musaEventCreate.argtypes=[ctypes.POINTER(ctypes.c_void_p)];lib.musaEventRecordWithFlags.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint];lib.musaEventElapsedTime.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_void_p];lib.musaEventDestroy.argtypes=[ctypes.c_void_p]
def check(rc):assert rc==0,rc
x=torch.randn(128,128,device='musa');s=torch.musa.Stream();s.wait_stream(torch.musa.current_stream());events=[ctypes.c_void_p() for _ in range(3)]
for e in events:check(lib.musaEventCreate(ctypes.byref(e)))
with torch.musa.stream(s):
 for _ in range(3):y=x@x
 s.synchronize();g=torch.musa.MUSAGraph()
 with torch.musa.graph(g,stream=s):
  check(lib.musaEventRecordWithFlags(events[0],s.musa_stream,1));y=x@x;check(lib.musaEventRecordWithFlags(events[1],s.musa_stream,1))
  for _ in range(5):z=y@x
  check(lib.musaEventRecordWithFlags(events[2],s.musa_stream,1))
 vals=[]
 for _ in range(3):
  g.replay();s.synchronize();a=ctypes.c_float();b=ctypes.c_float();check(lib.musaEventElapsedTime(ctypes.byref(a),events[0],events[1]));check(lib.musaEventElapsedTime(ctypes.byref(b),events[1],events[2]));assert a.value>0 and b.value>0;vals.append({'one_gemm_ms':a.value,'five_gemm_ms':b.value})
 del g
for e in events:check(lib.musaEventDestroy(e))
(r/'result.json').write_text(json.dumps({'status':'PASS','flags_header':matches,'measurements':vals,'scope':'External event nodes inside real MUSA Graph replay'},indent=2));print(json.dumps(vals))
