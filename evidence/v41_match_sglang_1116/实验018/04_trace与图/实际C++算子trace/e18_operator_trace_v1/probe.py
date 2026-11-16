from pathlib import Path
import json, torch, torch_musa
import v41_q4_projection_test_module as ext
r=Path('/expt/diagnostics/e18_operator_trace_v1');torch.manual_seed(1810);torch.set_num_threads(2)
x=torch.randn(4,5120,device='musa',dtype=torch.bfloat16)
kv=torch.randn(512,5120,device='musa',dtype=torch.bfloat16)/64
gate=torch.randn(288,5120,device='musa',dtype=torch.bfloat16)/64
up=torch.randn_like(gate)/64;down=torch.randn(5120,288,device='musa',dtype=torch.bfloat16)/16
results=[]
for name,enabled in [('off',False),('on',True)]:
 dense=ext.Dense(kv,True,enabled);shared=ext.Shared(gate,up,down,10.,True,True,enabled)
 for _ in range(3):dense.forward(x);shared.forward(x)
 torch.musa.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.PrivateUse1],record_shapes=True) as prof:
  for i in range(3):
   with torch.profiler.record_function(f'E18/{name}/kv_shared/{i}'):
    dense.forward(x);shared.forward(x)
  torch.musa.synchronize()
 path=r/f'{name}.pt.trace.json';prof.export_chrome_trace(str(path))
 ev=json.loads(path.read_text())['traceEvents'];kernels=[v for v in ev if 'kernel' in v.get('cat','').lower()]
 counts={}
 for v in kernels:counts[v['name']]=counts.get(v['name'],0)+1
 z={'variant':name,'gpu_kernels':len(kernels),'aten_linear':sum(v.get('name')=='aten::linear' for v in ev),'skinny_kernels':sum('skinny_gemm_kernel' in v['name'] for v in kernels),'kernel_counts':counts};assert kernels;results.append(z)
assert results[0]['skinny_kernels']==0 and results[1]['skinny_kernels']==9
assert results[0]['aten_linear']-results[1]['aten_linear']==9
(r/'result.json').write_text(json.dumps({'status':'PASS','scope':'Isolated actual production Dense+Shared, q4, QDQ retained; not full Graph timings or kernel counts','results':results},indent=2));print(json.dumps(results))
