from pathlib import Path
base=Path('/expt/diagnostics/e42_model_projection_checks_v1/probe.py').read_text();exec(compile(base.split('with torch.inference_mode():')[0],'/expt/diagnostics/e42_model_projection_checks_v1/probe.py','exec'));r=Path('/expt/diagnostics/e42_shared_checks_v1')
with torch.inference_mode():
 for size,rank in [(1,0),(8,0),(8,3),(8,7)]:
  params=[load('layers.0.ffn.shared_experts.'+p,(1 if p=='w2' else 0) if size>1 else None,rank,size) for p in ['w1','w3','w2']];ws=[p[0] for p in params];ss=[p[1] for p in params];cs=[carrier(w,s) for w,s in params];refs=[old.NativeV41Dense(c,True) for c in cs]
  def reference(x):
   gate=refs[0].forward(x).float().clamp(max=10.);up=refs[1].forward(x).float().clamp(-10.,10.);return refs[2].forward((torch.nn.functional.silu(gate)*up).bfloat16())
  for m in [1,4,6,32,65,128]:
   x=torch.randn(m,5120,device='musa',dtype=torch.bfloat16)*.2;a=reference(x);b=new.shared_forward(x,ws,ss,aot);torch.musa.synchronize();e=rms(a,b);assert e<.02,(size,rank,m,e);shared_checks.append({'size':size,'rank':rank,'rows':m,'rms':e});save()
  stream=torch.musa.Stream();stream.wait_stream(torch.musa.current_stream())
  with torch.musa.stream(stream):
   x=torch.randn(4,5120,device='musa',dtype=torch.bfloat16)
   for _ in range(2):new.shared_forward(x,ws,ss,aot)
   stream.synchronize();g=torch.musa.MUSAGraph()
   with torch.musa.graph(g,stream=stream):out=new.shared_forward(x,ws,ss,aot)
   for amp in [.2,1e-6,.7]:
    x.copy_(torch.randn_like(x)*amp);g.replay();a=reference(x);stream.synchronize();e=rms(a,out);assert e<.02;graph_checks.append({'size':size,'rank':rank,'amplitude':amp,'rms':e});save()
  del refs,params,ws,ss,cs,x,a,b,g,out;gc.collect();torch.musa.empty_cache()
z={'status':'PASS','cpu':cpu_checks,'shared':shared_checks,'graph':graph_checks,'module':new.__file__,'module_sha256':hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest(),'reference':'Previous native BF16 Dense projections, torch float32 clamp/silu matching native SwiGLU; no full-model accuracy conclusion'};(r/'result.json').write_text(json.dumps(z,indent=2));print(json.dumps({k:len(z[k]) for k in ['cpu','graph','shared']}),flush=True)
