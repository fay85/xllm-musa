from pathlib import Path
import json,tarfile,hashlib
root=Path('/expt');out=root/'exports/e40_native_prequant_checks_v2.tar.gz';summary={}
with tarfile.open(out,'w:gz') as t:
 for name in ['e40_prequant_native_gpu_v2','e40_prequant_native_graph_v2']:
  p=root/'diagnostics'/name;assert json.loads((p/'finished.json').read_text())['exit_code']==0;z=json.loads((p/'result.json').read_text());assert z['status']=='PASS';summary[name]={'checks':len(z['checks']),'max_RMS':max(x['rms'] for x in z['checks'])}
  for f in p.iterdir():
   if f.is_file() and f.suffix in ['.json','.py','.log']:t.add(f,arcname=name+'/'+f.name)
 for name in ['e40_prequant_native_v3','e40_prequant_aot_v2']:
  p=root/'builds'/name
  for f in p.rglob('*'):
   if f.is_file() and 'triton_cache' not in f.parts and f.suffix in ['.json','.cpp','.h','.py','.log']:t.add(f,arcname='builds/'+name+'/'+str(f.relative_to(p)))
 for f in (root/'artifacts/e40_prequant_aot_v2').rglob('*.json'):t.add(f,arcname='aot_manifests/'+str(f.relative_to(root/'artifacts/e40_prequant_aot_v2')))
print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'summary':summary}))
