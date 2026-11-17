from pathlib import Path
import json,subprocess,datetime
r=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116');d=r/'diagnostics/e50_production_dispatch_gpu_v1';done=json.loads((d/'finished.json').read_text());assert done['exit_code']==0
p=subprocess.run(['timeout','20','mthreads-gmi'],capture_output=True,text=True,timeout=25);assert p.returncode==0 and 'No running processes found' in p.stdout,p.stdout
v={'at':datetime.datetime.now().astimezone().isoformat(),'gpu_free':True,'gpu_stdout':p.stdout,'host':'sj10','job':json.loads((d/'job.json').read_text()),'exit_code':0};(d/'resource_release.json').write_text(json.dumps(v,indent=2));(r/'00_control/sj10_preflight_p0.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v[k] for k in ['at','gpu_free','host','exit_code']}))
