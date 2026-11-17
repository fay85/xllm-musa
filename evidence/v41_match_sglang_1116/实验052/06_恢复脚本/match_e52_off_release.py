from pathlib import Path
import json,subprocess,datetime,socket
host='sj9' if socket.gethostname()=='worker32007' else 'sj10'
r=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116');run=r/'runs/p52_rope_table_off_v1';state=json.loads((run/host/'result.json').read_text());assert state['exit_codes']==[0]*8,state
exec(Path('/tmp/match_preflight.py').read_text()) if False else None
g=subprocess.run(['timeout','20','mthreads-gmi'],capture_output=True,text=True,timeout=25)
assert g.returncode==0 and 'No running processes found' in g.stdout,g.stdout
ports=subprocess.check_output(['ss','-ltnH'],text=True)
v={'at':datetime.datetime.now().astimezone().isoformat(),'host':host,'hostname':socket.gethostname(),'gpu_exit':g.returncode,'gpu_stdout':g.stdout,'gpu_stderr':g.stderr,'gpu_free':True,'ports':ports,'rank_exit_codes':state['exit_codes'],'automatic_kill':state.get('automatic_kill')}
(run/host/'resource_release.json').write_text(json.dumps(v,indent=2));(r/'00_control'/f'{host}_preflight_p0.json').write_text(json.dumps(v,indent=2));print(json.dumps({'host':host,'at':v['at'],'gpu_free':True,'exit_codes':state['exit_codes']}))
