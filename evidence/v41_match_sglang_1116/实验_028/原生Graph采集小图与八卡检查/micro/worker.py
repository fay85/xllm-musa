from pathlib import Path
import subprocess,json,datetime,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
r=Path('/expt/diagnostics/e28_native_graph_registry_v1')
with (r/'probe.log').open('wb') as f:
 p=subprocess.Popen(['python3','/expt/builds/native_graph_profile_registry_v1/validate_native_graph_profile.py','--output',str(r)],stdout=f,stderr=subprocess.STDOUT);(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]}));rc=p.wait()
(r/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
