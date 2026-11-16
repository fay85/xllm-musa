from pathlib import Path
import subprocess,json,datetime,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
r=Path('/expt/diagnostics/e34_verify_swa_read_v1')
with (r/'probe.log').open('wb') as f:
 p=subprocess.Popen(['python3',str(r/'validate_verify_swa_read.py'),'--output',str(r),'--sparse-library','/work/artifacts/native_stage_i/lib/mate/libdeepseek_v41_sparse_h8_bf16.so'],stdout=f,stderr=subprocess.STDOUT);(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]}));rc=p.wait()
(r/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
