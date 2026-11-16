from pathlib import Path
import subprocess,json,datetime,os
r=Path('/expt/diagnostics/e18_small_gemm_v1')
with (r/'probe.log').open('wb') as f:
 p=subprocess.Popen(['python3',str(r/'probe.py')],stdout=f,stderr=subprocess.STDOUT);(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]}));rc=p.wait()
(r/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
