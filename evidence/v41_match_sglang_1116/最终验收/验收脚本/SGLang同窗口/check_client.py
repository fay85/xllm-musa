from pathlib import Path
import json,os,subprocess
r=Path('/expt/runs/sglang_B_recheck_v1')
print(json.dumps({'files':[x.name for x in r.iterdir()],'job':json.loads((r/'job.json').read_text()),'top':subprocess.check_output(['ps','-eo','pid,ppid,stat,etime,args'],text=True),'client_tail':(r/'client.log').read_text()[-2500:],'supervisor':(r/'supervisor.log').read_text()[-2500:],'finished':(r/'finished.json').read_text() if (r/'finished.json').exists() else None}))
