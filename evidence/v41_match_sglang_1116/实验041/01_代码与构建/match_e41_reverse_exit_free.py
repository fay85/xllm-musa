from pathlib import Path
import json,subprocess,socket,datetime,re
r=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/runs/p41_static_activation_proposal3_reverse_v1');h='sj9' if socket.gethostname()=='worker32007' else 'sj10';v={'at':datetime.datetime.now().astimezone().isoformat(),'host':h}
for n in ['result.json','status.json']:
 p=r/h/n
 if p.exists():v[n]=json.loads(p.read_text())
p=subprocess.run(['timeout','15','mthreads-gmi'],capture_output=True,text=True,timeout=20);v['gmi']=p.stdout;v['gmi_exit']=p.returncode
p=subprocess.run(['docker','top','xiaopang1117_v41_sg438_'+('prefill' if h=='sj9' else 'decode')+'_'+h,'-eo','pid,ppid,stat,args'],capture_output=True,text=True,timeout=10);v['docker_top']=p.stdout
print(json.dumps(v))
