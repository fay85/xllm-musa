"""Run a task-owned command over the documented SSH connection."""
from pathlib import Path
import sys,json,argparse
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE.parents[2]/'sj9_12_登陆工具'))
import sj_access
p=argparse.ArgumentParser();p.add_argument('host');p.add_argument('--timeout',type=int,default=45);p.add_argument('--save');p.add_argument('command');a=p.parse_args()
with sj_access.ssh_client(a.host) as c:r=sj_access.execute(c,a.host,a.command,root=True,timeout=a.timeout)
if a.save:(BASE/'证据'/a.save).write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(r['stdout']);print(r['stderr'],file=sys.stderr);sys.exit(r['exit_status'])
