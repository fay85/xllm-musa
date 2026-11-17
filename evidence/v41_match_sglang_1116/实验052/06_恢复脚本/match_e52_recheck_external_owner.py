from pathlib import Path
import subprocess,json,datetime
name="zhangyu_sgl13_single"
z=subprocess.run(["docker","inspect","--format","{{.Id}} {{.State.Running}} {{.State.Pid}}",name],capture_output=True,text=True,timeout=10)
v={"at":datetime.datetime.now().astimezone().isoformat(),"container":name,"inspect_exit":z.returncode,"identity":z.stdout.strip(),"processes":[]}
for pid in [5583,*range(21207,21216)]:
 p=Path("/proc")/str(pid)
 if not p.exists():continue
 try:
  row={"pid":pid,"comm":(p/"comm").read_text().strip(),"start_ticks":(p/"stat").read_text().split()[21],"cgroup":(p/"cgroup").read_text().strip(),"exe":str((p/"exe").resolve(strict=True))}
  v["processes"].append(row)
 except FileNotFoundError:pass
out=Path("/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/diagnostics/e52_external_owner_recheck_1445.json");assert not out.exists();out.write_text(json.dumps(v,indent=2))
print(json.dumps(v))
