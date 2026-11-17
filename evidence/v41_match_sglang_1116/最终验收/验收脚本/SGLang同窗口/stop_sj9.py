from pathlib import Path
import subprocess,json,hashlib,os,signal,time,datetime,urllib.request
r=Path('/data/nfs_shared/xiaopang/sglang_v41_reference_20261116');out=r/'B_recheck_20261117'/'runtime'/'prefill';assert json.loads(Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/runs/sglang_B_recheck_v1/performance/summary.json').read_text())['status']=='PASS_BASIC_FUNCTION_AND_EXACT_TOKEN_PERFORMANCE'
c=json.loads(subprocess.check_output(['docker','inspect','034c599262ae8136bc51533e86ec79a2bccb27ef843e7f18269c430002bf00f2'],text=True))[0];assert c['Id']=='034c599262ae8136bc51533e86ec79a2bccb27ef843e7f18269c430002bf00f2' and c['Name']=='/'+'xiaopang1116_sglang_prefill_sj9' and c['Config']['Labels']['task']=='sglang-ab-trace-20261116';rootpid=c['State']['Pid'];records=[]
for line in subprocess.check_output(['docker','top','034c599262ae8136bc51533e86ec79a2bccb27ef843e7f18269c430002bf00f2','-eo','pid,ppid,stat,args'],text=True).splitlines()[1:]:
 parts=line.split(None,3);pid=int(parts[0]);q=Path('/proc')/str(pid)
 if pid==rootpid:continue
 try:
  assert '034c599262ae8136bc51533e86ec79a2bccb27ef843e7f18269c430002bf00f2' in (q/'cgroup').read_text();cmd=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');exe=os.readlink(q/'exe');start=(q/'stat').read_text().rsplit(')',1)[1].split()[19];states=[]
  for t in (q/'task').iterdir():
   try:
    if (t/'stat').read_text().rsplit(')',1)[1].split()[0]=='D':states.append(t.name)
   except FileNotFoundError:pass
  records.append({'pid':pid,'exe':exe,'exe_sha256':hashlib.sha256((q/'exe').read_bytes()).hexdigest(),'start_ticks':start,'cmd':cmd,'D_threads':states})
 except FileNotFoundError:continue
# Actual rank0 loaded libraries, hashed before process exit.
rank0=next((x for x in records if 'sglang::scheduler' in x['cmd'] and 'TP0' in x['cmd']),None);libs=[]
assert rank0 is not None, 'No actual SG rank0 identified; archive before stop'
if rank0:
 q=Path('/proc')/str(rank0['pid']);maps=(q/'maps').read_text();(out/'rank0_actual_maps.txt').write_text(maps)
 paths=sorted(set(line.split()[-1] for line in maps.splitlines() if len(line.split())>=6 and line.split()[-1].startswith('/') and any(key in line.split()[-1] for key in ['libmusa','libmccl','mate/','tvm_ffi','/jit/'])))
 for path in paths:
  f=q/'root'/path.lstrip('/')
  try:
   if f.is_file():libs.append({'path':path,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
  except (OSError,FileNotFoundError):pass
assert libs and any('libmusa' in row['path'] for row in libs), 'Actual runtime library archive missing; do not stop yet'
(out/'rank0_loaded_libraries_sha.json').write_text(json.dumps(libs,indent=2))
try:
 with urllib.request.urlopen('http://127.0.0.1:28586/server_info',timeout=8) as z:(out/'server_info_final.json').write_bytes(z.read())
except Exception as e:(out/'server_info_final_error.txt').write_text(repr(e))
cmd=['docker','exec','-e','MUSA_VISIBLE_DEVICES=-1','xiaopang1116_sglang_prefill_sj9','/opt/venv/bin/python','-I','-c',"import importlib.metadata as m,importlib.util as u,subprocess,json,sys;print(json.dumps({'python':sys.version,'mate_path':u.find_spec('mate').origin,'versions':{n:m.version(n) for n in ['torch','torch_musa','mate','mate-mubin','sglang-router','deep_ep'] if any(d.metadata.get('Name','').lower().replace('_','-')==n.replace('_','-') for d in m.distributions())},'git':subprocess.check_output(['git','rev-parse','HEAD'],cwd='/workspace/sglang',text=True).strip(),'git_status':subprocess.check_output(['git','status','--porcelain'],cwd='/workspace/sglang',text=True)}))"]
z=subprocess.run(cmd,capture_output=True,text=True,timeout=20);(out/'environment_final.json').write_text(json.dumps({'exit':z.returncode,'stdout':z.stdout,'stderr':z.stderr},indent=2))
report={'at':datetime.datetime.now().astimezone().isoformat(),'container':'034c599262ae8136bc51533e86ec79a2bccb27ef843e7f18269c430002bf00f2','case':'B_recheck_20261117','processes':records,'reason':'Same-window frozen B formal performance finished; archive then normal TERM. No SIGKILL/reset/driver action in this script.'}
(out/'stop_precheck.json').write_text(json.dumps(report,indent=2))
if any(x['D_threads'] for x in records):report['status']='D_STATE_FOUND_NO_SIGNALS'
else:
 targets=[x for x in records if '-m sglang.launch_server' in x['cmd'] or '-m sglang_router.launch_router' in x['cmd']]
 for x in targets:
  q=Path('/proc')/str(x['pid'])
  try:
   fd=os.pidfd_open(x['pid']);assert '034c599262ae8136bc51533e86ec79a2bccb27ef843e7f18269c430002bf00f2' in (q/'cgroup').read_text() and (q/'stat').read_text().rsplit(')',1)[1].split()[19]==x['start_ticks'] and hashlib.sha256((q/'exe').read_bytes()).hexdigest()==x['exe_sha256'];signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd);x['TERM_sent']=True
  except FileNotFoundError:x['already_exited']=True
 for _ in range(60):
  alive=[x for x in records if Path('/proc',str(x['pid'])).exists()]
  if not alive:break
  time.sleep(0.25)
 report['remaining_pids']=[x['pid'] for x in records if Path('/proc',str(x['pid'])).exists()];report['status']='NORMAL_EXIT_COMPLETE' if not report['remaining_pids'] else 'WAITING_AFTER_NORMAL_TERM_NO_FORCE'
report['gpu_after']=subprocess.check_output(['mthreads-gmi'],text=True);(out/'stop_result.json').write_text(json.dumps(report,indent=2));print(json.dumps({'host':'sj9','status':report['status'],'remaining_pids':report.get('remaining_pids'),'gpu_rows':[x for x in report['gpu_after'].splitlines() if 'MiB(81920MiB)' in x]}))

