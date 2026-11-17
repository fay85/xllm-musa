# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""One normal TERM for verified remaining children after parent exit request."""
from pathlib import Path
import datetime,hashlib,json,os,signal,socket,subprocess,time
host={'worker32007':'sj9','worker32009':'sj10'}[socket.gethostname()]
role='prefill' if host=='sj9' else 'decode'
root=Path('/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/B_recheck_20261117/runtime')/role
prior=json.loads((root/'stop_result.json').read_text());cid=prior['container']
client=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/runs/sglang_B_recheck_v1')
assert json.loads((client/'finished.json').read_text())['exit_code']==0
c=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert c['Name']=='/xiaopang1116_sglang_'+role+'_'+host and c['Config']['Labels']['owner']=='xiaopang'
records=[]
for old in prior['processes']:
 p=Path('/proc',str(old['pid']))
 if not p.exists() or not (p/'exe').exists():continue
 if p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]!=old['start_ticks']:continue
 assert cid in (p/'cgroup').read_text()
 assert hashlib.sha256((p/'exe').read_bytes()).hexdigest()==old['exe_sha256']
 ds=[]
 for t in (p/'task').iterdir():
  try:
   if (t/'stat').read_text().rsplit(')',1)[1].split()[0]=='D':ds.append(t.name)
  except FileNotFoundError:pass
 records.append(dict(old,current_D_threads=ds))
record={'at':datetime.datetime.now().astimezone().isoformat(),'host':host,'container':cid,'records':records,'scope':'Completed own SG benchmark; normal TERM remaining child processes once.'}
(root/'children_term_precheck.json').write_text(json.dumps(record,indent=2))
assert not any(x['current_D_threads'] for x in records),'D state: no further signals'
assert not (root/'children_term_result.json').exists(),'Do not repeat TERM batch'
for old in records:
 if old.get('TERM_sent'):continue
 p=Path('/proc',str(old['pid']))
 try:
  fd=os.pidfd_open(old['pid'])
  assert cid in (p/'cgroup').read_text() and (p/'stat').read_text().rsplit(')',1)[1].split()[19]==old['start_ticks']
  assert hashlib.sha256((p/'exe').read_bytes()).hexdigest()==old['exe_sha256']
  signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd);old['child_TERM_at']=datetime.datetime.now().astimezone().isoformat()
 except FileNotFoundError:old['already_exited']=True
time.sleep(15)
record['remaining']=[x['pid'] for x in records if Path('/proc',str(x['pid'])).exists()]
(root/'children_term_result.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'host':host,'remaining':record['remaining'],'sent_TERM':[x['pid'] for x in records if 'child_TERM_at' in x]}))

