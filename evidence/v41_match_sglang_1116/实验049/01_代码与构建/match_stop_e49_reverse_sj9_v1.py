from pathlib import Path
import json,os,hashlib,datetime
host='sj9';r=Path('/expt/runs/p49_full_draft_reverse_control_v1');d=r/host
assert json.loads((r/'formal_client/finished.json').read_text())['exit_code']==0





cfg=json.loads((d/'role_config.json').read_text());state=json.loads((d/'status.json').read_text());assert state['exit_codes']==[None]*8
records=[];seen={};libs=set()
for rank,pid in enumerate(state['pids']):
 p=Path('/proc')/str(pid);actual=p.joinpath('exe').resolve();key=str(actual)
 if key not in seen:seen[key]=hashlib.sha256(p.joinpath('exe').read_bytes()).hexdigest()
 assert seen[key]==cfg['binary_sha256'];cmd=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode();assert str(cfg['binary']) in cmd
 maps=p.joinpath('maps').read_text();(d/f'rank{rank}_runtime_maps.txt').write_text(maps)
 for line in maps.splitlines():
  fields=line.split()
  if len(fields)>5 and fields[-1].startswith('/') and '.so' in fields[-1]:libs.add(fields[-1])
 records.append({'rank':rank,'pid':pid,'exe':str(actual),'sha256':seen[key],'start_ticks':p.joinpath('stat').read_text().split()[21],'cgroup':p.joinpath('cgroup').read_text()})
librec=[]
for name in sorted(libs):
 if not any(w in name.lower() for w in ['musa','mate','torch','tvm','mccl','mtml','mooncake','deepseek','deep_ep','xllm_native','kineto']):continue
 p=Path(name)
 if not p.is_file():continue
 sha=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):sha.update(b)
 librec.append({'path':name,'bytes':p.stat().st_size,'sha256':sha.hexdigest()})
receipt={'host':host,'at':datetime.datetime.now().astimezone().isoformat(),'ranks':records,'runtime_libraries':librec};(d/'runtime_identity.json').write_text(json.dumps(receipt,indent=2));(d/'stop.request').write_text('E49 reverse formal control complete; normal shutdown before the next candidate.\n');print(json.dumps({'host':host,'verified_ranks':len(records),'libraries':len(librec),'normal_stop_requested':True}))
