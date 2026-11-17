from pathlib import Path
import hashlib,json,datetime
r=Path('/expt/runs/final_short2_v1')/'sj10';cfg=json.loads((r/'role_config.json').read_text());s=json.loads((r/'status.json').read_text());assert s['exit_codes']==[None]*8 and not (r/'result.json').exists()
seen={};rows=[]
for rank,pid in enumerate(s['pids']):
 p=Path('/proc',str(pid));exe=str((p/'exe').resolve());cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert cfg['namespace'] in cmd and cfg['binary'] in cmd
 if exe not in seen:seen[exe]=hashlib.sha256((p/'exe').read_bytes()).hexdigest()
 assert seen[exe]==cfg['binary_sha256'];rows.append({'rank':rank,'pid':pid,'start_ticks':(p/'stat').read_text().split()[21],'exe':exe,'sha256':seen[exe],'namespace':cfg['namespace']})
v={'at':datetime.datetime.now().astimezone().isoformat(),'host':'sj10','run':'final_short2_v1','ranks':rows};(r/'resume_identity.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
