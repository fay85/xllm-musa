from pathlib import Path
import json,datetime,hashlib
r=Path('/expt');old='p39_static_activation_on_v1';new='p41_static_activation_proposal3_reverse_v1';out=r/'runs'/new;assert not out.exists();out.mkdir()
rows=[]
for host in ['sj9','sj10']:
 base=json.loads((r/'runs'/old/host/'prepared_config.json').read_text());ns=base['namespace'];cfg=json.loads(json.dumps(base).replace(old,new).replace(ns,'xiaopang1117-match-p41-static-proposal3-reverse-v1'))
 count=0
 for rank in cfg['ranks']:
  command=rank['command'];changed=[]
  for arg in command:
   if arg=='--num_speculative_tokens=3':arg='--num_speculative_tokens=3';count+=1
   changed.append(arg)
  rank['command']=changed
 assert count==8,(host,count)
 cfg['treatment']='E41 proposal3 reverse control, unchanged E39 ON flags/ELF/static library; required by <3% E2E difference rule'
 cfg['source_commit']='0130ef9a frozen E37 ELF; E39 validated static AOT; reverse unchanged proposal3'
 d=out/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(cfg,indent=2));rows.append({'host':host,'changed_flags':count,'source_binary':cfg['binary'],'sha256':cfg['binary_sha256']})
p=r/'scripts/match_e41_reverse_formal_client.py';assert not p.exists();s=(r/'scripts/match_e39_formal_client.py').read_text().replace('E39 static activation treatment identified in role configs','E41 proposal3 reverse control identified in role configs');p.write_text(s)
(out/'experiment.json').write_text(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'control':'p39_static_activation_on_v1','new_evidence':'E41 proposal5 output33.4738 vs E39 proposal3 34.4465, difference2.82%; reverse control checks temporal drift','rows':rows,'status':'PREPARED_NOT_STARTED'},indent=2));print(json.dumps(rows))
