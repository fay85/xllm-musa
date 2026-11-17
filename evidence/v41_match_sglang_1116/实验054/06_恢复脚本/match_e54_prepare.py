from pathlib import Path
import json,datetime
root=Path('/expt');old='p52_rope_table_on_v1'
for mode,proposal in [('off_v1',3),('on_v1',5)]:
 name='p54_rope_proposal_'+mode;r=root/'runs'/name;r.mkdir(exist_ok=False);(r/'runtime_timing').mkdir()
 for host in ['sj9','sj10']:
  base=json.loads((root/'runs'/old/host/'prepared_config.json').read_text())
  cfg=json.loads(json.dumps(base).replace(old,name).replace(base['namespace'],'xiaopang1117-match-'+name.replace('_','-')))
  assert cfg['binary_sha256']=='7d2ffa0662fd53f0074d6f29c7a28e0f4db5e057cb1c0ad6febc706bfc74303f'
  for rank in cfg['ranks']:
   assert rank['environment']['XLLM_DSV41_DIRECT_ROPE_TABLE']==('1' if host=='sj10' else '0')
   assert rank['environment']['XLLM_DSV41_GATHER_BF16_BITS']=='0'
   assert '--num_speculative_tokens=3' in rank['command']
   rank['command']=[a.replace('--num_speculative_tokens=3','--num_speculative_tokens='+str(proposal)) for a in rank['command']]
  cfg['treatment']='E54 E52 champion ELF; only proposal3 vs proposal5; prior E41/E45 negatives retained'
  cfg['source_commit']='980d9da1287ba99a12749d89b89c55e0d8ed233f'
  d=r/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
 (r/'experiment.json').write_text(json.dumps({'prepared_at':datetime.datetime.now().astimezone().isoformat(),'proposal':proposal,'status':'PREPARED_NOT_STARTED','source':'E52 champion, no source or environment change'},indent=2))
client=root/'scripts/match_e54_client_v1.py'
assert not client.exists()
s=(root/'scripts/match_e52_client_v1.py').read_text().replace('d1963124','980d9da').replace('E52','E54')
client.write_text(s)
print(json.dumps({'prepared':['p54_rope_proposal_off_v1','p54_rope_proposal_on_v1'],'started':False}))
