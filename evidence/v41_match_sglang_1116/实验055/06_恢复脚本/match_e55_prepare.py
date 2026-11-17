from pathlib import Path
import json,hashlib,datetime
root=Path('/expt');build=root/'builds/e55_native_draft_gate_v1'
selection=json.loads((build/'selected_baseline.json').read_text())
assert selection['decision_complete'] and selection['run'] in ['p54_rope_proposal_off_v1','p54_rope_proposal_on_v1']
old=selection['run'];artifact=json.loads((build/'artifact.json').read_text())
assert artifact['sha256']=='c8a4f4cf1819baad39f30b8ad9962de73bb69653b5a6bd71d75c2d600c7bef2d'
for mode,enabled in [('off_v1',False),('on_v1',True)]:
 name='p55_draft_gate_'+mode;r=root/'runs'/name;r.mkdir(exist_ok=False);(r/'runtime_timing').mkdir()
 for host in ['sj9','sj10']:
  base=json.loads((root/'runs'/old/host/'prepared_config.json').read_text())
  cfg=json.loads(json.dumps(base).replace(old,name).replace(base['namespace'],'xiaopang1117-match-'+name.replace('_','-')).replace(base['binary'],artifact['artifact']))
  cfg.update(binary_sha256=artifact['sha256'],source_manifest=str(build/'source_manifest.json'),source_dir=str(build/'source_after'),source_commit='980d9da + E55 source_manifest',treatment='E55 default-off V4 draft Gate SIMT; same selected proposal')
  for rank in cfg['ranks']:
   rank['environment']['XLLM_DSV41_DRAFT_GATE_SIMT']='1' if enabled and host=='sj10' else '0'
   assert rank['environment']['XLLM_DSV41_DIRECT_ROPE_TABLE']==('1' if host=='sj10' else '0')
  d=r/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
 (r/'experiment.json').write_text(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'selection':selection,'enabled':enabled},indent=2))
client=root/'scripts/match_e55_client_v1.py';assert not client.exists()
s=(root/'scripts/match_e54_client_v1.py').read_text()
start=s.index("identity={'source_commit':");end=s.index(",'binary_sha256':",start)
s=s[:start]+"identity={'source_commit':json.loads((run/'sj10/role_config.json').read_text())['source_commit'],'candidate_delta_manifest':json.loads((run/'sj10/role_config.json').read_text())['source_manifest']"+s[end:]
client.write_text(s)
print(json.dumps({'prepared':True,'selection':selection,'started':False}))
