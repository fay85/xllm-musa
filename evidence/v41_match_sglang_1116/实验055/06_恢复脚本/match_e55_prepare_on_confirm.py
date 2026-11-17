from pathlib import Path
import json,datetime
root=Path('/expt');old='p55_draft_gate_on_v1';name='p55_draft_gate_on_confirm_v1';r=root/'runs'/name;r.mkdir(exist_ok=False);(r/'runtime_timing').mkdir()
for host in ['sj9','sj10']:
 cfg=json.loads(json.dumps(json.loads((root/'runs'/old/host/'role_config.json').read_text())).replace(old,name))
 assert cfg['binary_sha256']=='c8a4f4cf1819baad39f30b8ad9962de73bb69653b5a6bd71d75c2d600c7bef2d'
 assert all(rank['environment']['XLLM_DSV41_DRAFT_GATE_SIMT']==('1' if host=='sj10' else '0') for rank in cfg['ranks'])
 cfg['treatment']='E55 second ON confirmation after OFF2 nearly matches ON1; retain every round'
 d=r/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps({'prepared':name,'at':datetime.datetime.now().astimezone().isoformat(),'started':False}))
