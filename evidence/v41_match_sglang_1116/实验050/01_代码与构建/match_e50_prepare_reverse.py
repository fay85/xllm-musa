from pathlib import Path
import json,datetime
root=Path('/expt');old='p50_gate_simt_off_v1';name='p50_gate_simt_reverse_off_v1';r=root/'runs'/name;r.mkdir(exist_ok=False);(r/'runtime_timing').mkdir()
for host in ['sj9','sj10']:
 base=json.loads((root/'runs'/old/host/'prepared_config.json').read_text());cfg=json.loads(json.dumps(base).replace(old,name));cfg['treatment']='E50 reverse OFF after ON; same ELF, E49 champion full draft Graph; only D target Gate SIMT disabled'
 assert cfg['binary_sha256']=='1b82cf0ded826db867a67ecf4429ad96368936149597a9cc07c431465506ae8a'
 for rank in cfg['ranks']:
  env=rank['environment'];assert env['XLLM_DSV41_GATE_SIMT']=='0'
  if host=='sj10':assert env['XLLM_DSV41_FULL_DRAFT_GRAPH']=='1'
 d=r/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps({'prepared':name,'at':datetime.datetime.now().astimezone().isoformat(),'started':False}))
