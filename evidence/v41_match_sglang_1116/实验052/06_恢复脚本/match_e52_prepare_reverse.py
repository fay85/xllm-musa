from pathlib import Path
import json,datetime
root=Path('/expt');old='p52_rope_table_off_v1';name='p52_rope_table_reverse_off_v1';r=root/'runs'/name;r.mkdir(exist_ok=False);(r/'runtime_timing').mkdir()
for host in ['sj9','sj10']:
 base=json.loads((root/'runs'/old/host/'prepared_config.json').read_text());cfg=json.loads(json.dumps(base).replace(old,name));cfg['treatment']='E52 reverse OFF after ON; same ELF, E50 champion with target Gate SIMT and full draft Graph; D direct RoPE disabled; E51 BF16 bit gather remains off'
 assert cfg['binary_sha256']=='7d2ffa0662fd53f0074d6f29c7a28e0f4db5e057cb1c0ad6febc706bfc74303f'
 for rank in cfg['ranks']:
  env=rank['environment'];assert env['XLLM_DSV41_GATHER_BF16_BITS']=='0';assert env['XLLM_DSV41_DIRECT_ROPE_TABLE']=='0'
  if host=='sj10':assert env['XLLM_DSV41_FULL_DRAFT_GRAPH']=='1'
 d=r/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps({'prepared':name,'at':datetime.datetime.now().astimezone().isoformat(),'started':False}))
