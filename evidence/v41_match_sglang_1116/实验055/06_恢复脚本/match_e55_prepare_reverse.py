from pathlib import Path
import json,datetime
root=Path('/expt');old='p55_draft_gate_off_v1';name='p55_draft_gate_reverse_off_v1';r=root/'runs'/name;r.mkdir(exist_ok=False);(r/'runtime_timing').mkdir()
for host in ['sj9','sj10']:
 base=json.loads((root/'runs'/old/host/'prepared_config.json').read_text());cfg=json.loads(json.dumps(base).replace(old,name));cfg['treatment']='E55 reverse OFF after ON; same ELF, E52 champion with target Gate SIMT and full draft Graph; D direct RoPE enabled, fused QDQ disabled; E51 BF16 bit gather remains off'
 assert cfg['binary_sha256']=='c8a4f4cf1819baad39f30b8ad9962de73bb69653b5a6bd71d75c2d600c7bef2d'
 for rank in cfg['ranks']:
  env=rank['environment'];assert env['XLLM_DSV41_GATHER_BF16_BITS']=='0';assert env['XLLM_DSV41_DIRECT_ROPE_TABLE']==('1' if host=='sj10' else '0');assert env['XLLM_DSV41_DRAFT_GATE_SIMT']=='0'
  if host=='sj10':assert env['XLLM_DSV41_FULL_DRAFT_GRAPH']=='1'
 d=r/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps({'prepared':name,'at':datetime.datetime.now().astimezone().isoformat(),'started':False}))
