from pathlib import Path
import json,datetime
root=Path('/expt');old='p53_rope_qdq_off_v1';name='p53_rope_qdq_reverse_off_v1';r=root/'runs'/name;r.mkdir(exist_ok=False);(r/'runtime_timing').mkdir()
for host in ['sj9','sj10']:
 base=json.loads((root/'runs'/old/host/'prepared_config.json').read_text());cfg=json.loads(json.dumps(base).replace(old,name));cfg['treatment']='E53 reverse OFF after ON; same ELF, E52 champion with target Gate SIMT and full draft Graph; D direct RoPE enabled, fused QDQ disabled; E51 BF16 bit gather remains off'
 assert cfg['binary_sha256']=='b9e480b25f364ee6b7e6b2675ecdf31bfb03039fa9e23909f962d4af96fd5886'
 for rank in cfg['ranks']:
  env=rank['environment'];assert env['XLLM_DSV41_GATHER_BF16_BITS']=='0';assert env['XLLM_DSV41_DIRECT_ROPE_TABLE']==('1' if host=='sj10' else '0');assert env['XLLM_DSV41_FUSED_ROPE_QDQ']=='0'
  if host=='sj10':assert env['XLLM_DSV41_FULL_DRAFT_GRAPH']=='1'
 d=r/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps({'prepared':name,'at':datetime.datetime.now().astimezone().isoformat(),'started':False}))
