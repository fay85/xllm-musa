from pathlib import Path
import json,datetime
root=Path('/expt');old='p51_bit_gather_off_v1';name='p51_bit_gather_reverse_off_v1';r=root/'runs'/name;r.mkdir(exist_ok=False);(r/'runtime_timing').mkdir()
for host in ['sj9','sj10']:
 base=json.loads((root/'runs'/old/host/'prepared_config.json').read_text());cfg=json.loads(json.dumps(base).replace(old,name));cfg['treatment']='E51 reverse OFF after ON; same ELF, E50 champion with target Gate SIMT and full draft Graph; only D BF16 bit-gather disabled'
 assert cfg['binary_sha256']=='61a191d34a9719b2792eaa565bb02a30d3d01ede929e4efcc6cc47ca8fe1f3f4'
 for rank in cfg['ranks']:
  env=rank['environment'];assert env['XLLM_DSV41_GATHER_BF16_BITS']=='0'
  if host=='sj10':assert env['XLLM_DSV41_FULL_DRAFT_GRAPH']=='1'
 d=r/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps({'prepared':name,'at':datetime.datetime.now().astimezone().isoformat(),'started':False}))
