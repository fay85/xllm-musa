from pathlib import Path
import json,hashlib
r=Path('/expt');old='p37_sg438_native_mccl_v2';sha='6468707445ac26b74f6e8766dd20b13b5567e367a3d172d482aff777b0d09293';static=r/'builds/e38_static_mate_v1/lib/libdeepseek_v41_w4a8_static.so';assert hashlib.sha256(static.read_bytes()).hexdigest()=='9a23ce3a6dfe2381e20573ae17643fba3d22760a79dd1e390bb5ab4531a36241';assert json.loads((r/'diagnostics/e38_deepep_w4a8_v3/finished.json').read_text())['all_pass']
for mode in ['off','on']:
 run=f'p39_static_activation_{mode}_v1';ns=f'xiaopang1117-match-p39-static-{mode}-v1'
 for h in ['sj9','sj10']:
  p=r/'runs'/run/h;p.mkdir(parents=True,exist_ok=False);cfg=json.loads((r/'runs'/old/h/'prepared_config.json').read_text());oldns=cfg['namespace'];assert cfg['binary_sha256']==sha;assert hashlib.sha256(Path(cfg['binary']).read_bytes()).hexdigest()==sha
  cfg.update(namespace=ns,run_id=run,status='PREPARED_NOT_LAUNCHED',source_commit='0130ef9a7f330fa0c2897053b4e6bfea16befa40',treatment='E39: same E37 ELF/environment/protocol; only D target+draft static checkpoint activation scale and matching MATE Apertoken library ON; P unchanged group128. No DeepEP/DP changes enabled.')
  for rank in cfg['ranks']:
   rank['command']=[x.replace(old,run).replace(oldns,ns) for x in rank['command']]
   for k,v in list(rank['environment'].items()):rank['environment'][k]=v.replace(old,run)
   enabled=mode=='on' and h=='sj10';rank['environment']['XLLM_DSV41_W4A8_STATIC_ACTIVATION']='1' if enabled else '0'
   if enabled:rank['environment']['XLLM_DSV41_MATE_W4A8_SO']=str(static)
  (p/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
s=(r/'scripts/match_e37_formal_client.py').read_text();s=s.replace('52dd25cc local E36 compatibility plus E37 native synchronous MCCL; unchanged E34 model settings','0130ef9a E37 native SG environment control; E39 static activation treatment identified in role configs');(r/'scripts/match_e39_formal_client.py').write_text(s)
print(json.dumps({'runs':['p39_static_activation_off_v1','p39_static_activation_on_v1'],'binary_sha256':sha,'static_library_sha256':hashlib.sha256(static.read_bytes()).hexdigest(),'client_script_tail':s[-1800:]}))
