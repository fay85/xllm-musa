from pathlib import Path
import json,hashlib,datetime
r=Path('/expt');checks=json.loads((r/'diagnostics/e42_combined_checks.json').read_text());assert checks['all_pass'];artifact=json.loads((r/'builds/e42_native_full_v1/artifact.json').read_text());binary=artifact['artifact'];sha=artifact['sha256'];assert hashlib.sha256(Path(binary).read_bytes()).hexdigest()==sha
records=[]
for mode in ['off','on']:
 run='p42_fp8dense_'+mode+'_v1';out=r/'runs'/run;out.mkdir(exist_ok=False)
 for host in ['sj9','sj10']:
  base=json.loads((r/'runs/p39_static_activation_on_v1'/host/'prepared_config.json').read_text());ns=base['namespace'];cfg=json.loads(json.dumps(base).replace('p39_static_activation_on_v1',run).replace(ns,'xiaopang1117-match-p42-fp8dense-'+mode+'-v1').replace(base['binary'],binary));cfg['binary_sha256']=sha;cfg['source_manifest']=str(r/'builds/e42_native_full_v1/source_manifest.json');cfg['source_dir']=str(r/'builds/e42_native_full_v1/source_after');cfg['source_commit']='b34f1e6d2786198fdc507883238b4f3c0fe09f48 + E42 source_manifest';cfg['artifact_source_note']='New E42 ELF full build; exact changed source snapshot and parent commit recorded; all P settings and every other D setting inherited E39';cfg['treatment']='E42 '+mode+': D target/draft FP8 projections and aligned CPU weight/scale slicing; P unchanged; TP8/EP8 unchanged, not DP/LL'
  for rank in cfg['ranks']:
   env=rank['environment'];assert env['XLLM_DSV41_DECODE_PACKED_QKV']=='0' and env['XLLM_DSV41_DECODE_PACKED_SHARED_GATE_UP']=='0';assert rank['command'][0]==binary
   if mode=='on' and host=='sj10':env['XLLM_DSV41_DENSE_PREQUANT_DIR']='/expt/artifacts/e42_prequant_aot_v1'
   else:assert 'XLLM_DSV41_DENSE_PREQUANT_DIR' not in env
  d=out/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(cfg,indent=2));records.append({'run':run,'host':host,'binary_sha256':sha,'fp8_enabled':mode=='on' and host=='sj10'})
 (out/'experiment.json').write_text(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'role':'CPU_CONFIG_PREPARED_NOT_STARTED','source_champion':'p39_static_activation_on_v1','checks':checks,'artifact':artifact,'test':'Same new ELF OFF then ON; 1warm5 actual256/256, independent profile, long smoke'},indent=2))
s=(r/'scripts/match_e39_formal_client.py').read_text();(r/'scripts/match_e42_formal_client.py').write_text(s.replace('E39 static activation treatment identified in role configs','E42 FP8 projection OFF/ON treatment identified in role configs'))
print(json.dumps(records))
