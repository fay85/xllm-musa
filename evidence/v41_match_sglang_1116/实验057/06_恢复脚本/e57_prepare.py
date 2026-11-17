from pathlib import Path
import hashlib,json,datetime
root=Path('/expt');selection=json.loads((root/'00_control/e57_selected_baseline.json').read_text())
assert selection['decision_complete'] and selection['run'] in ['p56_short_indexer_off_v1','p56_short_indexer_on_v1','p56_short_indexer_reverse_off_v1']
old=selection['run'];control=root/'runs'/old
assert json.loads((control/'formal_client/finished.json').read_text())['exit_code']==0
assert json.loads((control/'formal_client/compilation_window.json').read_text())['all_clear']
for host in ['sj9','sj10']:assert json.loads((control/host/'result.json').read_text())['exit_codes']==[0]*8
for name,enabled in [('p57_progress_disabled_v1',False),('p57_progress_reverse_enabled_v1',True)]:
 run=root/'runs'/name;run.mkdir(exist_ok=False);(run/'runtime_timing').mkdir()
 for host in ['sj9','sj10']:
  base=json.loads((control/host/'role_config.json').read_text());assert hashlib.sha256(Path(base['binary']).read_bytes()).hexdigest()==base['binary_sha256']
  cfg=json.loads(json.dumps(base).replace(old,name).replace(base['namespace'],'xiaopang1117-match-'+name.replace('_','-')))
  for rank in cfg['ranks']:
   assert rank['environment']['XLLM_DSV41_DIAGNOSTIC_PROGRESS']=='1'
   if not enabled:rank['environment'].pop('XLLM_DSV41_DIAGNOSTIC_PROGRESS')
  cfg['treatment']='E57 progress logging '+('enabled reverse control' if enabled else 'disabled; preserve errors/request logs and communication checks')
  out=run/host;out.mkdir();(out/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
 (run/'experiment.json').write_text(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'selection':selection,'diagnostic_progress_enabled':enabled,'source_modified':False,'reuse_control_formal_only_not_final_acceptance':old},indent=2))
print(json.dumps({'prepared':True,'selected_control':old,'no_gpu_started':True}))
