from pathlib import Path
import datetime,json
root=Path('/expt');old='p56_short_indexer_off_v1';name='p56_short_indexer_reverse_off_v1'
run=root/'runs'/name;run.mkdir(exist_ok=False);(run/'runtime_timing').mkdir()
for host in ['sj9','sj10']:
    cfg=json.loads((root/'runs'/old/host/'prepared_config.json').read_text())
    cfg=json.loads(json.dumps(cfg).replace(old,name))
    cfg['treatment']='E56 reverse OFF, exact same artifact and champion configuration'
    assert all(x['environment']['XLLM_DSV41_SHORT_CONTEXT_INDEXER']=='0' for x in cfg['ranks'])
    out=run/host;out.mkdir();(out/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps({'prepared':name,'at':datetime.datetime.now().astimezone().isoformat(),'gpu_started':False}))
