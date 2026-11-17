"""Prepare a config-only candidate after E55's real promotion decision."""
from pathlib import Path
import datetime
import hashlib
import json

root=Path('/expt')
selection=json.loads((root/'00_control/e56_selected_baseline.json').read_text())
assert selection['decision_complete'] and selection['run'] in ['p54_rope_proposal_off_v1','p55_draft_gate_on_v1']
old=selection['run']
for mode,enabled in [('off_v1',False),('on_v1',True)]:
    name='p56_short_indexer_'+mode
    run=root/'runs'/name
    run.mkdir(exist_ok=False)
    (run/'runtime_timing').mkdir()
    for host in ['sj9','sj10']:
        base=json.loads((root/'runs'/old/host/'role_config.json').read_text())
        assert hashlib.sha256(Path(base['binary']).read_bytes()).hexdigest()==base['binary_sha256']
        cfg=json.loads(json.dumps(base).replace(old,name).replace(base['namespace'],'xiaopang1117-match-'+name.replace('_','-')))
        cfg.update(treatment='E56 short-context Indexer; inherit selected champion, same ELF')
        for rank in cfg['ranks']:
            assert rank['environment']['XLLM_DSV41_SHORT_CONTEXT_INDEXER']=='0'
            rank['environment']['XLLM_DSV41_SHORT_CONTEXT_INDEXER']='1' if enabled and host=='sj10' else '0'
        out=run/host
        out.mkdir()
        (out/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
    (run/'experiment.json').write_text(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'selection':selection,'enabled':enabled,'source_modified':False},indent=2))
client=root/'scripts/match_e56_client_v1.py'
assert not client.exists()
client.write_text((root/'scripts/match_e55_client_v1.py').read_text())
print(json.dumps({'prepared':True,'selection':selection,'gpu_started':False}))
