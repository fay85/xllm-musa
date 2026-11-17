"""Inside owned /expt container; prepare final rounds only after candidate closure."""
from pathlib import Path
import hashlib
import json

root = Path('/expt')
selection = json.loads((root / '00_control/final_selection.json').read_text())
assert selection['decision_complete'] and selection['closure_review_complete']
assert selection['run'] in ['p54_rope_proposal_off_v1', 'p54_rope_proposal_on_v1', 'p55_draft_gate_on_v1', 'p56_short_indexer_on_v1', 'p57_progress_disabled_v1']
old = selection['run']
for suffix, length in [('short1', 256), ('short2', 256), ('short3', 256), ('long', 2048)]:
    name = 'final_' + suffix + '_v1'
    run = root / 'runs' / name
    run.mkdir(exist_ok=False)
    (run / 'runtime_timing').mkdir()
    for host in ['sj9', 'sj10']:
        base = json.loads((root / 'runs' / old / host / 'role_config.json').read_text())
        assert hashlib.sha256(Path(base['binary']).read_bytes()).hexdigest() == base['binary_sha256']
        new_namespace = 'xiaopang1117-final-' + suffix + '-v1'
        cfg = json.loads(json.dumps(base).replace(base['namespace'], new_namespace).replace(old, name))
        cfg.update(source_commit=selection['source_commit'], treatment='Final frozen champion; new independent round', run_id=name)
        assert cfg['namespace'] == new_namespace
        out = run / host
        out.mkdir()
        (out / 'prepared_config.json').write_text(json.dumps(cfg, indent=2))
    (run / 'experiment.json').write_text(json.dumps({'selection': selection, 'length': length, 'formal_requests': 5, 'warmup': 1}, indent=2))
print(json.dumps({'prepared': ['final_' + x + '_v1' for x in ['short1', 'short2', 'short3', 'long']], 'gpu_started': False}))

