from pathlib import Path
import json, hashlib, tarfile, datetime
r = Path('/expt/runs/p56_short_indexer_reverse_off_v1')
assert json.loads((r/'formal_client/finished.json').read_text())['exit_code'] == 0
for host in ['sj9', 'sj10']:
    assert json.loads((r/host/'result.json').read_text())['exit_codes'] == [0] * 8
    assert json.loads((r/host/'resource_release.json').read_text())['gpu_free']
p = Path('/expt/exports/p56_short_indexer_reverse_off_v1_evidence.tar.gz')
with tarfile.open(p, 'w:gz', compresslevel=3) as t:
    t.add(r/'formal_client', arcname='formal_client')
    for host in ['sj9', 'sj10']:
        for name in ['role_config.json', 'runtime_identity.json', 'status.json', 'result.json', 'rank0.log', 'supervisor_job.json', 'resource_release.json']:
            t.add(r/host/name, arcname='roles/'+host+'/'+name)
v = {'archive': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'at': datetime.datetime.now().astimezone().isoformat()}
p.with_suffix('.receipt.json').write_text(json.dumps(v, indent=2))
print(json.dumps(v))
