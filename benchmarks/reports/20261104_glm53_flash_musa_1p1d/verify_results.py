"""Verify this stopped evaluation snapshot from its raw records (stdlib only)."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
manifest=json.loads((R/'provenance/publication_manifest.json').read_text())
for item in manifest['files']:
    data=(R/item['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==item['published_sha256'],item['path']
summary=json.loads((R/'accuracy/summary.json').read_text())
official={p.stem:json.loads(p.read_text()) for p in (R/'accuracy/raw/official').glob('mathqa-test-*.json')}
local={p.stem:json.loads(p.read_text()) for p in (R/'accuracy/raw/local').glob('mathqa-test-*.json')}
assert len(official)==100 and len(local)==66
assert sum(v.get('answer')==v['ground_truth'] and not v.get('truncated') for v in official.values())==93
pairs=[]
for q,l in local.items():
    if l.get('http_status')!=200:continue
    a=official[q]
    assert a['request']==l['request'] and a['ground_truth']==l['ground_truth'],q
    pairs.append((a,l))
correct=lambda x:x.get('answer') in list('ABCDE') and x['answer']==x['ground_truth'] and not x.get('truncated')
observed={'count':len(pairs),'official_correct':sum(correct(a) for a,l in pairs),'local_correct':sum(correct(l) for a,l in pairs),'matching_answers':sum(l.get('answer')==a.get('answer') and l.get('answer') in list('ABCDE') and not l.get('truncated') for a,l in pairs),'official_prompt_tokens':sum(a['usage']['prompt_tokens'] for a,l in pairs),'local_prompt_tokens':sum(l['usage']['prompt_tokens'] for a,l in pairs),'official_completion_tokens':sum(a['usage']['completion_tokens'] for a,l in pairs),'local_completion_tokens':sum(l['usage']['completion_tokens'] for a,l in pairs)}
for k,v in observed.items():assert summary['paired_same_questions'][k]==v,(k,v)
assert summary['no_full_local100_score'] is True
assert all(x['measured']==0 for x in summary['performance']['groups'])
for side,expected in [('official',['D','D','D']),('local',['B','B','B'])]:
    records=[json.loads(p.read_text()) for p in sorted((R/f'accuracy/raw/discordance_repeats/mathqa-test-0062/{side}').glob('repeat-*.json'))]
    assert [r.get('answer') for r in records]==expected
    assert all(r['request']==official['mathqa-test-0062']['request'] for r in records)
print(json.dumps({'verified':True,'publication_files_checked':len(manifest['files']),'official_full_correct':93,'paired_raw_results':observed,'performance_measured_samples':0,'note':'Partial snapshot only; not a complete local100 evaluation'},ensure_ascii=False,indent=2))
