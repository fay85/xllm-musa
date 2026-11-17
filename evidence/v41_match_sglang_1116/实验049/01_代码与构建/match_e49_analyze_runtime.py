from pathlib import Path
import base64, collections, io, json, pickle, re, sqlite3, statistics

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('Global loading is disabled')

r = Path('/expt/runs/p49_full_draft_graph_sg438_v1')
phases = {}
requests = []
logs = (r / 'sj10/rank0.log').read_text().splitlines()
for folder in ['formal_client', 'diagnostic_client']:
    for db_path in sorted((r / folder).rglob('benchmark_data.db')):
        phase = db_path.relative_to(r / folder).parts[0]
        with sqlite3.connect(db_path) as db:
            db.row_factory = sqlite3.Row
            for row in db.execute('SELECT * FROM result'):
                messages = SafeUnpickler(io.BytesIO(base64.b64decode(row['response_messages']))).load()
                ids = {x['id'] for x in messages if x.get('id')}
                assert len(ids) == 1
                rid = next(iter(ids))
                phases[rid] = [folder, phase]
                matches = [x for x in logs if 'sequence_terminal' in x and 'request_id: ' + rid + ',' in x]
                assert len(matches) == 1
                keys = ['prompt_tokens', 'generated_tokens', 'speculative_accepted_tokens', 'speculative_proposed_tokens', 'finish_reason']
                terminal = {key: re.search(key + r': ([^,\s]+)', matches[0]).group(1) for key in keys}
                assert row['success'] and row['prompt_tokens'] == row['completion_tokens'] == 256
                assert terminal['generated_tokens'] == '256'
                requests.append({'request_id': rid, 'folder': folder, 'phase': phase, 'terminal': terminal})
out = []
for path in sorted((r / 'runtime_timing').glob('*.jsonl')):
    data = [json.loads(line) for line in path.read_text().splitlines()]
    formal = []
    counts = collections.Counter()
    for sample in data:
        assert len(sample['request_ids']) == 1
        folder, phase = phases[sample['request_ids'][0]]
        assert folder == 'diagnostic_client'
        counts[phase] += 1
        assert all(mark['record_status'] == 0 and mark.get('elapsed_status', 0) == 0 for mark in sample['boundaries'])
        if phase == 'diagnostic':
            formal.append(sample)
    assert formal
    stats = {}
    for mark in formal[0]['boundaries'][1:]:
        key = mark['name']
        values = [next(m['gpu_stream_span_ms'] for m in sample['boundaries'] if m['name'] == key) for sample in formal]
        stats[key] = {'mean_ms': statistics.mean(values), 'median_ms': statistics.median(values)}
    out.append({'pid': data[0]['pid'], 'stream': data[0]['stream'], 'samples_by_phase': dict(counts), 'formal_samples': len(formal), 'phase_stats': stats, 'total_mean_ms': sum(v['mean_ms'] for v in stats.values())})
assert len(out) == 8
result = {'status': 'ALL8_RUNTIME_EVENTS_VALID_AND_HTTP_IDS_MATCHED', 'scope': 'Formal diagnostic samples only; phases are runtime stream spans, not kernel busy time; rank times are not additive', 'ranks': out, 'requests': requests}
(r / 'diagnostic_client/phase_analysis.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'rank0': out[0], 'requests': len(requests)}))
