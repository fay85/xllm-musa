"""Audit independent final runs from original EvalScope DBs and server logs."""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import pickle
import sqlite3
import statistics


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Only primitive benchmark records are accepted')


def audit(root, length):
    summary = json.loads((root / 'formal_client/summary.json').read_text())
    identity = json.loads((root / 'formal_client/identity.json').read_text())
    assert summary['profiler_on'] is False and identity['profiler_on'] is False
    assert json.loads((root / 'formal_client/finished.json').read_text())['exit_code'] == 0
    compilation = json.loads((root / 'formal_client/compilation_window.json').read_text())
    assert compilation['all_clear'], 'Compilation overlap or missing monitoring invalidates final acceptance'
    assert all(x['passed'] for x in json.loads((root / 'formal_client/basic_gate.json').read_text()))
    assert identity['tokenizer_sha256'] == '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf'
    assert identity['requests']['256']['dataset_sha256'] == 'c3d7cb38060a2f81b228b83d8939e1a682480e9ab26a2c5a2df29ca66dadbc51'
    warm_dbs = list((root / 'formal_client/formal_warmup').rglob('benchmark_data.db'))
    assert len(warm_dbs) == 1
    with sqlite3.connect('file:' + str(warm_dbs[0]) + '?mode=ro', uri=True) as db:
        warm = db.execute('SELECT success,prompt_tokens,completion_tokens FROM result').fetchall()
    assert warm == [(1, length, 256)], 'Each independent round needs one successful full-length warmup'
    dbs = list((root / 'formal_client/formal').rglob('benchmark_data.db'))
    assert len(dbs) == 1
    with sqlite3.connect('file:' + str(dbs[0]) + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(x) for x in db.execute('SELECT * FROM result')]
    assert len(rows) == 5
    log = (root / 'roles/sj10/rank0.log').read_text()
    checked = []
    for row in rows:
        assert row['success'] == 1 and row['prompt_tokens'] == length and row['completion_tokens'] == 256
        chunks = PrimitiveUnpickler(io.BytesIO(base64.b64decode(row['response_messages']))).load()
        ids = {x['id'] for x in chunks if x.get('id')}
        assert len(ids) == 1
        rid = next(iter(ids))
        terminal = [x for x in log.splitlines() if 'sequence_terminal' in x and 'request_id: ' + rid + ',' in x]
        assert len(terminal) == 1
        assert f'finish_reason: length, prompt_tokens: {length}, generated_tokens: 256' in terminal[0]
        assert 'num_prefix_cache_tokens: 0' in terminal[0]
        text = ''.join(c.get('delta', {}).get('content') or '' for x in chunks for c in x.get('choices', []))
        tpot = (row['latency'] - row['first_chunk_latency']) * 1000 / 255
        assert abs(tpot - row['time_per_output_token'] * 1000) < 1e-8
        checked.append({'id': rid, 'TTFT_ms': row['first_chunk_latency'] * 1000,
                        'TPOT_ms': tpot, 'E2E_s': row['latency'],
                        'content_sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'terminal': terminal[0]})
    role_identity = {}
    for host in ['sj9', 'sj10']:
        role = root / 'roles' / host
        ended = json.loads((role / 'result.json').read_text())
        assert ended['exit_codes'] == [0] * 8 and not ended.get('automatic_kill', False)
        cfg = json.loads((role / 'role_config.json').read_text())
        runtime = json.loads((role / 'runtime_identity.json').read_text())
        assert len(runtime['ranks']) == 8
        assert all(x['sha256'] == summary['binary_sha256'] for x in runtime['ranks'])
        for record in runtime['ranks']:
            assert record['actual_selected_environment'] == {k: cfg['ranks'][record['rank']]['environment'].get(k) for k in ['XLLM_DSV41_DIAGNOSTIC_PROGRESS', 'XLLM_DSV41_SHORT_CONTEXT_INDEXER', 'XLLM_DSV41_GATE_SIMT', 'XLLM_DSV41_DIRECT_ROPE_TABLE', 'XLLM_DSV41_FULL_DRAFT_GRAPH', 'XLLM_DSV41_DRAFT_GATE_SIMT']}, 'Final actual runtime flag drift'
        # Run namespace/log paths differ by design; compare actual executable,
        # dependency hashes and canonicalized full rank command/environment.
        normalized = json.dumps(cfg['ranks'], sort_keys=True).replace(cfg['namespace'], '<NAMESPACE>')
        for rank in cfg['ranks']:
            for value in rank['environment'].values():
                if isinstance(value, str) and '/runs/' in value:
                    run_name = value.split('/runs/', 1)[1].split('/', 1)[0]
                    normalized = normalized.replace(run_name, '<RUN>')
                if isinstance(value, str) and '/profiling/' in value:
                    run_name = value.split('/profiling/', 1)[1].split('/', 1)[0]
                    normalized = normalized.replace(run_name, '<RUN>')
        role_identity[host] = {
            'binary_sha256': cfg['binary_sha256'],
            'source_commit': cfg['source_commit'],
            'source_manifest': cfg['source_manifest'],
            'rank_configuration': normalized,
            'libraries': sorted((x['path'], x['sha256']) for x in runtime['runtime_libraries']),
        }
    metrics = {'TTFT_ms': statistics.mean(x['TTFT_ms'] for x in checked),
               'TPOT_ms': statistics.mean(x['TPOT_ms'] for x in checked),
               'E2E_s': statistics.mean(x['E2E_s'] for x in checked),
               'output_tokens_s': 1280 / sum(x['E2E_s'] for x in checked)}
    assert abs(metrics['TPOT_ms'] - summary['mean_TPOT_ms_per_token']) < 1e-8
    assert abs(metrics['output_tokens_s'] - summary['output_tokens_per_latency_sum_second']) < 1e-8
    if length == 256:
        assert metrics['TTFT_ms'] <= 990.04 and metrics['TPOT_ms'] <= 21.57 and metrics['output_tokens_s'] >= 39.45, metrics
    return {'root': str(root), 'length': length, 'metrics': metrics,
            'requests': checked, 'identity': role_identity,
            'request_protocol_sha256': hashlib.sha256(json.dumps(identity['requests'], sort_keys=True).encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--short', nargs=3, type=Path, required=True)
    ap.add_argument('--long', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert len({p.resolve() for p in args.short}) == 3
    records = [audit(p, 256) for p in args.short] + [audit(args.long, 2048)]
    identities = [r['identity'] for r in records]
    assert all(v == identities[0] for v in identities), 'Final source/artifact/config/library drift'
    assert len({r['request_protocol_sha256'] for r in records}) == 1, 'Prompt/sampling/token ID drift'
    all_ids = [x['id'] for r in records for x in r['requests']]
    assert len(all_ids) == len(set(all_ids)) == 20, 'Repeated request cannot count as an independent round'
    out = {'status': 'THREE_SHORT_AND_ONE_LONG_FORMAL_PASSED',
           'accuracy_status': '仅通过基础功能检查，未完成模型精度验收',
           'rounds': records, 'scope': 'Still requires same-window SGLang, closure audit and delivery'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({'status': out['status'], 'metrics': [r['metrics'] for r in records]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
