from pathlib import Path
import base64
import collections
import hashlib
import io
import json
import pickle
import re
import sqlite3
import statistics

W = Path.cwd()
O = W / '王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它'
E = O / '02_实验记录/实验_054'

class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Serialized EvalScope response must contain primitives only')

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--modes',default='off_v1,on_v1,reverse_off_v1')
args=parser.parse_args()
requested_modes=args.modes.split(',')
labels={'off_v1':'OFF1','on_v1':'ON1','reverse_off_v1':'OFF2'}
groups = []
for mode in requested_modes:
    name=labels[mode]
    root = E / ('原始证据_' + mode)
    summary = json.loads((root / 'formal_client/summary.json').read_text())
    db_path = next((root / 'formal_client/formal').rglob('benchmark_data.db'))
    server = (root / 'roles/sj10/rank0.log').read_text()
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        values = [dict(x) for x in db.execute('SELECT * FROM result')]
    assert len(values) == 5
    checked = []
    for row in values:
        assert row['success'] == 1 and row['prompt_tokens'] == row['completion_tokens'] == 256
        chunks = PrimitiveUnpickler(io.BytesIO(base64.b64decode(row['response_messages']))).load()
        ids = {chunk['id'] for chunk in chunks if chunk.get('id')}
        assert len(ids) == 1
        response_id = next(iter(ids))
        content = ''.join(choice.get('delta', {}).get('content') or ''
                          for chunk in chunks for choice in chunk.get('choices', []))
        usage = next((chunk['usage'] for chunk in reversed(chunks) if chunk.get('usage')), None)
        if usage is not None:
            assert usage['prompt_tokens'] == usage['completion_tokens'] == 256
            assert usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0
        lines = [line for line in server.splitlines()
                 if 'sequence_terminal' in line and 'request_id: ' + response_id + ',' in line]
        assert len(lines) == 1
        line = lines[0]
        assert 'finish_reason: length, prompt_tokens: 256, generated_tokens: 256' in line
        assert 'num_prefix_cache_tokens: 0' in line
        acceptance = re.search(r'speculative_accepted_tokens: (\d+), speculative_proposed_tokens: (\d+)', line)
        assert acceptance
        tpot = (row['latency'] - row['first_chunk_latency']) * 1000 / 255
        assert abs(tpot - row['time_per_output_token'] * 1000) < 1e-8
        checked.append({
            'response_id': response_id,
            'latency_s': row['latency'], 'first_chunk_latency_s': row['first_chunk_latency'],
            'recomputed_tpot_ms': tpot, 'success': row['success'], 'sse_usage_if_present': usage,
            'actual_server_prompt_tokens': 256, 'actual_server_generated_tokens': 256,
            'actual_server_prefix_cached_tokens': 0, 'server_terminal_line': line,
            'content_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'content_characters': len(content),
            'accepted_tokens': int(acceptance[1]), 'proposed_tokens': int(acceptance[2]),
            'proposal_rounds': int(acceptance[2]) / (5 if mode == 'on_v1' else 3),
        })
    assert abs(statistics.mean(x['recomputed_tpot_ms'] for x in checked) - summary['mean_TPOT_ms_per_token']) < 1e-8
    assert abs(1280 / sum(x['latency_s'] for x in checked) - summary['output_tokens_per_latency_sum_second']) < 1e-8
    assert all(x['passed'] for x in json.loads((root / 'formal_client/basic_gate.json').read_text()))
    for host in ['sj9', 'sj10']:
        assert json.loads((root / 'roles' / host / 'result.json').read_text())['exit_codes'] == [0] * 8
        release = root / 'roles' / host / 'resource_release.json'
        if release.exists():
            released = json.loads(release.read_text())
            assert released['gpu_free'] or released.get('own_resources_released', False)
        else:
            receipt = json.loads((root / 'roles' / host / 'resource_release_receipt.json').read_text())
            assert receipt['exit_status'] == 0 and json.loads(receipt['stdout'])['gpu_free']
    groups.append({'name': name, 'mode': mode, 'summary': summary, 'requests': checked})

paths = []
for mode in [m for m in requested_modes if m in ['off_v1','on_v1']]:
    root = E / ('原始证据_' + mode)
    validation = json.loads((root / 'trace_analysis/all_rank_validation.json').read_text())
    assert len(validation['ranks']) == 16
    cases = []
    for rank in validation['ranks']:
        if rank['host'] != 'sj10':
            continue
        steps = rank['steps']
        target = [g for g in rank['graph_replays']
                  if any(s['ts'] <= g['launch_ts'] <= s['ts'] + s['dur'] for s in steps)]
        draft = [g for g in rank['graph_replays'] if g not in target]
        def classify(graph):
            counts = graph['top_kernels_by_sum_us']
            return {
                'nodes': graph['kernels'],
                'small_fp32_gemm': sum(x['count'] for x in counts if 'batch_gemm_small_m_nt_tt_kernel<float' in x['name']),
                'bf16_to_fp32_cast': sum(x['count'] for x in counts if 'kernel_unary<__mt_bfloat16' in x['name'] and 'CastOp, float, float' in x['name']),
                'mccl_allgather': sum(x['count'] for x in counts if 'mcclKernel_AllGather' in x['name']),
                'custom_reduce': sum(x['count'] for x in counts if 'custom' in x['name'].lower() or 'cross_device_reduce' in x['name'].lower()),
                'direct_rope': sum(x['count'] for x in counts if 'deepseek_v41_rope_table_kernel' in x['name']),
                'fused_rope_qdq': sum(x['count'] for x in counts if 'deepseek_v41_rope_table_qdq_kernel' in x['name']),
                'old_rope': sum(x['count'] for x in counts if 'deepseek_v4_partial_rope_kernel' in x['name']),
                'index_select_float': sum(x['count'] for x in counts if 'IndexSelectVectorKernel<float' in x['name']),
                'native_gate_gemv': sum(x['count'] for x in counts if 'deepseek_v4_gate_gemv_kernel' in x['name']),
            }
        assert all(classify(g)['direct_rope']==120 for g in target)
        assert all(classify(g)['old_rope']==0 for g in target)
        assert all(classify(g)['direct_rope']==9 for g in draft)
        assert all(classify(g)['fused_rope_qdq']==0 for g in target)
        assert all(classify(g)['fused_rope_qdq']==0 for g in draft)
        cases.append({'pid': rank['pid'], 'target_graphs': [classify(g) for g in target],
                      'draft_graphs': [classify(g) for g in draft]})
    paths.append({'mode': mode, 'rank_details': cases,
                  'profile_requests': json.loads((root / 'trace/requests.json').read_text())})

ons = [g for g in groups if g['name'].startswith('ON')]
offs = [g for g in groups if g['name'].startswith('OFF')]
stable = None if not (ons and offs) else min(g['summary']['output_tokens_per_latency_sum_second'] for g in ons) > max(g['summary']['output_tokens_per_latency_sum_second'] for g in offs)
result = {'groups': groups, 'trace_paths': paths, 'on_faster_than_available_controls': stable, 'available_modes':requested_modes,
          'accuracy_status': '仅通过基础功能检查，未完成模型精度验收',
          'attribution_limit': 'Proposal length changes target row and draft work; compare actual accepted/proposed and committed tokens. Heavy trace timing is disturbed and cannot supply production Amdahl fractions.'}
out = E / '03_性能'
out.mkdir(exist_ok=True)
(out / ('逐请求与实际工作量核验_'+'_'.join(requested_modes)+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'stable_direction': stable, 'groups': [{k: g[k] for k in ['name', 'summary']} for g in groups]}, ensure_ascii=False))
