"""Compare unmodified official outputs and public same-prefix logprobs."""
from pathlib import Path
import argparse
import hashlib
import base64
import io
import json
import pickle
import sqlite3
import statistics

E = Path(__file__).resolve().parents[1]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class _PlainDataOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Only plain own-benchmark list/dict data are expected')


def _content(encoded):
    events = _PlainDataOnly(io.BytesIO(base64.b64decode(encoded))).load()
    assert isinstance(events, list)
    return ''.join(choice.get('delta', {}).get('content') or ''
                   for event in events for choice in event.get('choices', []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-custom', action='store_true')
    args = parser.parse_args()
    roots = {'off': E.parent / 'ToDoList_4/04_性能/t01_off_v1_raw/output', 'custom': E / '04_性能/t01_donly_custom_v1_raw/output'}
    if args.failed_custom:
        roots['custom'] = E / '04_性能/t01_donly_custom_v1_failed_raw/output'
    summaries = {}
    samples = {}
    for mode, root in roots.items():
        gate = json.loads((root / 'basic/answers.json').read_text())
        assert len(gate) == 2 and all(x['passed'] for x in gate)
        assert json.loads((root / 'long_gate/result.json').read_text())['status'] == 'PASS_2048_INPUT_SHORT_ANSWER'
        summaries[mode] = {}
        samples[mode] = json.loads((root / 'prefix_logprobs.json').read_text())
        for length in [256, 2048]:
            directory = root / 'performance' / f'input{length}_output256/measured'
            official = json.loads(next(directory.rglob('benchmark_summary.json')).read_text())
            with sqlite3.connect(next(directory.rglob('benchmark_data.db'))) as connection:
                connection.row_factory = sqlite3.Row
                records = [dict(row) for row in connection.execute('SELECT * FROM result ORDER BY start_time')]
            valid = len(records) == 5 and all(x['success'] == 1 and x['prompt_tokens'] == length and x['completion_tokens'] == 256 for x in records)
            if not valid:
                assert args.failed_custom and mode == 'custom'
                summaries[mode][str(length)] = {'status': 'FAIL_NOT_VALID_PERFORMANCE', 'official_evalscope_summary': official,
                                               'rows': records, 'scope': 'Tool exit zero is not success: HTTP/stream/native counts failed. No TTFT/TPOT or speedup is accepted for this case.'}
                continue
            for row in records:
                assert abs((row['latency'] - row['first_chunk_latency']) / 255 - row['time_per_output_token']) < 1e-9
                row['output_content'] = _content(row['response_messages'])
                row['output_text_sha256'] = _digest(row['output_content'])
            summary = {'status': 'PASS_5_FORMAL_EXACT_NATIVE_COUNTS', 'official_evalscope_summary': official,
                       'TTFT_ms': statistics.mean(x['first_chunk_latency'] for x in records) * 1000,
                       'TPOT_ms': statistics.mean(x['time_per_output_token'] for x in records) * 1000,
                       'E2E_s': statistics.mean(x['latency'] for x in records),
                       'rows': records}
            summary['QPS_from_mean_request_latency'] = 1 / summary['E2E_s']
            summary['output_tps_from_mean_request_latency'] = 256 / summary['E2E_s']
            summaries[mode][str(length)] = summary
    comparisons = {}
    for length in ['256', '2048']:
        before, after = summaries['off'][length], summaries['custom'][length]
        if after['status'] != 'PASS_5_FORMAL_EXACT_NATIVE_COUNTS':
            comparisons[length] = {'status': 'FAILED_NO_VALID_AFTER_COMPARISON', 'promotion_allowed': False}
            continue
        comparisons[length] = {key + '_reduction_percent': 100 * (1 - after[key] / before[key]) for key in ['TTFT_ms', 'TPOT_ms', 'E2E_s']}
        comparisons[length]['output_tps_increase_percent'] = 100 * (after['output_tps_from_mean_request_latency'] / before['output_tps_from_mean_request_latency'] - 1)
        comparisons[length]['formal_request_body_hashes'] = {mode: [_digest(x['request']) for x in summaries[mode][length]['rows']] for mode in roots}
        comparisons[length]['identical_requests'] = comparisons[length]['formal_request_body_hashes']['off'] == comparisons[length]['formal_request_body_hashes']['custom']
        # EvalScope retains full original response_messages, not reconstructed output.
        comparisons[length]['response_messages_hashes'] = {mode: [_digest(x['response_messages']) for x in summaries[mode][length]['rows']] for mode in roots}
        comparisons[length]['actual_output_text_hashes'] = {mode: [x['output_text_sha256'] for x in summaries[mode][length]['rows']] for mode in roots}
        comparisons[length]['same_text_in_paired_formal_requests'] = sum(a['output_text_sha256'] == b['output_text_sha256'] for a,b in zip(before['rows'],after['rows']))
        comparisons[length]['distinct_outputs_within_mode'] = {mode: len(set(comparisons[length]['actual_output_text_hashes'][mode])) for mode in roots}
        assert comparisons[length]['identical_requests']
    prefix_results = []
    for before, after in zip(samples['off'], samples['custom']):
        assert before['input_tokens'] == after['input_tokens'] and before['request'] == after['request']
        row = {'input_tokens': before['input_tokens'], 'request_sha256': _digest(before['request']), 'positions': [],
               'scope': 'Public API top-5 normalized logprobs, not raw/full-vocabulary logits. Compare only positions with identical preceding token IDs.'}
        if before['status'] != 200 or after['status'] != 200:
            row['status'] = 'CAPABILITY_UNAVAILABLE'; prefix_results.append(row); continue
        a = before['response']['choices'][0]['logprobs']['content']
        b = after['response']['choices'][0]['logprobs']['content']
        same = True
        for index, (x, y) in enumerate(zip(a, b)):
            if not same:
                break
            left = {z['token_id']: z['logprob'] for z in x['top_logprobs']}
            right = {z['token_id']: z['logprob'] for z in y['top_logprobs']}
            intersection = left.keys() & right.keys()
            same = x['token_id'] == y['token_id']
            row['positions'].append({'index': index, 'same_prefix': True,
                                     'off_token_id': x['token_id'], 'custom_token_id': y['token_id'],
                                     'same_selected_token': same,
                                     'common_top5_tokens': len(intersection),
                                     'max_abs_logprob_diff_common_top5': max(abs(left[k]-right[k]) for k in intersection) if intersection else None})
        row['all_8_generated_tokens_identical'] = len(a) == len(b) == 8 and [x['token_id'] for x in a] == [x['token_id'] for x in b]
        row['status'] = 'OBSERVED_COMMON_PREFIX_RECORDED'
        prefix_results.append(row)
    (E / '04_性能/ab_summary.json').write_text(json.dumps({'status': 'PARTIAL_FAILED_FULL_AB_NOT_PROMOTABLE' if args.failed_custom else 'FULL_AB_READY_FOR_TRACE_AND_NUMERIC_AUDIT', 'cases': summaries, 'comparisons': comparisons}, ensure_ascii=False, indent=2))
    (E / '03_正确性/common_prefix_logprob_comparison.json').write_text(json.dumps(prefix_results, ensure_ascii=False, indent=2))
    print(json.dumps({'performance': comparisons, 'same_prefix': prefix_results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
