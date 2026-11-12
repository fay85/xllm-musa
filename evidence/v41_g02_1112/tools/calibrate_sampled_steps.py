# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check aggregate and sampled-step perturbation using real EvalScope ITLs."""
import argparse
import collections
import json
import sqlite3
import statistics
from pathlib import Path


def requests(perf: Path, length: int) -> list[dict]:
    databases = list((perf / f'input{length}_output256/measured').rglob('benchmark_data.db'))
    assert len(databases) == 1
    with sqlite3.connect('file:' + str(databases[0]) + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('select start_time,completed_time,latency,first_chunk_latency,time_per_output_token,completion_tokens,prompt_tokens,inter_token_latencies,success from result order by start_time')]
    assert len(rows) == 5 and all(r['success'] and r['completion_tokens'] == 256 and r['prompt_tokens'] == length for r in rows)
    for r in rows:
        r['itls'] = json.loads(r['inter_token_latencies'])
        r['exact_token_intervals'] = len(r['itls']) == 255
        r['stream_tail_gap_s'] = r['latency'] - r['first_chunk_latency'] - sum(r['itls'])
        assert abs(r['stream_tail_gap_s']) < 0.1, 'Arrival intervals do not reconstruct request latency closely'
    return rows


def assess(before: Path, measured: Path, timer_files: list[Path]) -> dict:
    roots = []
    for path in timer_files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['name'] == 'V41/step' and row['q'] == 1:
                roots.append(row)
    output = {'status': 'CALIBRATION_EVIDENCE_REQUIRES_REVIEW', 'notes': ['Only D-host timers may align with the D-host EvalScope monotonic clock.', 'Per-token mapping only when ITL count=255 in mode and baseline; other rows retain root timing and unknown local overhead.', 'Selected steps and successors are both flagged because a pending sample can flush at the next step.', 'Adjacent tokens may choose different experts; per-index comparisons are diagnostics, not exact causal estimates.'], 'root_count': len(roots), 'cases': {}}
    for length in [256, 2048]:
        off = requests(before, length)
        mode = requests(measured, length)
        per_request = []
        for ri, request in enumerate(mode):
            selected = set()
            matched = []
            for root in roots:
                start = root['begin_ns'] / 1e9
                if not request['start_time'] <= start <= request['completed_time']:
                    continue
                index = root['kv'] - length - 1
                assert 0 <= index < 255, (length, ri, root)
                if not request['exact_token_intervals'] or not all(r['exact_token_intervals'] for r in off):
                    matched.append({'pid': root['pid'], 'step': root['step'], 'kv': root['kv'], 'host_ms': (root['end_ns']-root['begin_ns'])/1e6, 'gpu_stream_ms': root.get('gpu_stream_span_ms'), 'gpu_status': root.get('device_event_status'), 'token_interval_mapping_valid': False, 'reason': 'SSE arrival count differs from native token transitions; per-index overhead is unknown.'})
                    continue
                token_arrival = request['start_time'] + request['first_chunk_latency'] + sum(request['itls'][:index+1])
                assert root['end_ns']/1e9 <= token_arrival + 0.01, 'Host scope/token arrival mapping not validated'
                selected.add(index)
                matched.append({'pid': root['pid'], 'step': root['step'], 'kv': root['kv'], 'itl_index': index, 'host_ms': (root['end_ns']-root['begin_ns'])/1e6, 'gpu_stream_ms': root.get('gpu_stream_span_ms'), 'gpu_status': root.get('device_event_status')})
            affected = selected | {x+1 for x in selected if x+1 < 255}
            comparison = []
            for i in sorted(affected):
                reference = statistics.mean(r['itls'][i] for r in off)
                actual = request['itls'][i]
                comparison.append({'itl_index': i, 'mode_ms': actual*1000, 'off_same_index_mean_ms': reference*1000, 'relative_change': actual/reference-1})
            per_request.append({'request_index':ri, 'matched_roots':matched, 'selected_token_indices':sorted(selected), 'selected_and_following_intervals':comparison})
        comparisons = [x for r in per_request for x in r['selected_and_following_intervals']]
        off_tpot = statistics.mean(r['time_per_output_token'] for r in off)
        mode_tpot = statistics.mean(r['time_per_output_token'] for r in mode)
        output['cases'][str(length)] = {'off_tpot_ms':off_tpot*1000, 'mode_tpot_ms':mode_tpot*1000, 'aggregate_tpot_change':mode_tpot/off_tpot-1, 'requests_with_clock_and_context_matches':sum(bool(x['matched_roots']) for x in per_request), 'flagged_intervals':len(comparisons), 'flagged_intervals_mean_change':statistics.mean(x['relative_change'] for x in comparisons) if comparisons else None, 'flagged_intervals_median_change':statistics.median(x['relative_change'] for x in comparisons) if comparisons else None, 'per_request':per_request, 'gpu_event_errors_in_matched_roots':sum(x['gpu_status'] not in [None,0] for r in per_request for x in r['matched_roots'])}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--mode', type=Path, required=True)
    parser.add_argument('--timers', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(assess(args.off, args.mode, args.timers), ensure_ascii=False, indent=2))
