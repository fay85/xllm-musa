# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze measured scopes; GPU durations stay inclusive until stream proof."""
import argparse
import collections
import json
import statistics
from pathlib import Path


def analyze(paths: list[Path]) -> dict:
    samples = collections.defaultdict(list)
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row['kind'] == 'host_wall_scope'
            assert row['end_ns'] >= row['begin_ns']
            key = (row['pid'], row['thread'], row['sample'], row['step'])
            samples[key].append(row)
    measurements = []
    for key, rows in samples.items():
        rows.sort(key=lambda x: (x['begin_ns'], x['depth']))
        assert rows[0]['name'] == 'V41/step' and rows[0]['depth'] == 0
        stack = []
        children = collections.defaultdict(list)
        inherited_layer = {}
        for i, row in enumerate(rows):
            while stack and rows[stack[-1]]['depth'] >= row['depth']:
                stack.pop()
            if stack:
                parent = stack[-1]
                assert rows[parent]['begin_ns'] <= row['begin_ns']
                assert row['end_ns'] <= rows[parent]['end_ns']
                children[parent].append(i)
            inherited_layer[i] = row['layer'] if row['layer'] >= 0 else (inherited_layer[stack[-1]] if stack else -1)
            stack.append(i)
        for i, row in enumerate(rows):
            exclusive_ns = row['end_ns'] - row['begin_ns'] - sum(rows[j]['end_ns'] - rows[j]['begin_ns'] for j in children[i])
            assert exclusive_ns >= 0
            measurements.append({'pid':key[0], 'thread':key[1], 'step':key[3], 'sample':key[2], 'q':rows[0]['q'], 'kv':rows[0]['kv'], 'layer':inherited_layer[i], 'scope':row['name'], 'host_begin_ns':row['begin_ns'], 'host_inclusive_ms':(row['end_ns']-row['begin_ns'])/1e6, 'host_exclusive_ms':exclusive_ns/1e6, 'gpu_stream_span_ms':row.get('gpu_stream_span_ms'), 'device_event_status':row.get('device_event_status'), 'payload_bytes':row.get('payload_bytes', -1), 'group_size':row.get('group_size', -1)})
    grouped = collections.defaultdict(list)
    for row in measurements:
        grouped[(row['pid'], row['scope'])].append(row)
    summary = []
    for (pid, name), rows in grouped.items():
        valid_gpu = [r['gpu_stream_span_ms'] for r in rows if r['device_event_status'] == 0 and r['gpu_stream_span_ms'] is not None]
        summary.append({'pid':pid, 'scope':name, 'count':len(rows), 'host_exclusive_median_ms':statistics.median(r['host_exclusive_ms'] for r in rows), 'gpu_inclusive_median_ms':statistics.median(valid_gpu) if valid_gpu else None, 'gpu_valid_count':len(valid_gpu), 'payload_bytes':sorted({r['payload_bytes'] for r in rows}), 'group_sizes':sorted({r['group_size'] for r in rows})})
    return {'status':'MEASURED_SCOPES_NOT_AUTOMATIC_CRITICAL_PATH','sample_count':len(samples), 'source_files':[str(x) for x in paths], 'limitations':['GPU stream spans are inclusive and may contain waiting/launch gaps; do not sum nested spans.', 'Host steady time aligns ranks only on the same host.', 'No valid GPU events means GPU timing remains unknown; host smoke data cannot establish model performance.'], 'summary':summary, 'measurements':measurements}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.paths), ensure_ascii=False, indent=2))
