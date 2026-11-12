"""Offline audit of all 16 actual traces and native entry backtraces."""
from pathlib import Path
import argparse
import collections
import gc
import hashlib
import json
import re
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['off', 'custom'])
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--compact', action='store_true', help='Store actual representative step subsets instead of full local copies')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    archive = args.archive or root / (args.mode + '_all16_raw.tar.gz')
    evidence = args.manifest or root.parent / '04_性能' / ('t01_' + args.mode + '_v1_raw/output/trace/archive_manifest.json')
    manifest = json.loads(evidence.read_text())
    expected = {(x['role'], Path(x['path']).name): x for x in manifest}
    assert len(expected) == 32
    output = (args.output_root or root) / args.mode
    output.mkdir(exist_ok=True)
    representatives = output / 'selected_raw'
    representatives.mkdir(exist_ok=True)
    records = {}
    rows = []
    files = []
    offsets = set()
    # Archives may be emitted in filesystem order. Read native definitions
    # before trace correlation, independently of tar member ordering.
    for phase in ['stacks', 'traces']:
        unpacker = subprocess.Popen(['zstd', '-d', '-c', str(archive)], stdout=subprocess.PIPE) if archive.suffix == '.zst' else None
        source = tarfile.open(fileobj=unpacker.stdout, mode='r|') if unpacker else tarfile.open(archive, 'r|gz')
        with source as packed:
            for member in packed:
                if not member.isfile():
                    continue
                path = Path(member.name)
                if (phase == 'stacks') != path.name.startswith('native_cpp_stacks_'):
                    continue
                host = path.parts[-2]
                entry = expected[(host, path.name)]
                raw = packed.extractfile(member).read()
                digest = hashlib.sha256(raw).hexdigest()
                assert len(raw) == entry['bytes'] and digest == entry['sha256'], member.name
                files.append({'role': host, 'file': path.name, 'sha256': digest, 'bytes': len(raw)})
                if path.name.startswith('native_cpp_stacks_'):
                    pid = int(path.stem.rsplit('_', 1)[1])
                    stacks = {}
                    calls = {}
                    semantic = {}
                    for line in raw.splitlines():
                        record = json.loads(line)
                        if record['kind'] == 'stack':
                            stacks[record['stack_id']] = record['frames']
                        elif record['kind'] == 'record':
                            calls[record['record_id']] = (record['name'], record['os_thread'])
                            if record['name'].startswith('V41/'):
                                frames = [f for f in stacks[record['stack_id']] if f['module'].endswith('/bin/xllm')]
                                assert frames, record
                                semantic[record['record_id']] = {**record, 'native_frames': frames}
                                offsets.update(f['module_offset'] for f in frames)
                    records[(host, pid)] = {'calls': calls, 'semantic': semantic,
                                            'stack_file': path.name, 'raw': raw}
                    continue
                match = re.fullmatch(r'xllm_rank(\d+)_(\d+)_\d+\.pt\.trace\.json', path.name)
                assert match, path
                rank, pid = map(int, match.groups())
                backtraces = records.pop((host, pid))
                events = json.loads(raw)['traceEvents']
                cpu_by_id = {e['args']['Record function id']: e for e in events
                             if 'Record function id' in e.get('args', {})}
                matched = set(cpu_by_id) & set(backtraces['calls'])
                assert matched
                for identifier in matched:
                    event = cpu_by_id[identifier]
                    assert (event['name'], event['tid']) == backtraces['calls'][identifier]
                steps = [e for e in events if e.get('ph') == 'X' and e.get('name', '').startswith('V41/step/')]
                layers = {int(e['name'].split('/')[2]) for e in events
                          if e.get('name', '').startswith('V41/layer/')}
                kernels = [e for e in events if 'kernel' in e.get('cat', '').lower()]
                assert len(layers) == 40 and steps and kernels
                step = steps[0] if host == 'sj9' else steps[-1]
                lo, hi = step['ts'], step['ts'] + step['dur']
                selected = [e for e in events if e.get('ph') == 'X' and lo <= e['ts']
                            and e['ts'] + e.get('dur', 0) <= hi + .001]
                cpu = [e for e in selected if e.get('cat') in ['cpu_op', 'user_annotation']]
                gpu = [e for e in selected if 'kernel' in e.get('cat', '').lower()]
                names = collections.Counter(e['name'] for e in gpu)
                cpu_names = collections.Counter(e['name'] for e in cpu)
                # Attribute asynchronous GPU work through the actual launch
                # correlation, not by taking the nearest kernel in wall time.
                communication = []
                runtime = [e for e in events if e.get('ph') == 'X' and
                           (e.get('cat', '').endswith('_runtime') or e.get('cat', '').endswith('_driver'))]
                kernel_by_correlation = collections.defaultdict(list)
                for event in kernels:
                    correlation = event.get('args', {}).get('correlation')
                    if correlation is not None:
                        kernel_by_correlation[correlation].append(event)
                for scope in cpu:
                    if not scope['name'].startswith(('V41/Attention/TP_allreduce', 'V41/MoE/EP_allreduce')):
                        continue
                    launches = [event for event in runtime if event.get('tid') == scope['tid']
                                and scope['ts'] <= event['ts']
                                and event['ts']+event.get('dur', 0) <= scope['ts']+scope['dur']+.001]
                    correlations = {event.get('args', {}).get('correlation') for event in launches}
                    linked = [event for correlation in correlations if correlation is not None
                              for event in kernel_by_correlation[correlation]]
                    communication.append({'scope': scope, 'launches': launches, 'gpu_kernels': linked})
                row = {'role': host, 'rank': rank, 'pid': pid,
                       'trace_file': path.name, 'stack_file': backtraces['stack_file'],
                       'matched_record_ids': len(matched), 'unmatched_record_ids': len(set(backtraces['calls'])-matched),
                       'semantic_entries': len(backtraces['semantic']), 'layers': sorted(layers),
                       'all_gpu_kernels': len(kernels), 'steps': steps,
                       'selected_step': step, 'selected_gpu_kernels': len(gpu),
                       'selected_kernel_counts': dict(names), 'selected_cpu_op_counts': dict(cpu_names),
                       'correlated_communication': communication}
                rows.append(row)
                if rank == 0:
                    target = representatives / host
                    target.mkdir(exist_ok=True)
                    if args.compact:
                        launches = [event for event in runtime if lo <= event['ts'] <= hi]
                        ids = {event.get('args', {}).get('correlation') for event in launches}
                        linked_gpu = [event for event in kernels if event.get('args', {}).get('correlation') in ids]
                        # Some MCCL launches have GPU correlation IDs but no
                        # matching CPU callback in the source capture. Preserve
                        # those actual in-window GPU intervals; do not silently
                        # drop them or invent a CPU launch correlation.
                        bounded_ids = {id(event) for event in gpu}
                        complete_gpu = gpu + [event for event in linked_gpu if id(event) not in bounded_ids]
                        compact_events = [event for event in selected if 'kernel' not in event.get('cat', '').lower()] + complete_gpu
                        (target / path.name).write_text(json.dumps({'traceEvents': compact_events}))
                        selected_ids = {event['args']['Record function id'] for event in compact_events if 'Record function id' in event.get('args', {})}
                        original_rows = [json.loads(line) for line in backtraces['raw'].splitlines()]
                        stack_ids = {record['stack_id'] for record in original_rows if record['kind'] == 'record' and record['record_id'] in selected_ids}
                        with (target / backtraces['stack_file']).open('w') as stream:
                            for record in original_rows:
                                if (record['kind'] == 'stack' and record['stack_id'] in stack_ids) or (record['kind'] == 'record' and record['record_id'] in selected_ids):
                                    stream.write(json.dumps(record)+'\n')
                        (target / 'crop_receipt.json').write_text(json.dumps({'original_trace_sha256': digest, 'original_trace_path': entry['path'], 'original_stack_path': backtraces['stack_file'], 'selected_step': step, 'scope': 'Actual full CPU step events, all GPU kernels fully within that window, plus runtime/driver-correlated asynchronous kernels. Missing CPU launch callbacks remain missing; no invented scope association. Native records and stack definitions filtered by actual Record function IDs; original NFS files remain unchanged.'}, indent=2))
                    else:
                        (target / path.name).write_bytes(raw)
                        (target / backtraces['stack_file']).write_bytes(backtraces['raw'])
                    (output / (host + '_semantic_records.json')).write_text(json.dumps(list(backtraces['semantic'].values())))
                print(json.dumps({'mode': args.mode, 'role': host, 'rank': rank,
                                  'matched': len(matched), 'kernels': len(kernels)}), flush=True)
                del raw, events, cpu_by_id, selected, cpu, gpu, kernels, backtraces
                gc.collect()
        if unpacker:
            assert unpacker.wait() == 0
    assert len(files) == 32 and len(rows) == 16 and not records
    assert {(r['role'], r['rank']) for r in rows} == {(h, i) for h in ['sj9', 'sj10'] for i in range(8)}
    result = {'status': 'PASS_ALL_16_RANK_TRACE_SHA_CPU_GPU_NATIVE_PC_CORRELATION',
              'scope': '256-input/8-output diagnostic. Native flame weights are entry COUNT, not CPU time. GPU events in selected step are bounded by its CPU window; full raw traces remain authoritative.',
              'files': files, 'ranks': rows}
    (output / 'all_rank_audit.json').write_text(json.dumps(result, indent=2))
    (output / 'native_offsets.json').write_text(json.dumps(sorted(offsets)))


if __name__ == '__main__':
    main()
