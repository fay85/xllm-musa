#!/usr/bin/env python3
"""Owned eight-GPU PD experiment: fixed Prefill, paired TP1/TP2 Decode."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, '/workspace/xllm-mt-main/tools/decode_tp2_dev')
import launch

BASE = Path('/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027')
HOST = '10.121.38.92'
MODEL = '/data/nfs_shared/models/Qwen3.8-27B-FP8'
SERVICE = '/opt/xllm_service/bin/xllm_master_serving.rr_probe_fix_20261024'
ETCD = '/opt/xllm_service/bin/etcd'


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + '\n')


def _stop(root: Path) -> None:
    records = json.loads((root / 'processes.json').read_text())
    for category in ['engine', 'service', 'etcd']:
        selected = [x for x in records if x['category'] == category]
        for record in selected:
            current = launch._current(record['identity'])
            if current is None or current['state'] == 'Z':
                continue
            if not launch._same_identity(current, record['identity']):
                raise RuntimeError(f"Process identity changed: {record['name']}")
            os.killpg(current['pgid'], signal.SIGTERM)
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if not any((launch._current(x['identity']) or {}).get('state', 'Z') != 'Z' for x in selected):
                break
            time.sleep(0.25)
        for record in selected:
            current = launch._current(record['identity'])
            if current is None or current['state'] == 'Z':
                continue
            if not launch._same_identity(current, record['identity']):
                raise RuntimeError('PID reuse during stop')
            os.killpg(current['pgid'], signal.SIGKILL)
    _dump(root / 'stopped.json', {'epoch': time.time()})


def _start(args: argparse.Namespace) -> None:
    config = json.loads(Path('/workspace/xllm-mt-main/tools/decode_tp2_dev/config.json').read_text())
    launch._assert_container(config)
    snapshot = launch._gpu_snapshot(subprocess.check_output(['mthreads-gmi'], text=True))
    for gpu in range(8):
        device = snapshot['devices'][gpu]
        if device['utilization'] or device['used_mib'] > 1:
            raise RuntimeError(f'GPU {gpu} is occupied: {device}')
    ports = [57579, 57580, 57888, 57889, 57890, 57891]
    for is_prefill, instances, ranks in [(True, 3, 2), (False, 1 if args.topology == 'tp2' else 2,
                                                     2 if args.topology == 'tp2' else 1)]:
        for instance in range(instances):
            master = (28000 if is_prefill else 29000) + instance * 32
            ports.extend(range(master, master + 17))
            ports.extend([(28100 if is_prefill else 29100) + instance,
                          (28200 if is_prefill else 29200) + instance])
            for rank in range(ranks):
                rpc = (30000 if is_prefill else 30400) + instance * 100 + rank * 24
                ports.extend(range(rpc, rpc + 17))
                ports.append((28300 if is_prefill else 29300) + instance * 32 + rank * 8)
    if len(ports) != len(set(ports)) or any(not 1 <= p <= 65535 for p in ports):
        raise RuntimeError('Invalid or overlapping experiment ports')
    busy = sorted(set(ports) & launch._listeners())
    if busy:
        raise RuntimeError(f'Host listeners occupy experiment ports: {busy}')
    root = Path(args.run)
    root.mkdir(parents=True, exist_ok=False)
    env_base = launch._environment(config, 0)
    env_base['XLLM_MUSA_ENABLE_TP_CUSTOM_AR'] = '1'
    env_base['XLLM_MUSA_PCP_CUSTOM_AR'] = '1' if args.pcp_custom_ar else '0'
    env_base.pop('VLLM_CUSTOM_ALLREDUCE_ALGO', None)
    records = []

    def spawn(name: str, command: list[str], env: dict, category: str) -> None:
        with (root / f'{name}.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True, cwd=root)
        records.append({'name': name, 'category': category, 'argv': command, 'environment': env,
                        'identity': launch._identity(process.pid)})
        _dump(root / 'processes.json', records)
        print('START', name, process.pid, flush=True)

    def wait_for(name: str, marker: str, timeout: int = 900) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = (root / f'{name}.log').read_text(errors='replace')
            if marker in text:
                return
            record = next(x for x in records if x['name'] == name)
            state = launch._current(record['identity'])
            if state is None or state['state'] == 'Z':
                raise RuntimeError(f'{name} exited during startup; see its log')
            time.sleep(1)
        raise RuntimeError(f'{name} startup timed out')

    _dump(root / 'config.json', vars(args))
    _dump(root / 'binary_sha256.json', {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                      for p in [args.prefill_bin, args.decode_bin, SERVICE]})
    control_env = dict(os.environ, LD_LIBRARY_PATH='/opt/xllm_service/lib:/usr/lib:/usr/lib/x86_64-linux-gnu')
    control_env.pop('LD_PRELOAD', None)
    client, peer = 57579, 57580
    spawn('etcd', [ETCD, '--name=tp2-comparison', f'--data-dir={root}/etcd_data',
                  f'--listen-client-urls=http://127.0.0.1:{client}',
                  f'--advertise-client-urls=http://127.0.0.1:{client}',
                  f'--listen-peer-urls=http://127.0.0.1:{peer}',
                  f'--initial-advertise-peer-urls=http://127.0.0.1:{peer}',
                  f'--initial-cluster=tp2-comparison=http://127.0.0.1:{peer}',
                  '--initial-cluster-state=new'], control_env, 'etcd')
    wait_for('etcd', 'ready to serve client requests', 30)
    common_service = [SERVICE, f'--server_host={HOST}', f'--etcd_addr=127.0.0.1:{client}',
                      '--tokenizer_path=/root/xllm_vl_bench/master_tokenizer', '--block_size=64',
                      '--num_threads=4', '--http_server_num_threads=4', '--rpc_server_num_threads=4',
                      '--logtostderr=true', '--internal_api_token=lab-local-token',
                      '--native_pd_routing_policy=STATIC_BUCKET', '--load_balance_policy=RR',
                      '--kv_route_mode=DISABLED', '--batch_credit_max_grant_units=8192',
                      '--batch_credit_max_grant_units_per_router_tenant=8192',
                      '--batch_credit_safety_permyriad=10000', '--first_output_watchdog_timeout_ms=180000',
                      '--flow_max_queue_wait_ms=60000', '--default_request_deadline_ms=1800000',
                      '--max_request_deadline_ms=3600000']
    spawn('master', common_service + ['--http_server_port=57888', '--rpc_server_port=57889',
                                     '--service_deployment_role=ROUTER'], control_env, 'service')
    spawn('coordinator', common_service + ['--http_server_port=57890', '--rpc_server_port=57891',
                                          '--service_deployment_role=COORDINATOR'], control_env, 'service')
    time.sleep(3)
    common = [f'--model={args.model}', f'--model_id={args.model_id}', f'--host={HOST}', '--backend=vlm',
              '--random_seed=42', '--max_body_size=268435456', '--block_size=64',
              '--max_encoder_cache_size=256',
              '--enable_graph_vmm_pool=false', '--enable_disagg_pd=true', '--enable_native_service_adapter=true',
              '--kv_cache_transfer_mode=PULL', '--kv_split_size=1', f'--etcd_addr=127.0.0.1:{client}',
              '--usercode_in_pthread=true', f'--draft_model={args.model}', '--speculative_algorithm=MTP',
              '--num_speculative_tokens=3', '--enable_mtp_draft_body_tp1=false', '--enable_mm_encoder_dp=false',
              '--enable_graph=true', '--enable_graph_mode_decode_no_padding=true', '--enable_schedule_overlap=true',
              '--enable_prefix_cache=true', '--enable_chunked_prefill=true', '--vmodule=disagg_pd_scheduler=1']

    def engine(name: str, binary: str, gpus: str, role: str, rank: int, instance: int, nnodes: int) -> None:
        is_prefill = role == 'PREFILL'
        master = (28000 if is_prefill else 29000) + instance * 32
        rpc = (30000 if is_prefill else 30400) + instance * 100 + rank * 24
        env = dict(env_base)
        for key in ['MUSA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'MTHREADS_VISIBLE_DEVICES']:
            env[key] = gpus
        env.update(MC_MIN_RPC_PORT=str(rpc), MC_MAX_RPC_PORT=str(rpc + 16))
        env['XLLM_BIN'] = binary
        env['XLLM_PACKED_PREFILL_PIECEWISE'] = '1' if is_prefill else '0'
        if not is_prefill:
            env['XLLM_MUSA_PULL_STAGING_MB'] = str(args.staging_mib)
            if args.decode_mccl_min_nchannels:
                env['MCCL_MIN_NCHANNELS'] = str(args.decode_mccl_min_nchannels)
        command = [binary, *common, f'--instance_role={role}', f'--nnodes={nnodes}', f'--node_rank={rank}',
                   '--dp_size=1', f'--cp_size={args.prefill_cp_size if is_prefill else 1}',
                   f'--port={(28100 if is_prefill else 29100) + instance}',
                   f'--master_node_addr={HOST}:{master}',
                   f'--transfer_listen_port={(28300 if is_prefill else 29300) + instance * 32 + rank * 8}',
                   f'--disagg_pd_port={(28200 if is_prefill else 29200) + instance}',
                   '--max_concurrent_requests=16', '--max_seqs_per_batch=16',
                   f'--max_tokens_per_batch={args.chunk if is_prefill else 32768}',
                   f'--max_tokens_per_chunk_for_prefill={args.chunk if is_prefill else 2048}',
                   f'--max_tokens_for_graph_mode={8192 if is_prefill else 4096}',
                   f'--enable_prefill_piecewise_graph={str(is_prefill).lower()}']
        memory = args.memory if is_prefill or args.decode_memory is None else args.decode_memory
        command.append(f'--max_memory_utilization={memory}')
        slots = args.prefill_state_cache_slots if is_prefill else args.decode_state_cache_slots
        if slots > 0:
            command.append(f'--max_linear_state_cache_slots={slots}')
        if is_prefill and args.online_profile_dir:
            command += ['--enable_online_profile=true', f'--profile_dir={args.online_profile_dir}']
        spawn(name, command, env, 'engine')

    try:
        if args.topology == 'tp2':
            for rank in [0, 1]:
                engine(f'decode0_r{rank}', args.decode_bin, '6,7', 'DECODE', rank, 0, 2)
            wait_for('decode0_r0', 'Application startup complete')
            if 'Attached Native Provider Descriptor' not in (root / 'decode0_r0.log').read_text(errors='replace'):
                raise RuntimeError('TP2 Decode did not publish its Native descriptor')
        else:
            for i, gpu in enumerate(['6', '7']):
                engine(f'decode{i}_r0', args.decode_bin, gpu, 'DECODE', 0, i, 1)
            for i in range(2):
                wait_for(f'decode{i}_r0', 'Application startup complete')
                if 'Attached Native Provider Descriptor' not in (root / f'decode{i}_r0.log').read_text(errors='replace'):
                    raise RuntimeError('TP1 Decode did not publish its Native descriptor')
        for i, gpus in enumerate(['0,1', '2,3', '4,5']):
            for rank in [0, 1]:
                engine(f'prefill{i}_r{rank}', args.prefill_bin, gpus, 'PREFILL', rank, i, 2)
        for i in range(3):
            wait_for(f'prefill{i}_r0', 'Application startup complete')
        deadline = time.monotonic() + 90
        last_registry = []
        while time.monotonic() < deadline:
            body = json.dumps({'key': base64.b64encode(b'XLLM:').decode(),
                               'range_end': base64.b64encode(b'XLLM;').decode()}).encode()
            request = urllib.request.Request(f'http://127.0.0.1:{client}/v3/kv/range', body,
                                             {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=5) as response:
                entries = json.load(response).get('kvs', [])
            last_registry = [{'key': base64.b64decode(x['key']).decode(),
                              'value': json.loads(base64.b64decode(x['value']))}
                             for x in entries if base64.b64decode(x['key']).startswith(
                                 (b'XLLM:PREFILL:', b'XLLM:DECODE:'))]
            native_prefills = sum(x['key'].startswith('XLLM:PREFILL:') and
                                  'provider_descriptor' in x['value'] for x in last_registry)
            native_decodes = sum(x['key'].startswith('XLLM:DECODE:') and
                                 'provider_descriptor' in x['value'] for x in last_registry)
            if native_prefills == 3 and native_decodes == (1 if args.topology == 'tp2' else 2):
                try:
                    with urllib.request.urlopen(f'http://{HOST}:57888/readyz', timeout=5) as response:
                        ready = response.status == 200
                    if ready:
                        break
                except urllib.error.HTTPError:
                    pass
            time.sleep(1)
        else:
            _dump(root / 'failed_native_registry.json', last_registry)
            raise RuntimeError('Native P/D registration or Service readiness failed')
        _dump(root / 'native_registry_ready.json', last_registry)
        _dump(root / 'ready.json', {'epoch': time.time(), 'url': f'http://{HOST}:57888'})
        print('READY', str(root), flush=True)
    except Exception:
        _stop(root)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--topology', choices=['tp1', 'tp2'], default='tp2')
    parser.add_argument('--prefill-bin', default='/opt/xllm/releases/musa-prefill-bounded-20261026/bin/xllm')
    parser.add_argument('--model', default='/data/nfs_shared/models/Qwen3.8-27B-FP8')
    parser.add_argument('--model-id', default='Qwen3.8-27B-FP8')
    parser.add_argument('--decode-bin', default='/workspace/decode_tp2_dev_artifacts/bin/xllm.native_tp2_twoshot')
    parser.add_argument('--chunk', type=int, default=1024)
    parser.add_argument('--memory', type=float, default=0.85)
    parser.add_argument('--decode-memory', type=float, default=None)
    parser.add_argument('--prefill-state-cache-slots', type=int, default=96)
    parser.add_argument('--prefill-cp-size', type=int, choices=[1, 2], default=2)
    parser.add_argument('--online-profile-dir', default='')
    parser.add_argument('--decode-state-cache-slots', type=int, default=0)
    parser.add_argument('--staging-mib', type=int, default=512)
    parser.add_argument('--pcp-custom-ar', type=int, choices=[0, 1], default=1)
    parser.add_argument('--decode-mccl-min-nchannels', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'stop':
        _stop(Path(args.run))
    else:
        _start(args)


if __name__ == '__main__':
    main()
