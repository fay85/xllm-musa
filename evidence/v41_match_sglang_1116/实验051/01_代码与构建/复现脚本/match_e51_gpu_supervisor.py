from pathlib import Path
import datetime
import hashlib
import json
import os
import resource
import signal
import subprocess
import time

root = Path('/expt')
out = root / 'diagnostics/e51_gather_gpu_v1'
out.mkdir(exist_ok=False)
module = root / 'builds/e51_gather_module_v3/v41_gather_candidate_test_module.so'
assert hashlib.sha256(module.read_bytes()).hexdigest() == '7ccc43c7cfcbce9c9a1b4a02eb54e621fbe953756a0e51f6d33a89ae7490bed9'
script = root / 'scripts/validate_v41_gather_candidate_v1.py'
role = json.loads((root / 'runs/p50_gate_simt_confirm_on_v1/sj10/role_config.json').read_text())
env = dict(os.environ)
env.update(role['ranks'][0]['environment'])
env.update(LD_PRELOAD='', PYTHONPATH=str(module.parent), MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
           MASTER_ADDR='127.0.0.1', MASTER_PORT='35691', NATIVE_PORT='35692', WORLD_SIZE='8',
           TEST_OUTPUT=str(out), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           XLLM_MUSA_NATIVE_SYNC_COLLECTIVES='1', XLLM_MUSA_SYNC_COLLECTIVE_STREAM='0',
           XLLM_MUSA_ENABLE_TP_CUSTOM_AR='1', TORCH_DISTRIBUTED_DEBUG='OFF')
assert env['XLLM_MUSA_NATIVE_SYNC_COLLECTIVES'] == '1'
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
now = lambda: datetime.datetime.now().astimezone().isoformat()
job = {'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().split()[21],
       'started_at': now(), 'container': 'xiaopang1117_v41_sg438_decode_sj10',
       'module_sha256': hashlib.sha256(module.read_bytes()).hexdigest(),
       'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
       'environment': {k: v for k, v in env.items() if k.startswith(('XLLM_', 'MCCL_', 'MUSA_')) or k in ['LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'MASTER_PORT', 'NATIVE_PORT']}}
(out / 'job.json').write_text(json.dumps(job, indent=2))
children = []
identities = []
for rank in range(8):
    child_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank))
    with (out / f'rank{rank}.log').open('xb') as log:
        child = subprocess.Popen(['/opt/venv/bin/python3', str(script)], env=child_env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    p = Path('/proc') / str(child.pid)
    identities.append({'rank': rank, 'pid': child.pid, 'start_ticks': p.joinpath('stat').read_text().split()[21],
                       'exe': str(p.joinpath('exe').resolve()), 'exe_sha256': hashlib.sha256(p.joinpath('exe').read_bytes()).hexdigest(),
                       'cgroup': p.joinpath('cgroup').read_text(), 'script': str(script)})
    children.append(child)
(out / 'children.json').write_text(json.dumps(identities, indent=2))
deadline = time.monotonic() + 360
failure = None
while True:
    codes = [p.poll() for p in children]
    (out / 'status.json').write_text(json.dumps({'at': now(), 'exit_codes': codes}))
    if all(c is not None for c in codes):
        break
    if any(c not in [None, 0] for c in codes):
        failure = 'rank_failed'
        break
    if time.monotonic() > deadline:
        failure = 'bounded_check_deadline'
        break
    time.sleep(2)

cleanup = []
if failure:
    # Exactly the children created above. TERM only; no automatic SIGKILL.
    for child, identity in zip(children, identities):
        if child.poll() is not None:
            continue
        p = Path('/proc') / str(child.pid)
        try:
            fd = os.pidfd_open(child.pid)
            assert p.joinpath('stat').read_text().split()[21] == identity['start_ticks']
            assert str(p.joinpath('exe').resolve()) == identity['exe']
            assert hashlib.sha256(p.joinpath('exe').read_bytes()).hexdigest() == identity['exe_sha256']
            assert p.joinpath('cgroup').read_text() == identity['cgroup']
            assert str(script).encode() in p.joinpath('cmdline').read_bytes()
            threads = {t.name: t.joinpath('stat').read_text().split(') ', 1)[1].split()[0] for t in p.joinpath('task').iterdir()}
            record = {'identity': identity, 'threads': threads, 'at': now(), 'signal': None}
            if 'D' not in threads.values():
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                record['signal'] = 'SIGTERM_ONCE'
            else:
                record['stop_reason'] = 'D_STATE_NO_FORCE'
            os.close(fd)
            cleanup.append(record)
        except ProcessLookupError:
            cleanup.append({'pid': child.pid, 'already_exited': True})
    (out / 'normal_cleanup.json').write_text(json.dumps(cleanup, indent=2))
    for _ in range(15):
        if all(p.poll() is not None for p in children):
            break
        time.sleep(2)
codes = [p.poll() for p in children]
checks = []
for rank in range(8):
    p = out / f'rank{rank}.json'
    if p.exists():
        checks.append(json.loads(p.read_text()))
passed = codes == [0] * 8 and len(checks) == 8 and all(x['status'] == 'FINITE_AND_GRAPH_PASS_SPECIAL_CONTRACT_RECORDED' for x in checks)
(out / 'finished.json').write_text(json.dumps({'at': now(), 'exit_codes': codes, 'all_pass_finite_and_graph': passed,
    'failure': failure, 'remaining_pids': [p.pid for p in children if p.poll() is None],
    'automatic_sigkill': False, 'special_contract_pass': passed and all(not x['special_value_mismatches'] for x in checks)}, indent=2))
