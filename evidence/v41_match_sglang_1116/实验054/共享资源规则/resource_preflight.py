"""Read-only host resource check; publish the exact snapshot used by launchers."""
from pathlib import Path
import datetime
import json
import re
import socket
import subprocess
import shutil
import runpy

host = {'worker32007': 'sj9', 'worker32009': 'sj10'}[socket.gethostname()]
now = datetime.datetime.now().astimezone()
earliest = datetime.datetime.fromisoformat('2026-11-17T17:05:00+08:00')
control = Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control')
cooldown_file = control / 'external_release_cooldown.json'
if cooldown_file.exists():
    cooldown = json.loads(cooldown_file.read_text())
    if cooldown.get('active') and now < datetime.datetime.fromisoformat(cooldown['not_before']):
        v = {'at': now.isoformat(), 'host': host, 'gpu_hardware_free': False,
             'external_window_elapsed': False, 'gpu_free': False, 'gpu_rows': [],
             'gpu_process_owners': [], 'resource_query_skipped': True,
             'reason': 'Mandatory twenty-minute wait after possible external termination',
             'cooldown': cooldown}
        (control / f'{host}_preflight_p0.json').write_text(json.dumps(v, indent=2))
        print(json.dumps(v))
        raise SystemExit(0)
g = subprocess.run(['timeout', '20', 'mthreads-gmi'], capture_output=True, text=True, timeout=25)
rows = re.findall(r'\|\s*(\d+)%\s+(\d+)MiB\(81920MiB\)', g.stdout)
free = g.returncode == 0 and len(rows) == 8 and all(x == ('0', '0') for x in rows) and 'No running processes found' in g.stdout
ids = subprocess.check_output(['docker', 'ps', '-q', '--no-trunc'], text=True).split()
containers = json.loads(subprocess.check_output(['docker', 'inspect'] + ids)) if ids else []
occupants = []
pids = sorted(set(re.findall(r'^\d+\s+(\d+)\s+', g.stdout, re.MULTILINE)), key=int)
for pid in pids:
    p = Path('/proc') / pid
    try:
        cgroup = (p / 'cgroup').read_text()
        owners = [{'name': c['Name'], 'cid': c['Id']} for c in containers if c['Id'] in cgroup]
        occupants.append({'pid': int(pid), 'comm': (p / 'comm').read_text().strip(),
                          'start_ticks': (p / 'stat').read_text().split()[21], 'containers': owners})
    except FileNotFoundError:
        occupants.append({'pid': int(pid), 'vanished_during_read': True})
compilation = runpy.run_path(str(control / 'compile_activity.py'))['inventory']()
monitor_path = control / 'compilation_monitor' / host / 'latest.json'
monitor = json.loads(monitor_path.read_text()) if monitor_path.exists() else {}
monitor_age = (datetime.datetime.now().astimezone() - datetime.datetime.fromisoformat(monitor['at'])).total_seconds() if monitor else float('inf')
compilation_clear = compilation['compilation_clear'] and monitor.get('compilation_clear', False) and monitor.get('continuous_clear_seconds', 0) >= 120 and monitor_age < 30
v = {'at': now.isoformat(), 'host': host, 'hostname': socket.gethostname(),
     'gpu_exit': g.returncode, 'gpu_stdout': g.stdout, 'gpu_stderr': g.stderr,
     'gpu_hardware_free': free, 'external_window_elapsed': now >= earliest,
     'gpu_free': free and now >= earliest and compilation_clear, 'gpu_rows': rows, 'gpu_process_owners': occupants,
     'compilation_clear': compilation_clear, 'compilation_inventory': compilation,
     'compilation_monitor': monitor, 'compilation_monitor_age_s': monitor_age,
     'ports': subprocess.check_output(['ss', '-ltnH'], text=True),
     'loadavg': Path('/proc/loadavg').read_text().strip(),
     'meminfo': {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:', 'SwapFree:'))},
     'nfs_free_bytes': shutil.disk_usage('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116').free,
     'running_containers': [{'name': c['Name'], 'cid': c['Id'], 'owner': (c['Config'].get('Labels') or {}).get('owner')} for c in containers]}
root = Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control')
history = root / 'resource_snapshots'
history.mkdir(exist_ok=True)
(history / (host + '_' + now.strftime('%Y%m%d_%H%M%S') + '.json')).write_text(json.dumps(v, indent=2))
(root / f'{host}_preflight_p0.json').write_text(json.dumps(v, indent=2))
print(json.dumps({k: v[k] for k in ['at', 'host', 'gpu_hardware_free', 'external_window_elapsed', 'gpu_free', 'gpu_rows', 'gpu_process_owners', 'compilation_clear', 'compilation_inventory']}))

