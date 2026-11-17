"""CPU-only observation, with identity and history; no resource acquisition."""
from pathlib import Path
import datetime
import json
import os
import runpy
import socket
import time

root = Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control')
inventory = runpy.run_path(str(root / 'compile_activity.py'))['inventory']
host = {'worker32007': 'sj9', 'worker32009': 'sj10'}[socket.gethostname()]
out = root / 'compilation_monitor' / host
out.mkdir(parents=True, exist_ok=True)
job = {'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().split()[21],
       'at': datetime.datetime.now().astimezone().isoformat(), 'host': host}
(out / 'job.json').write_text(json.dumps(job, indent=2))
clear_since = None
deadline = time.monotonic() + 6 * 3600
with (out / 'observations.jsonl').open('a', buffering=1) as history:
    while time.monotonic() < deadline and not (out / 'stop.request').exists():
        try:
            v = inventory()
            now = datetime.datetime.fromisoformat(v['at'])
            clear_since = (clear_since or now) if v['compilation_clear'] else None
            v['continuous_clear_seconds'] = (now - clear_since).total_seconds() if clear_since else 0
            v['monitor_job'] = job
        except Exception as e:
            clear_since = None
            v = {'at': datetime.datetime.now().astimezone().isoformat(),
                 'compilation_clear': False, 'continuous_clear_seconds': 0,
                 'monitor_job': job, 'error': repr(e)}
        history.write(json.dumps(v) + '\n')
        tmp = out / 'latest.tmp'
        tmp.write_text(json.dumps(v, indent=2))
        tmp.replace(out / 'latest.json')
        time.sleep(10)
(out / 'finished.json').write_text(json.dumps({'at': datetime.datetime.now().astimezone().isoformat(), 'exit_code': 0}))
