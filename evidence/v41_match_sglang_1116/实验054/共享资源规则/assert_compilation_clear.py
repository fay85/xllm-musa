"""Reject performance launches while either host is compiling or unobserved."""
from pathlib import Path
import datetime
import json
import runpy

root = Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control')
if not root.exists():
    root = Path('/expt/00_control')
for host in ['sj9', 'sj10']:
    v = json.loads((root / 'compilation_monitor' / host / 'latest.json').read_text())
    age = (datetime.datetime.now().astimezone() - datetime.datetime.fromisoformat(v['at'])).total_seconds()
    assert age < 30 and v['compilation_clear'] and v['continuous_clear_seconds'] >= 120, (host, v)
# On the host, also check live process state to reduce the launch race window.
if root.as_posix().startswith('/data/'):
    now = runpy.run_path(str(root / 'compile_activity.py'))['inventory']()
    assert now['compilation_clear'], now
