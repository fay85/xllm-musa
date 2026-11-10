"""Audit prepared configurations; actual selections are checked in rank logs."""
from pathlib import Path
import datetime
import base64
import json
import shlex
import sys
import zlib

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base.parents[2] / 'sj9_12_登陆工具'))
import sj_access

runs = ['stage1_ref', 'stage1_swiglu_d_v2', 'stage2_post', 'stage2_post_pre', 'stage2_all']
code = '''from pathlib import Path
import json,base64,zlib
p=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops')
runs=RUNS
out={r:{h:json.loads((p/'artifacts/prepared'/r/(h+'.json')).read_text()) for h in ('sj9','sj10')} for r in runs}
print(base64.b64encode(zlib.compress(json.dumps(out).encode())).decode())
'''.replace('RUNS', repr(runs))
with sj_access.ssh_client('sj10') as client:
    result = sj_access.execute(client, 'sj10', 'python3 -c ' + shlex.quote(code), root=True, timeout=30)
assert result['exit_status'] == 0, result['stderr']
# These task-created configurations contain no credentials. Compress their
# repetitive rank entries to fit the SSH helper's bounded diagnostic output.
configs = json.loads(zlib.decompress(base64.b64decode(result['stdout'])))
rows = []
allowed_flags = {'dump_config_json_file', 'etcd_namespace', 'python_model_path'}
allowed_env = {'PYTHONPATH', 'XLLM_DSV41_MUSA_SHARED_SWIGLU', 'XLLM_DSV41_MUSA_HC_PRE', 'XLLM_DSV41_MUSA_HC_POST', 'XLLM_DSV41_MUSA_HC_METADATA'}
for before, after in zip(runs, runs[1:]):
    for host in ('sj9', 'sj10'):
        a, b = configs[before][host], configs[after][host]
        for field in ('host_ip', 'role', 'TP', 'EP', 'MoE_TP', 'mode', 'eager', 'model', 'control_host', 'control_ports'):
            assert a[field] == b[field], (before, after, host, field)
        assert len(a['ranks']) == len(b['ranks']) == 8
        for rank, (x, y) in enumerate(zip(a['ranks'], b['ranks'])):
            flags = lambda v: dict(s[2:].split('=', 1) for s in v['command'][1:])
            for category, old, new, allowed in [('flags', flags(x), flags(y), allowed_flags), ('environment', x['environment'], y['environment'], allowed_env)]:
                for key in sorted(set(old) | set(new)):
                    if old.get(key) != new.get(key):
                        assert key in allowed, (before, after, host, rank, category, key)
                        rows.append({'before': before, 'after': after, 'host': host, 'rank': rank, 'category': category, 'key': key, 'old': old.get(key), 'new': new.get(key)})
report = {'status': 'PASS_EXPECTED_CONFIGURATION_DIFFERENCES_ONLY', 'at': datetime.datetime.now().astimezone().isoformat(), 'scope': 'Prepared configurations; use run role_config.json and runtime selection records for execution proof.', 'binary_by_run': {r: configs[r]['sj10']['binary_sha256'] for r in runs}, 'differences': rows}
(base / '证据/configuration_matrix_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({'status': report['status'], 'rank_pairs': 64, 'differences': len(rows)}))
