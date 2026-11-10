"""Mirror reviewable reports only; never touch running source or configuration."""
from pathlib import Path
import base64
import datetime
import hashlib
import json
import shlex
import sys

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base.parents[2] / 'sj9_12_登陆工具'))
import sj_access

files = {}
for name in ('今日工作总结.md', '今晚待运行的goal.md', '给家明的阶段性汇报.md'):
    files[name] = (base.parent / name).read_bytes()
for name in ('commit_message.txt', '最终提交与交付记录.md'):
    path = base.parent / name
    if path.exists():
        files[name] = path.read_bytes()
for name in ('mHC接入与数值问题说明.md', '代码复核.md', '固定前缀探针与V2协议缺口.md', '逐项算子完整模型对照.md'):
    files['阶段一二/' + name] = (base / name).read_bytes()
for path in (base / '证据').glob('*.json'):
    if path.name != 'reports_mirror_receipt.json' and path.stat().st_size <= 1000000:
        files['阶段一二/证据/' + path.name] = path.read_bytes()
payload = {name: {'data': base64.b64encode(data).decode(), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in files.items()}
code = '''from pathlib import Path
import sys,json,base64,hashlib
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops/04_reports')
files=json.load(sys.stdin)
for name,item in files.items():
 p=root/name
 assert p.resolve().is_relative_to(root.resolve())
 raw=base64.b64decode(item['data']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
print(json.dumps({'files':len(files),'root':str(root)}))
'''
with sj_access.ssh_client('sj10') as client:
    result = sj_access.execute(client, 'sj10', 'python3 -c ' + shlex.quote(code), root=True, payload=json.dumps(payload), timeout=60)
assert result['exit_status'] == 0, result['stderr']
receipt = {'at': datetime.datetime.now().astimezone().isoformat(), 'result': json.loads(result['stdout']), 'files': {name: item['sha256'] for name, item in payload.items()}}
(base / '证据/reports_mirror_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
print(json.dumps(receipt['result'], ensure_ascii=False))
