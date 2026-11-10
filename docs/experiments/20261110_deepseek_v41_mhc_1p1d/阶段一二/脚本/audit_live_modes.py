"""Check actual operator selection logs, without modifying the running roles."""
from pathlib import Path
import argparse
import json
import shlex
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()
assert args.run.replace('_', '').isalnum()
base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base.parents[2] / 'sj9_12_登陆工具'))
import sj_access

code = '''from pathlib import Path
import ast,json,re,datetime
p=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops/runs')/RUN
rows=[]
for host in ('sj9','sj10'):
 cfg=json.loads((p/host/'role_config.json').read_text())
 for rank in cfg['ranks']:
  text=(p/host/f"rank{rank['rank']}.log").read_text(errors='replace')
  modes=[ast.literal_eval(x.split('V4.1 MUSA mHC: ',1)[1]) for x in text.splitlines() if 'V4.1 MUSA mHC: ' in x]
  shared=re.findall(r'V4.1 MUSA shared SwiGLU: mode=([^,]+),',text)
  expected={k:rank['environment']['XLLM_DSV41_MUSA_HC_'+k.upper()] for k in ('pre','post','metadata')}
  expected_shared=rank['environment']['XLLM_DSV41_MUSA_SHARED_SWIGLU']
  assert len(modes)==40 and all(x==expected for x in modes)
  assert len(shared)==40 and set(shared)=={expected_shared}
  rows.append({'host':host,'rank':rank['rank'],'mhc':expected,'shared':expected_shared,'mhc_records':len(modes),'shared_records':len(shared)})
print(json.dumps({'status':'PASS_ALL_16_RANKS_40_LAYERS','at':datetime.datetime.now().astimezone().isoformat(),'rows':rows}))
'''.replace('RUN', repr(args.run))
with sj_access.ssh_client('sj10') as client:
    result = sj_access.execute(client, 'sj10', 'python3 -c ' + shlex.quote(code), root=True, timeout=30)
assert result['exit_status'] == 0, result['stderr']
record = json.loads(result['stdout'])
(base / '证据' / f'{args.run}_early_modes.json').write_text(json.dumps(record, indent=2))
print(json.dumps({'run': args.run, 'status': record['status'], 'per_host': [x for x in record['rows'] if x['rank'] == 0]}))
