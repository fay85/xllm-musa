"""Finite CPU-only control-plane preflight; no model requests and no GPUs."""
from pathlib import Path
import argparse,json,os,re,socket,subprocess,time
parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args();assert re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id)
root=Path('/work');config=root/'artifacts/prepared/candidate_v1/dev7.json';cfg=json.loads(config.read_text());out=root/'evidence'/args.run_id;out.mkdir(exist_ok=False)
for port in cfg['control_ports'].values():
 with socket.socket() as s:
  s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind((cfg['control_host'],port))
env=dict(os.environ,LD_PRELOAD='',MUSA_VISIBLE_DEVICES='-1')
with (out/'supervisor.log').open('w') as log:
 p=subprocess.Popen(['python3','/work/scripts/pd_control_supervisor.py','--config',str(config),'--output',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
 ready=False
 for _ in range(160):
  if (out/'ready.json').exists():ready=True;break
  if p.poll() is not None:break
  time.sleep(1)
 (out/'stop.request').write_text('Finite CPU control preflight complete\n')
 if p.poll() is None:p.terminate()
 try:code=p.wait(timeout=30)
 except subprocess.TimeoutExpired:code='TERM_PENDING_NO_KILL'
 (out/'preflight_result.json').write_text(json.dumps({'ready':ready,'exit':code,'scope':'CPU control health only; no inference or PD transfer validation','automatic_kill':False},indent=2));print(json.dumps({'ready':ready,'exit':code}))
