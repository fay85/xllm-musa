"""Run only this task's eight ranks; preserve logs, TERM once, never SIGKILL."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,time
import requests
parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
cfg=json.loads(args.config.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
assert hashlib.sha256(Path(cfg['binary']).read_bytes()).hexdigest()==cfg['binary_sha256'],'Binary changed after preparation'
(out/'role_config.json').write_text(json.dumps(cfg,indent=2))
children=[];streams=[];control=None;stopping=False

def stop(signum=0,frame=None):
 global stopping
 if stopping:return
 stopping=True
 for p in children:
  if p.poll() is None:p.terminate()

signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 if cfg['role']=='DECODE':
  env=dict(os.environ);env.pop('LD_PRELOAD',None)
  control=subprocess.Popen(['python3','/work/scripts/pd_control_supervisor.py','--config',str(args.config),'--output',str(out/'control')],env=env)
 if cfg.get('requires_control_plane',True):
  for _ in range(180):
   if stopping:raise RuntimeError('Stopped before rank launch')
   if control and control.poll() is not None:raise RuntimeError('Control supervisor exited')
   try:
    h=cfg['control_host'];p=cfg['control_ports']
    if requests.get(f"http://{h}:{p['router_http']}/livez",timeout=1).ok and requests.get(f"http://{h}:{p['coordinator_http']}/readyz",timeout=1).ok:break
   except requests.RequestException:pass
   time.sleep(1)
  else:raise RuntimeError('No ready control plane; no GPU ranks started')
 for rank in cfg['ranks']:
  if stopping:break
  stream=(out/f"rank{rank['rank']}.log").open('xb',buffering=0);streams.append(stream)
  children.append(subprocess.Popen(rank['command'],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,**rank['environment'])))
 while any(p.poll() is None for p in children):
  if (out/'stop.request').exists() or any(p.poll() is not None for p in children) or (control and control.poll() is not None):stop()
  state={'at':datetime.datetime.now().astimezone().isoformat(),'pids':[p.pid for p in children],'exit_codes':[p.poll() for p in children],'stopping':stopping}
  (out/'status.tmp').write_text(json.dumps(state,indent=2));os.replace(out/'status.tmp',out/'status.json');time.sleep(2)
finally:
 stop()
 while any(p.poll() is None for p in children):time.sleep(2)
 if control and control.poll() is None:control.terminate()
 if control:control.wait()
 (out/'result.json').write_text(json.dumps({'exit_codes':[p.poll() for p in children],'control_exit':control.poll() if control else None,'stopping':stopping,'automatic_kill':False},indent=2))
raise SystemExit(0 if len(children)==8 and all(p.returncode==0 for p in children) else 1)
