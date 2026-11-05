"""Task-owned CPU control plane, configured addresses, no kill escalation."""
from pathlib import Path
import argparse, json, os, secrets, signal, subprocess, time
import requests

parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
cfg=json.loads(args.config.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
host=cfg['control_host'];ports=cfg['control_ports'];namespace=cfg['namespace'];token=secrets.token_urlsafe(32)
etcd=f"http://{host}:{ports['etcd']}";peer=f"http://127.0.0.1:{ports['etcd_peer']}"
commands=[['/opt/xllm_service/bin/etcd','--name=xiaopang1105','--data-dir='+str(out/'etcd-data'),'--listen-client-urls='+etcd,'--advertise-client-urls='+etcd,'--listen-peer-urls='+peer,'--initial-advertise-peer-urls='+peer,'--initial-cluster=xiaopang1105='+peer]]
for role,prefix in [('ROUTER','router'),('COORDINATOR','coordinator')]:
 commands.append(['/opt/xllm_service/bin/xllm_master_serving','--server_host='+host,'--etcd_addr='+f"{host}:{ports['etcd']}",'--etcd_namespace='+namespace,'--http_server_port='+str(ports[prefix+'_http']),'--rpc_server_port='+str(ports[prefix+'_rpc']),'--http_server_num_threads=4','--rpc_server_num_threads=4','--num_threads=4','--tokenizer_path='+cfg['model'],'--block_size=128','--logtostderr=true','--enable_request_trace=false','--internal_api_token='+token,'--service_deployment_role='+role,'--native_pd_routing_policy=STATIC_BUCKET','--kv_route_mode=DISABLED','--load_balance_policy=CAR','--first_output_watchdog_timeout_ms=1800000','--default_request_deadline_ms=3600000','--max_request_deadline_ms=3600000'])
(out/'commands.json').write_text(json.dumps([[x.replace(token,'[TASK_INTERNAL_TOKEN]') for x in cmd] for cmd in commands],indent=2))
control_env=dict(os.environ,LD_PRELOAD='',MUSA_VISIBLE_DEVICES='-1')
control_env['LD_LIBRARY_PATH']='/opt/xllm_service/lib:'+os.environ.get('LD_LIBRARY_PATH','')
children=[];streams=[];stopping=False;ready=False

def stop(signum=0,frame=None):
 global stopping
 if stopping:return
 stopping=True
 for p in reversed(children):
  if p.poll() is None:p.terminate()

signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def healthy(url):
 try:return requests.get(url,timeout=1).ok
 except requests.RequestException:return False
try:
 for i,command in enumerate(commands):
  if stopping:break
  stream=(out/['etcd.log','router.log','coordinator.log'][i]).open('xb',buffering=0);streams.append(stream)
  children.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=control_env))
  if i==0:
   for _ in range(60):
    if stopping or children[0].poll() is not None:raise RuntimeError('etcd stopped')
    if healthy(etcd+'/health'):break
    time.sleep(1)
   else:raise RuntimeError('etcd health timeout')
 for _ in range(90):
  if stopping or (out/'stop.request').exists() or any(p.poll() is not None for p in children):break
  if healthy(f"http://{host}:{ports['router_http']}/livez") and healthy(f"http://{host}:{ports['coordinator_http']}/readyz"):
   ready=True;(out/'ready.json').write_text(json.dumps({'control_plane_ready':True,'inference_capacity_not_checked':True}));break
  time.sleep(1)
 if not ready:stop()
 while any(p.poll() is None for p in children):
  if (out/'stop.request').exists() or any(p.poll() is not None for p in children):stop()
  time.sleep(2)
finally:
 stop()
 while any(p.poll() is None for p in children):time.sleep(2)
 (out/'result.json').write_text(json.dumps({'control_plane_was_ready':ready,'exit_codes':[p.poll() for p in children],'automatic_kill':False},indent=2))
raise SystemExit(0 if ready else 1)
