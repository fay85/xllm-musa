from pathlib import Path
import os,json,subprocess,time,datetime,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
r=Path('/expt/builds/gather_probe_v1');g=r/'gpu_checks';cfg=json.loads(Path('/expt/runs/p2_full_decode_graph_v1/sj9/role_config.json').read_text());env=dict(os.environ,**cfg['ranks'][0]['environment']);env.update(LD_PRELOAD='',MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONPATH=str(r)+':/workspace/xllm-mt-main',MASTER_ADDR='127.0.0.1',MASTER_PORT='39161',WORLD_SIZE='8',XLLM_MUSA_ENABLE_TP_CUSTOM_AR='1',XLLM_MUSA_SYNC_COLLECTIVE_STREAM='0',OMP_NUM_THREADS='1')
children=[];ids=[];streams=[]
for rank in range(8):
 log=(g/f'rank{rank}.log').open('xb');streams.append(log);e=dict(env,RANK=str(rank),LOCAL_RANK=str(rank));cmd=['python3',str(r/'benchmark_gather_probe.py'),'--native-port','39162','--output',str(g/'results')];p=subprocess.Popen(cmd,env=e,stdout=log,stderr=subprocess.STDOUT);children.append(p);ident={'rank':rank,'pid':p.pid,'start_ticks':Path('/proc').joinpath(str(p.pid),'stat').read_text().split()[21]};ids.append(ident)
(g/'ranks.json').write_text(json.dumps(ids,indent=2));start=time.monotonic();requested=False
while any(p.poll() is None for p in children):
 codes=[p.poll() for p in children];(g/'status.json').write_text(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'exit_codes':codes,'pids':[p.pid for p in children]},indent=2))
 if (any(x is not None and x!=0 for x in codes) or time.monotonic()-start>300) and not requested:
  receipt=[]
  for ident,p in zip(ids,children):
   path=Path('/proc')/str(p.pid)
   if p.poll() is None and path.exists() and (path/'stat').read_text().split()[21]==ident['start_ticks']:
    cmd=(path/'cmdline').read_bytes();assert str(r/'benchmark_gather_probe.py').encode() in cmd;receipt.append(dict(ident,exe=str((path/'exe').resolve()),cmd=cmd.decode().replace('\0',' '),cgroup=(path/'cgroup').read_text()));p.terminate()
  (g/'term_once.json').write_text(json.dumps(receipt,indent=2));requested=True
 if requested and time.monotonic()-start>360:break
 time.sleep(2)
for f in streams:f.close()
(g/'finished.json').write_text(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'exit_codes':[p.poll() for p in children],'all_pass':all(p.poll()==0 for p in children),'normal_term_requested':requested},indent=2))
