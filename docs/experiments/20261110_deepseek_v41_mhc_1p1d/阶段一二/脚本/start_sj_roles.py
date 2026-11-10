from pathlib import Path
import sys,shlex,json,concurrent.futures,argparse,subprocess
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE.parents[2]/'sj9_12_登陆工具'));import sj_access
parser=argparse.ArgumentParser();parser.add_argument('--run',default='stage1_ref');parser.add_argument('--hosts',nargs='+',choices=('sj9','sj10'),default=['sj10','sj9']);a=parser.parse_args();assert a.run.replace('_','').isalnum()
if a.run=='stage1_swiglu_d' and not (BASE/'证据/sj9_operator_preflight/exit').exists():
 subprocess.run([sys.executable,str(BASE/'脚本/preflight_sj.py')],check=True)
script=r'''from pathlib import Path
import subprocess,json,re,socket,time
host=HOST;run=RUN;root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops');cfg=json.loads((root/f'artifacts/prepared/{run}/{host}.json').read_text())
name='xiaopang1110_v41_'+('prefill' if host=='sj9' else 'decode')+'_'+host
for gpu_attempt in range(31):
 gmi=subprocess.check_output(['mthreads-gmi'],text=True);cards=re.findall(r'(\d+)%\s+(\d+)MiB\((\d+)MiB\)',gmi)
 if len(cards)==8 and all(u=='0' and m=='0' for u,m,_ in cards):break
 if gpu_attempt==30:raise RuntimeError('GPU still in use after 60 seconds; no existing process was stopped')
 time.sleep(2)
ports=set()
for r in cfg['ranks']:
 f=dict(i[2:].split('=',1) for i in r['command'][1:]);ports.update([int(f['port']),int(f['master_node_addr'].split(':')[-1]),int(f['disagg_pd_port']),int(f['transfer_listen_port'])]);ports.update(range(int(r['environment']['MC_MIN_RPC_PORT']),int(r['environment']['MC_MAX_RPC_PORT'])+1))
if cfg['role']=='DECODE':ports.update(cfg['control_ports'].values())
for attempt in range(90):
 busy=[]
 for port in sorted(ports):
  try:
   with socket.socket() as s:s.bind((cfg['host_ip'],port))
  except OSError as e:
   if e.errno!=98:raise
   busy.append(port)
 if not busy:break
 if attempt==89:raise RuntimeError(f'Ports still unavailable after three minutes: {busy}; no process was stopped')
 time.sleep(2)
out=root/f'runs/{run}/{host}';out.mkdir(parents=True,exist_ok=False)
source=cfg.get('source_dir','/work/repos/xllm-musa');assert source in ('/work/repos/xllm-musa','/work/01_sources/34815f6a','/work/01_sources/stage2_mhc')
shell=f'cd {source}; python3 /work/scripts/pd_rank_supervisor.py --config /work/artifacts/prepared/{run}/{host}.json --output /work/runs/{run}/{host} > /work/runs/{run}/{host}/supervisor.log 2>&1; echo $? > /work/runs/{run}/{host}/supervisor.exit'
cmd=['docker','exec','-d',name,'bash','-c',shell];r=subprocess.run(cmd,capture_output=True,text=True);record={'command':cmd,'gpu_before':gmi,'ports_checked':sorted(ports),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr};(out/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps({'host':host,'run':run,'exit':r.returncode,'stderr':r.stderr}));r.check_returncode()
'''
def go(host):
 with sj_access.ssh_client(host) as c:r=sj_access.execute(c,host,'python3 -c '+shlex.quote(script.replace('HOST',repr(host)).replace('RUN',repr(a.run))),root=True,timeout=220)
 (BASE/'证据'/f'{host}_{a.run}_launch.json').write_text(json.dumps(r,indent=2));return host,r
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(go,a.hosts))
 for r in results:print(r)
 if any(r['exit_status'] for _,r in results):raise RuntimeError('Role launch failed; inspect both host receipts before acceptance')
