"""Remote launch of one reviewed role after immediate whole-host idle checks."""
from pathlib import Path
import argparse, datetime, json, re, socket, subprocess
parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--run-id',required=True);args=parser.parse_args()
assert re.fullmatch(r'[A-Za-z0-9_-]{1,48}',args.run_id)
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离');cfg=json.loads(args.config.read_text())
assert cfg['host'] in ['dev4','dev5','dev6','dev7']
name=f"xiaopang1105_dsv41_{cfg['role'].lower()}_{cfg['host']}_{args.run_id}"
assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0,'Existing task container; inspect it before relaunch'
gmi=subprocess.check_output(['mthreads-gmi'],text=True);cards=re.findall(r'(\d+)%\s+(\d+)MiB\((\d+)MiB\)',gmi)
assert len(cards)==8 and all(util=='0' and memory=='0' for util,memory,total in cards),'All eight GPUs must be unused'
image='registry.mthreads.com/presale/devtech/xllm:1102-dev';iid=subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()
assert iid=='sha256:09b8f5f4a467fc73332dee57b36a2e5610ba3264357dfc87f103cf7bd7430be3','Image changed'
ports=set()
for rank in cfg['ranks']:
 flags=dict(item[2:].split('=',1) for item in rank['command'][1:])
 ports.update([int(flags['port']),int(flags['master_node_addr'].split(':')[-1]),int(flags['disagg_pd_port']),int(flags['transfer_listen_port'])])
 ports.update(range(int(rank['environment']['MC_MIN_RPC_PORT']),int(rank['environment']['MC_MAX_RPC_PORT'])+1))
if cfg['role']=='DECODE':ports.update(cfg['control_ports'].values())
reserved=[]
try:
 for port in sorted(ports):
  sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind((cfg['host_ip'],port));reserved.append(sock)
finally:
 for sock in reserved:sock.close()
# This preflight is not an exclusive reservation: supervise immediately after launch.
out=root/'runs'/args.run_id/cfg['host'];out.mkdir(parents=True,exist_ok=False)
container_out='/work/runs/'+args.run_id+'/'+cfg['host']
config_in='/work/'+str(args.config.relative_to(root))
command=['docker','run','-d','--init','--pull=never','--restart=no','--runtime=mthreads','--network=host','--workdir','/work/repos/xllm-musa','--hostname',name,'--add-host',name+':'+cfg['host_ip'],'--name',name,'--label','owner=xiaopang','--label','task=deepseek-v41-1105','--cpus=64','--memory=800g','--shm-size=32g','--pids-limit=16384','--ulimit','memlock=-1:-1','--ulimit','core=0:0','--ulimit','nofile=1048576:1048576','--cap-add=IPC_LOCK','--env','MTHREADS_VISIBLE_DEVICES=all','--env','MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','--env','OMP_NUM_THREADS=2','--mount',f'type=bind,src={root},dst=/work','--mount','type=bind,src=/data/nfs_shared/models/DeepSeek-V4.1-Flash,dst=/data/nfs_shared/models/DeepSeek-V4.1-Flash,readonly']
for device in sorted(Path('/dev').glob('mtgpu.*')):command+=['--device',str(device)]
for device in sorted(Path('/dev/dri').glob('renderD*')):command+=['--device',str(device)]
for device in sorted(Path('/dev/infiniband').glob('*')):command+=['--device',str(device)]
command+=['--entrypoint','python3',image,'/work/scripts/pd_rank_supervisor.py','--config',config_in,'--output',container_out]
r=subprocess.run(command,capture_output=True,text=True)
(out/'container_launch.json').write_text(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'command':command,'gpu_before':gmi,'ports_checked':sorted(ports),'image_id':iid,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2))
print(json.dumps({'container':name,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}));r.check_returncode()
