from pathlib import Path
import subprocess,json,os,shutil
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离')
d=root/'artifacts/docker_dev4';d.mkdir(exist_ok=True)
assert shutil.disk_usage(d).free>100*1024**3
assert not Path('/run/xiaopang1105-docker-dev4.sock').exists()
config=d/'daemon.json'
settings={'data-root':str(d/'data'),'exec-root':'/run/xiaopang1105-docker-dev4','pidfile':'/run/xiaopang1105-docker-dev4.pid','hosts':['unix:///run/xiaopang1105-docker-dev4.sock'],'bridge':'none','iptables':False,'ip-forward':False,'ip-masq':False,'storage-driver':'overlay2','runtimes':{'mthreads':{'path':'/usr/bin/musa/mthreads-container-runtime'}},'default-runtime':'runc','containerd-namespace':'xiaopang1105-dev4','containerd-plugins-namespace':'xiaopang1105-dev4-plugins'}
config.write_text(json.dumps(settings,indent=2));log=(d/'daemon.log').open('ab',buffering=0)
p=subprocess.Popen(['/usr/bin/dockerd','--config-file',str(config)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(d/'launch.json').write_text(json.dumps({'pid':p.pid,'config':settings,'purpose':'Task-only Docker daemon; original Docker service untouched; host network only'},indent=2));print(json.dumps({'pid':p.pid,'root':str(d)}))
