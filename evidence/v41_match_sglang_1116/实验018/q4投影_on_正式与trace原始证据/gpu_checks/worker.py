from pathlib import Path
import json,os,subprocess,datetime
r=Path('/expt/builds/q4_projection_checks_v1');o=r/'gpu_v1';cfg=json.loads(Path('/expt/runs/p15_spec3_current_boundary_timing_v1/sj9/role_config.json').read_text())['ranks'][0]['environment'];env=dict(os.environ,LD_LIBRARY_PATH=cfg['LD_LIBRARY_PATH'],LD_PRELOAD='',PYTHONPATH=str(r)+':/workspace/xllm-mt-main',MUSA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2')
cmd=['python3',str(r/'validate_q4_projection.py'),'--output',str(o/'result.json')]
with (o/'test.log').open('wb') as f:
 p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);(o/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'command':cmd}));rc=p.wait()
(o/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
