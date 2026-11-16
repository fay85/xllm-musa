from pathlib import Path
import json,os,subprocess,datetime
r=Path('/expt/builds/static_w4a8_checks_v2');o=r/'gpu_v1';cfg=json.loads(Path('/expt/runs/p15_spec3_current_boundary_timing_v1/sj9/role_config.json').read_text())['ranks'][0]['environment'];env=dict(os.environ,LD_LIBRARY_PATH=cfg['LD_LIBRARY_PATH'],LD_PRELOAD='',PYTHONPATH=str(r)+':/workspace/xllm-mt-main',MUSA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2')
cmd=['python3',str(r/'validate_static_w4a8.py'),'--fixture','/work/correctness/fixtures/w4afp8_mt_layer0_experts0_1.safetensors','--gemm-so','/expt/builds/static_w4a8_mate_v1/output/libdeepseek_v41_w4a8_static.so','--output',str(o/'result.json')]
with (o/'test.log').open('wb') as f:
 p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);(o/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'command':cmd}));rc=p.wait()
(o/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
