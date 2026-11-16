from pathlib import Path
import json,os,subprocess,datetime
r=Path('/expt/builds/mhc_q4_graph_checks_v1');cfg=json.loads(Path('/expt/runs/p21_replica_off_v2/sj9/role_config.json').read_text());env=dict(os.environ,LD_LIBRARY_PATH='/work/artifacts/native_mhc_small_batch/lib:'+cfg['ranks'][0]['environment']['LD_LIBRARY_PATH'],LD_PRELOAD='',PYTHONPATH='/work/artifacts/native_mhc_small_batch/lib:/workspace/xllm-mt-main',MUSA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',TVM_FFI_GPU_BACKEND='musa',XLLM_DSV41_MHC_SMALL_BATCH_FUSION='1')
cmd=['python3','/workspace/xllm-mt-main/tools/musa_deepseek_v41/validate_small_batch_mhc_graph.py','--fixture','/work/correctness/fixtures/w4afp8_mt_layer0_mhc.safetensors','--library-directory','/work/artifacts/native_stage_i/lib/mate','--output',str(r/'result.json')]
with (r/'test.log').open('wb') as f:
 p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'command':cmd}));rc=p.wait()
(r/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
