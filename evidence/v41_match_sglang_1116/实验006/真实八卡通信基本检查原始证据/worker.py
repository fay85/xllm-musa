from pathlib import Path
import os,json,subprocess,shlex,sysconfig,datetime,hashlib
r=Path('/expt/builds/verify_collective_v1');b=Path('/workspace/xllm-mt-main/build/cmake.linux-x86_64-cpython-310');rc=1;commands=[]
(r/'job.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()},indent=2))
try:
 rows=json.loads((b/'compile_commands.json').read_text());row=next(x for x in rows if x['file'].endswith('/xllm.cpp'));cmd=shlex.split(row['command']);cmd[cmd.index('-o')+1]=str(r/'test_module.o');cmd[cmd.index('-c')+1]=str(r/'v41_verify_collective_test_module.cpp');cmd=[x for x in cmd if x!='-fPIE'];cmd+=['-fPIC','-O2'];commands.append(cmd)
 with (r/'compile.log').open('wb') as f:p=subprocess.run(cmd,cwd=b,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0 and (r/'test_module.o').exists(),'compile'
 cmd=json.loads(Path('/expt/builds/attention_replica_v1/small_checks_v7/link_command.json').read_text());target=r/('v41_verify_collective_test_module'+sysconfig.get_config_var('EXT_SUFFIX'));cmd[cmd.index('-o')+1]=str(target);cmd=[str(r/'test_module.o') if x.endswith('/test_module.o') else x for x in cmd];index=cmd.index('-Wl,--start-group')+1;cmd[index:index]=[str(b/'xllm/core/framework/parallel_state/libparallel_state.a'),str(b/'xllm/core/framework/config/libconfig.a')];commands.append(cmd)
 with (r/'link.log').open('wb') as f:p=subprocess.run(cmd,cwd=b,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0 and target.exists(),'link'
 cfg=json.loads(Path('/expt/runs/p2_full_decode_graph_v1/sj9/role_config.json').read_text());env=dict(os.environ,LD_LIBRARY_PATH=cfg['ranks'][0]['environment']['LD_LIBRARY_PATH'],LD_PRELOAD='',PYTHONPATH=str(r),MUSA_VISIBLE_DEVICES='-1')
 p=subprocess.run(['python3','-c','import torch,torch_musa;import v41_verify_collective_test_module;print("IMPORT_PASS")'],env=env,capture_output=True,text=True);(r/'import.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'import';rc=0
finally:
 (r/'commands.json').write_text(json.dumps(commands,indent=2));state={'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}
 if 'target' in globals() and target.exists():state.update(bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
 (r/'finished.json').write_text(json.dumps(state,indent=2))
