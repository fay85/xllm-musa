from pathlib import Path
import json,subprocess,shlex,sysconfig,hashlib,os,datetime
r=Path('/expt/builds/markov_logits_production_checks_v1');s=Path('/workspace/xllm-mt-main');b=s/'build/cmake.linux-x86_64-cpython-310';rc=1;commands=[]
try:
 assert json.loads(Path('/expt/builds/markov_fp32_logits_native_v1/finished.json').read_text())['exit_code']==0
 rows=json.loads((b/'compile_commands.json').read_text());row=next(x for x in rows if x['file'].endswith('/xllm.cpp'));cmd=shlex.split(row['command']);cmd[cmd.index('-o')+1]=str(r/'test_module.o');cmd[cmd.index('-c')+1]=str(s/'tests/core/kernels/musa/v41_markov_test_module.cpp');cmd=[x for x in cmd if x!='-fPIE'];cmd+=['-fPIC'];commands.append(cmd)
 with (r/'compile.log').open('wb') as f:p=subprocess.run(cmd,cwd=b,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0 and (r/'test_module.o').exists()
 cmd=json.loads(Path('/expt/builds/attention_replica_v1/small_checks_v7/link_command.json').read_text());target=r/('v41_markov_test_module'+sysconfig.get_config_var('EXT_SUFFIX'));cmd[cmd.index('-o')+1]=str(target);cmd=[str(r/'test_module.o') if x.endswith('/test_module.o') else x for x in cmd];commands.append(cmd)
 with (r/'link.log').open('wb') as f:p=subprocess.run(cmd,cwd=b,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0 and target.exists()
 cfg=json.loads(Path('/expt/runs/p29_parallel_gate_on_v2/sj9/role_config.json').read_text())['ranks'][0]['environment'];env=dict(os.environ,LD_LIBRARY_PATH=cfg['LD_LIBRARY_PATH'],LD_PRELOAD='',PYTHONPATH=str(r),MUSA_VISIBLE_DEVICES='-1');p=subprocess.run(['python3','-c','import torch,torch_musa;import v41_markov_test_module;print("IMPORT_PASS")'],env=env,capture_output=True,text=True);(r/'import.log').write_text(p.stdout+p.stderr);assert p.returncode==0;rc=0
finally:
 (r/'commands.json').write_text(json.dumps(commands,indent=2));v={'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}
 if 'target' in globals() and target.exists():v.update(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size)
 (r/'finished.json').write_text(json.dumps(v,indent=2))
