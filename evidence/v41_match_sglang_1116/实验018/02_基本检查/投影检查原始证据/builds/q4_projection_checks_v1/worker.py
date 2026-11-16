from pathlib import Path
import os,json,subprocess,shlex,sysconfig,datetime,hashlib
r=Path('/expt/builds/q4_projection_checks_v1');src=Path('/workspace/xllm-mt-main');b=src/'build/cmake.linux-x86_64-cpython-310'
(r/'job.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()},indent=2))
commands=[];rc=1
try:
 rows=json.loads((b/'compile_commands.json').read_text())
 for label,match,source,out in [('module','/xllm.cpp',r/'v41_q4_projection_test_module.cpp',r/'test_module.o')]:
  row=next(x for x in rows if x['file'].endswith(match));cmd=shlex.split(row['command']);cmd[cmd.index('-o')+1]=str(out);cmd[cmd.index('-c')+1]=str(source);cmd=[x for x in cmd if x!='-fPIE'];cmd+=['-fPIC','-O2'];commands.append(cmd)
  with (r/(label+'_compile.log')).open('wb') as f:res=subprocess.run(cmd,cwd=b,stdout=f,stderr=subprocess.STDOUT)
  assert res.returncode==0 and out.exists(),label
 import shutil
 assert json.loads(Path('/expt/builds/q4_skinny_projection_v1/finished.json').read_text())['exit_code']==0
 copied=[]
 for match,out in [('/skinny_gemm.cu',r/'static_w4a8.o')]:
  row=next(x for x in rows if x['file'].endswith(match));cmd=shlex.split(row['command']);obj=b/cmd[cmd.index('-o')+1];assert obj.exists();shutil.copy2(obj,out);copied.append({'source':str(obj),'sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),'compile_command':row['command']})
 (r/'production_object_manifest.json').write_text(json.dumps(copied,indent=2))
 cmd=json.loads(Path('/expt/builds/attention_replica_v1/small_checks_v7/link_command.json').read_text());target=r/('v41_q4_projection_test_module'+sysconfig.get_config_var('EXT_SUFFIX'));cmd[cmd.index('-o')+1]=str(target)
 cmd=[str(r/'test_module.o') if x.endswith('/test_module.o') else x for x in cmd];cmd.insert(cmd.index('-Wl,--start-group'),str(r/'static_w4a8.o'));commands.append(cmd)
 with (r/'link.log').open('wb') as f:res=subprocess.run(cmd,cwd=b,stdout=f,stderr=subprocess.STDOUT)
 assert res.returncode==0 and target.exists(),'link'
 cfg=json.loads(Path('/expt/runs/p2_full_decode_graph_v1/sj9/role_config.json').read_text());env=dict(os.environ,LD_LIBRARY_PATH=cfg['ranks'][0]['environment']['LD_LIBRARY_PATH'],LD_PRELOAD='',PYTHONPATH=str(r),MUSA_VISIBLE_DEVICES='-1')
 p=subprocess.run(['python3','-c','import torch,torch_musa;import v41_q4_projection_test_module;print("IMPORT_PASS")'],env=env,capture_output=True,text=True);(r/'import.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'import'
 rc=0
finally:
 (r/'commands.json').write_text(json.dumps(commands,indent=2));state={'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}
 if 'target' in globals() and target.exists():state.update(size=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
 (r/'finished.json').write_text(json.dumps(state,indent=2))
