from pathlib import Path
import base64
repo=Path('王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/03_代码/xllm-musa')
source=base64.b64encode((repo/'xllm/core/kernels/musa/deepseek_v41_dense_block32.cpp').read_bytes()).decode()
s='''from pathlib import Path
import os,json,base64,shutil,subprocess,datetime,hashlib
old=Path('/expt/builds/e40_prequant_native_v1');r=Path('/expt/builds/e40_prequant_native_v3');r.mkdir(exist_ok=False);shutil.copytree(old/'src',r/'src');p=r/'src/core/kernels/musa/deepseek_v41_dense_block32.cpp';p.write_bytes(base64.b64decode(SOURCE));(r/'source_manifest.json').write_text(json.dumps({str(f.relative_to(r/'src')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (r/'src').rglob('*') if f.is_file()},indent=2))
compile_cmd=[s.replace(str(old),str(r)) for s in json.loads((old/'commands.json').read_text())[0]]
link_cmd=json.loads(Path('/expt/builds/e40_prequant_native_v2/link_command.json').read_text());link_cmd=[str(r/'0.o') if x==str(old/'0.o') else x for x in link_cmd];link_cmd[link_cmd.index('-o')+1]=str(r/'v41_prequant_projection_test_module.so');(r/'commands.json').write_text(json.dumps([compile_cmd,link_cmd],indent=2))
worker="""from pathlib import Path
import subprocess,json,datetime,hashlib,os
r=Path('/expt/builds/e40_prequant_native_v3');commands=json.loads((r/'commands.json').read_text());rc=1
try:
 for i,cmd in enumerate(commands):
  with (r/(str(i)+'.log')).open('xb') as f:
   p=subprocess.Popen(cmd,cwd='/workspace/xllm-mt-main/build/cmake.linux-x86_64-cpython-310',stdout=f,stderr=subprocess.STDOUT);(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]}));code=p.wait()
  assert code==0,(i,code)
 p=subprocess.run(['/opt/venv/bin/python3','-c','import torch,torch_musa;import v41_prequant_projection_test_module;print("IMPORT_PASS")'],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(r)));(r/'import.log').write_text(p.stdout+p.stderr);assert p.returncode==0;rc=0
finally:
 z={'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat(),'unchanged_objects_from':'/expt/builds/e40_prequant_native_v1'};p=r/'v41_prequant_projection_test_module.so'
 if p.exists():z.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 (r/'finished.json').write_text(json.dumps(z,indent=2))
""";(r/'worker.py').write_text(worker)
py='/opt/venv/lib/python3.10/site-packages';env=dict(os.environ,LD_PRELOAD='',LD_LIBRARY_PATH=py+'/torch_musa/lib:'+py+'/torch/lib:'+py+'/tvm_ffi/lib:/usr/local/musa/lib',MUSAMAPPING_PATH='/usr/local/musa/tools/musamapping',MUSA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2')
with (r/'worker.log').open('xb') as f:p=subprocess.Popen(['/opt/venv/bin/python3',str(r/'worker.py')],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
z={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()};(r/'job.json').write_text(json.dumps(z));print(json.dumps(z))
'''.replace('SOURCE',repr(source));Path('/tmp/match_e40_start_native_v3.py').write_text(s)
s=Path('/tmp/match_e40_prequant_download.py').read_text().replace("mode='prequant_v1'","mode='native_checks_v1'").replace('match_e40_prequant_export.json','match_e40_native_export.json');Path('/tmp/match_e40_native_download.py').write_text(s)
