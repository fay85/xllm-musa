from pathlib import Path
import os,json,subprocess,datetime,hashlib
old=Path('/expt/builds/e40_prequant_native_v1');assert json.loads((old/'finished.json').read_text())['exit_code']==1;assert 'cannot find -lmudart' in (old/'link.log').read_text();assert Path('/usr/local/musa/lib/libmusart.so').exists();r=Path('/expt/builds/e40_prequant_native_v2');r.mkdir(exist_ok=False)
cmd=json.loads((old/'commands.json').read_text())[-1];cmd=[x.replace('-lmudart','-lmusart') for x in cmd];cmd[cmd.index('-o')+1]=str(r/'v41_prequant_projection_test_module.so');(r/'link_command.json').write_text(json.dumps(cmd,indent=2));py='/opt/venv/lib/python3.10/site-packages';env=dict(os.environ,LD_PRELOAD='',LD_LIBRARY_PATH=py+'/torch_musa/lib:'+py+'/torch/lib:'+py+'/tvm_ffi/lib:/usr/local/musa/lib',MUSA_VISIBLE_DEVICES='-1',PYTHONPATH=str(r));rc=1
try:
 with (r/'link.log').open('xb') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 assert p.returncode==0,'link';p=subprocess.run(['/opt/venv/bin/python3','-c','import torch,torch_musa;import v41_prequant_projection_test_module;print("IMPORT_PASS")'],capture_output=True,text=True,env=env);(r/'import.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'import';rc=0
finally:
 v={'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat(),'source_manifest':str(old/'source_manifest.json'),'objects_reused_from':str(old)};p=r/'v41_prequant_projection_test_module.so'
 if p.exists():v.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 (r/'finished.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
