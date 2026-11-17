from pathlib import Path
import json,os,subprocess,datetime,hashlib
r=Path('/expt/builds/e50_native_gate_gemv_v3');rc=1
try:
 for i,cmd in enumerate(json.loads((r/'commands.json').read_text())):
  with (r/f'{i}.log').open('xb') as f:
   p=subprocess.Popen(cmd,cwd='/workspace/xllm-mt-main/build/cmake.linux-x86_64-cpython-310',stdout=f,stderr=subprocess.STDOUT);(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]}));code=p.wait()
  assert code==0,(i,code)
 p=subprocess.run(['python3','-c','import torch,torch_musa; import v41_gate_gemv_test_module; print("IMPORT_PASS")'],env=dict(os.environ,PYTHONPATH=str(r)),capture_output=True,text=True);(r/'import.log').write_text(p.stdout+p.stderr);assert p.returncode==0;rc=0
finally:
 val={'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()};p=r/'v41_gate_gemv_test_module.so'
 if p.exists():val.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 (r/'finished.json').write_text(json.dumps(val,indent=2))
