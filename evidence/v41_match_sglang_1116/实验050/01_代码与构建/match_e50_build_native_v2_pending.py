from pathlib import Path
import base64,json,hashlib,datetime
r=Path('/expt/builds/e50_native_gate_gemv_v2');r.mkdir(exist_ok=False);(r/'src').mkdir()
p=r/'src/v41_gate_gemv_test_module.cpp';p.write_bytes(base64.b64decode('LyogQ29weXJpZ2h0IDIwMjYgVGhlIHhMTE0gQXV0aG9ycy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC4KU1BEWC1MaWNlbnNlLUlkZW50aWZpZXI6IEFwYWNoZS0yLjAKPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki8KI2luY2x1ZGUgPHRvcmNoL2V4dGVuc2lvbi5oPgoKI2luY2x1ZGUgPGNzdGRpbnQ+CiNpbmNsdWRlIDxvcHRpb25hbD4KCiNpbmNsdWRlICJjb3JlL2tlcm5lbHMvbXVzYS9tdXNhX29wc19hcGkuaCIKCm5hbWVzcGFjZSB7CnRvcmNoOjpUZW5zb3IgZ2F0ZV9nZW12KGNvbnN0IHRvcmNoOjpUZW5zb3ImIGlucHV0LAogICAgICAgICAgICAgICAgICAgICAgICBjb25zdCB0b3JjaDo6VGVuc29yJiB3ZWlnaHQpIHsKICByZXR1cm4geGxsbTo6a2VybmVsOjptdXNhOjpkZWVwc2Vla192NF9nYXRlX2dlbXYoCiAgICAgIGlucHV0LCB3ZWlnaHQsIC8qb3V0cHV0PSovc3RkOjpudWxsb3B0KTsKfQoKdG9yY2g6OlRlbnNvciBnYXRlX3JlZmVyZW5jZShjb25zdCB0b3JjaDo6VGVuc29yJiBpbnB1dCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjb25zdCB0b3JjaDo6VGVuc29yJiBmcDMyX3dlaWdodCkgewogIHJldHVybiB0b3JjaDo6bGluZWFyKGlucHV0LnRvKHRvcmNoOjprRmxvYXQzMiksIGZwMzJfd2VpZ2h0KTsKfQp9ICAvLyBuYW1lc3BhY2UKClBZQklORDExX01PRFVMRSh2NDFfZ2F0ZV9nZW12X3Rlc3RfbW9kdWxlLCBtb2R1bGUpIHsKICBtb2R1bGUuZGVmKCJnZW12IiwgJmdhdGVfZ2Vtdik7CiAgbW9kdWxlLmRlZigicmVmZXJlbmNlIiwgJmdhdGVfcmVmZXJlbmNlKTsKICBtb2R1bGUuZGVmKCJyb3V0aW5nIiwgJnhsbG06Omtlcm5lbDo6bXVzYTo6ZGVlcHNlZWtfdjQxX2dhdGVfdG9wayk7Cn0K'))
(r/'source_manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'at':datetime.datetime.now().astimezone().isoformat(),'scope':'isolate existing V4 SIMT GEMV; no model change'},indent=2))
from pathlib import Path
import json,os,subprocess,datetime
r=Path('/expt/builds/e50_native_gate_gemv_v2');assert (r/'source_manifest.json').exists() and not (r/'job.json').exists();old=Path('/expt/builds/e40_prequant_native_v1');template=json.loads((old/'commands.json').read_text())[0];commands=[]
for i,name in enumerate(['v41_gate_gemv_test_module.cpp']):
 cmd=[x.replace(str(old),str(r)) for x in template];cmd[cmd.index('-o')+1]=str(r/f'{i}.o');cmd[cmd.index('-c')+1]=str(r/'src'/name);commands.append(cmd)
link=json.loads(Path('/expt/builds/hc_metadata_production_checks_v1/commands.json').read_text())[1];link=[x for x in link if not x.endswith('/test_module.o')];link[link.index('-o')+1]=str(r/'v41_gate_gemv_test_module.so');link[4:4]=[str(r/'0.o')];commands.append(link);(r/'commands.json').write_text(json.dumps(commands,indent=2))
worker='''from pathlib import Path
import json,os,subprocess,datetime,hashlib
r=Path('/expt/builds/e50_native_gate_gemv_v2');rc=1
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
''';(r/'worker.py').write_text(worker);py='/opt/venv/lib/python3.10/site-packages';env=dict(os.environ,LD_PRELOAD='',LD_LIBRARY_PATH=py+'/torch_musa/lib:'+py+'/torch/lib:'+py+'/tvm_ffi/lib:/usr/local/musa/lib',MUSAMAPPING_PATH='/usr/local/musa/tools/musamapping',MUSA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2')
with (r/'worker.log').open('xb') as f:p=subprocess.Popen(['python3',str(r/'worker.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
v={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()};(r/'job.json').write_text(json.dumps(v));print(json.dumps(v))
