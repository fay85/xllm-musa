from pathlib import Path
import json,hashlib,datetime,shutil,subprocess,base64
root=Path('/expt');build=root/'builds/e53_native_rope_qdq_v1';assert json.loads((build/'finished.json').read_text())['exit_code']==0
source=Path('/workspace/xllm-mt-main');b=source/'build/cmake.linux-x86_64-cpython-310';v=subprocess.run(['ninja','-C',str(b),'-f','build.sg438.ninja','-n','xllm'],capture_output=True,text=True,timeout=30);(build/'dry_run.txt').write_text(v.stdout+v.stderr);assert v.returncode==0 and 'no work to do' in v.stdout
for n,z in json.loads((build/'source_manifest.json').read_text()).items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==z['after']
a=root/'artifacts/native_sg438_rope_qdq_v1';a.mkdir(exist_ok=False);(a/'bin').mkdir();binary=a/'bin/xllm';shutil.copy2(source/'build/lib.linux-x86_64-cpython-310/xllm/xllm',binary);(a/'lib').symlink_to(root/'artifacts/native_sg438_control_v1/lib',target_is_directory=True);sha=hashlib.sha256(binary.read_bytes()).hexdigest();record={'artifact':str(binary),'sha256':sha,'bytes':binary.stat().st_size,'source_manifest':str(build/'source_manifest.json'),'source_parent':'d19631240a283e17295a5823f19e79f8dec6c991','at':datetime.datetime.now().astimezone().isoformat()};(build/'artifact.json').write_text(json.dumps(record,indent=2))

print(json.dumps(record))
