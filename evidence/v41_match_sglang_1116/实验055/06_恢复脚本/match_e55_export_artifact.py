from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
r=Path('/expt');build=r/'builds/e55_native_draft_gate_v1';root=Path('/workspace/xllm-mt-main')
assert json.loads((build/'finished.json').read_text())['exit_code']==0
for n,z in json.loads((build/'source_manifest.json').read_text()).items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==z['after']
b=root/'build/cmake.linux-x86_64-cpython-310';v=subprocess.run(['ninja','-C',str(b),'-f','build.sg438.ninja','-n','xllm'],capture_output=True,text=True,timeout=30)
assert v.returncode==0 and 'no work to do' in v.stdout
a=r/'artifacts/native_sg438_draft_gate_v1';a.mkdir(exist_ok=False);(a/'bin').mkdir();binary=a/'bin/xllm'
shutil.copy2(root/'build/lib.linux-x86_64-cpython-310/xllm/xllm',binary)
(a/'lib').symlink_to(r/'artifacts/native_sg438_control_v1/lib',target_is_directory=True)
v={'artifact':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'bytes':binary.stat().st_size,'source_manifest':str(build/'source_manifest.json'),'source_parent':'980d9da1287ba99a12749d89b89c55e0d8ed233f','at':datetime.datetime.now().astimezone().isoformat()}
(build/'artifact.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
