from pathlib import Path
import json,hashlib,datetime,shutil,subprocess,base64
root=Path('/expt');build=root/'builds/e51_native_bit_gather_v1';assert json.loads((build/'finished.json').read_text())['exit_code']==0
source=Path('/workspace/xllm-mt-main');b=source/'build/cmake.linux-x86_64-cpython-310';v=subprocess.run(['ninja','-C',str(b),'-f','build.sg438.ninja','-n','xllm'],capture_output=True,text=True,timeout=30);(build/'dry_run.txt').write_text(v.stdout+v.stderr);assert v.returncode==0 and 'no work to do' in v.stdout
for n,z in json.loads((build/'source_manifest.json').read_text()).items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==z['after']
a=root/'artifacts/native_sg438_bit_gather_v1';a.mkdir(exist_ok=False);(a/'bin').mkdir();binary=a/'bin/xllm';shutil.copy2(source/'build/lib.linux-x86_64-cpython-310/xllm/xllm',binary);(a/'lib').symlink_to(root/'artifacts/native_sg438_control_v1/lib',target_is_directory=True);sha=hashlib.sha256(binary.read_bytes()).hexdigest();record={'artifact':str(binary),'sha256':sha,'bytes':binary.stat().st_size,'source_manifest':str(build/'source_manifest.json'),'source_parent':'a93a4542be60d2d719dee817f069b2a22ab7b0c1','at':datetime.datetime.now().astimezone().isoformat()};(build/'artifact.json').write_text(json.dumps(record,indent=2))

for run,enabled in [('p51_bit_gather_off_v1',False),('p51_bit_gather_on_v1',True)]:
 out=root/'runs'/run;out.mkdir(exist_ok=False);(out/'runtime_timing').mkdir()
 for host in ['sj9','sj10']:
  old='p50_gate_simt_confirm_on_v1';base=json.loads((root/'runs'/old/host/'prepared_config.json').read_text());cfg=json.loads(json.dumps(base).replace(old,run).replace(base['namespace'],'xiaopang1117-'+run).replace(base['binary'],str(binary)))
  cfg.update(binary_sha256=sha,source_manifest=str(build/'source_manifest.json'),source_dir=str(build/'source_after'),source_commit=record['source_parent']+' + E51 source_manifest',treatment='E50 champion + D bit-preserving embedding/logits gather='+str(enabled))
  for rank in cfg['ranks']:
   env=rank['environment'];assert env.get('XLLM_DSV41_GATE_BF16_FP32','0')=='0';assert env.get('XLLM_DSV41_OWNER_DEEPEP_LL','0')=='0'
   assert env['XLLM_DSV41_GATE_SIMT']==('1' if host=='sj10' else '0')
   env['XLLM_DSV41_GATHER_BF16_BITS']='1' if host=='sj10' and enabled else '0'
   if host=='sj10':assert env['XLLM_DSV41_FULL_DRAFT_GRAPH']=='1'
  d=out/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps(record))
