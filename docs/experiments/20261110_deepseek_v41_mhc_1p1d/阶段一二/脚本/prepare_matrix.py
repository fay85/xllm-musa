from pathlib import Path
import sys,shlex,json
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b.parents[2]/'sj9_12_登陆工具'));import sj_access
script=r'''from pathlib import Path
import json,hashlib
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops')
for run,parts in [('stage1_swiglu_d',()),('stage2_post',('POST',)),('stage2_post_pre',('POST','PRE')),('stage2_all',('POST','PRE','METADATA'))]:
 out=root/'artifacts/prepared'/run;out.mkdir(exist_ok=False)
 for host in ('sj9','sj10'):
  cfg=json.loads((root/f'artifacts/prepared/stage1_ref/{host}.json').read_text().replace('stage1_ref',run))
  source='/work/01_sources/stage2_mhc' if run.startswith('stage2') else '/work/01_sources/34815f6a'
  binary='/work/artifacts/stage2_mhc/xllm' if run.startswith('stage2') else '/work/artifacts/bin/xllm'
  cfg['binary']=binary;cfg['binary_sha256']=hashlib.sha256((root/binary.removeprefix('/work/')).read_bytes()).hexdigest();cfg['source_dir']=source
  cfg['shared_expert_parallelism']='replicated';cfg['source_manifest']='00_manifest/stage2_source.json' if run.startswith('stage2') else '00_manifest/stage1_source.json'
  for rank in cfg['ranks']:
   rank['command'][0]=binary;rank['command']=[f'--python_model_path={source}' if x.startswith('--python_model_path=') else x for x in rank['command']]
   env=rank['environment'];env['PYTHONPATH']=source;env['XLLM_DSV41_MUSA_SHARED_SWIGLU']='v4_fused' if host=='sj10' else 'reference'
   for part in ('POST','PRE','METADATA'):env['XLLM_DSV41_MUSA_HC_'+part]='v4_fused' if part in parts else 'reference'
  (out/f'{host}.json').write_text(json.dumps(cfg,indent=2))
 print(run,'prepared, not launched')
'''
with sj_access.ssh_client('sj9') as c:
 r=sj_access.execute(c,'sj9','python3 -c '+shlex.quote(script),root=True,timeout=30);(b/'证据/matrix_prepared.json').write_text(json.dumps(r,indent=2));print(r)
