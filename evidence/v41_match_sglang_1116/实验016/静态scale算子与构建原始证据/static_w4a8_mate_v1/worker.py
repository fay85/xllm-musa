from pathlib import Path
import os,json,subprocess,datetime
r=Path('/expt/builds/static_w4a8_mate_v1');src=Path('/workspace/xllm-mt-main');env=dict(os.environ,MUSA_VISIBLE_DEVICES='-1',MATE_MUSA_ARCH_LIST='3.1',MATE_WORKSPACE_BASE=str(r/'mate_cache'),PYTHONPATH='/donor')
cmd=['python3',str(src/'tools/musa_deepseek_v41/build_w4a8_mate.py'),'--patched-include','/donor/sglang/python/sglang/jit_kernel/musa_w4a8/include','--output',str(r/'output'),'--activation','static']
(r/'command.json').write_text(json.dumps({'cmd':cmd,'env_delta':{k:env[k] for k in ['MUSA_VISIBLE_DEVICES','MATE_MUSA_ARCH_LIST','MATE_WORKSPACE_BASE','PYTHONPATH']}}))
with (r/'build.log').open('wb') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=600)
(r/'finished.json').write_text(json.dumps({'exit_code':p.returncode,'at':datetime.datetime.now().astimezone().isoformat()}))
