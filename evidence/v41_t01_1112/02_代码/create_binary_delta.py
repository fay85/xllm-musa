from pathlib import Path
import sys,json
W=Path.cwd();E=W/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261112新的开始/ToDoList_4';sys.path.insert(0,str(W/'sj9_12_登陆工具'));import sj_access as a
R='/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261112_sglang_mt_cpp'
code=r'''import pathlib,subprocess,hashlib,json,datetime
r=pathlib.Path(ROOT);e=r/'new_start_t01_tp_decode_build_v1';result=json.loads((e/'result.json').read_text());assert result['exit_code']==0
base=r/'new_start_g02_v2_build/xllm';target=e/'xllm';assert hashlib.sha256(base.read_bytes()).hexdigest()=='896191bc0a14d8d0e7f5e96c15d090250b3a2dc637416cfec4c374f141d70cdf'
cmd=['zstd','-q','-f','--long=29','--patch-from='+str(base),str(target),'-o',str(e/'xllm.zstpatch')]
with open(e/'delta.log','wb') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'command':cmd,'binary_sha256':result['sha256'],'bytes':target.stat().st_size}))
'''.replace('ROOT',repr(R))
with a.ssh_client('dev6') as c:
 z=a.execute(c,'dev6','python3 -',root=True,timeout=30,payload=code);(E/'02_代码/delta_launch.json').write_text(json.dumps(z,indent=2));assert z['exit_status']==0;print(z['stdout'])
