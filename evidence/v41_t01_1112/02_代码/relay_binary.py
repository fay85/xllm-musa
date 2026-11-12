from pathlib import Path
import sys,json,hashlib,tempfile
W=Path.cwd();E=W/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261112新的开始/ToDoList_4';sys.path.insert(0,str(W/'sj9_12_登陆工具'));import sj_access as a
root='/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261112_sglang_mt_cpp'
with a.ssh_client('dev6') as source:
 code='import pathlib,hashlib,json,subprocess\np=pathlib.Path('+repr(root+'/new_start_t01_tp_decode_build_v1/xllm.zstpatch')+')\nr=json.loads(p.with_name("result.json").read_text());assert r["exit_code"]==0\nlive=subprocess.run(["pgrep","-f","^zstd .*new_start_t01_tp_decode_build_v1/xllm"],capture_output=True,text=True).stdout\nassert not live,live\nprint(json.dumps({"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"binary_sha256":r["sha256"]}))'
 z=a.execute(source,'dev6','python3 -',root=True,timeout=30,payload=code);assert z['exit_status']==0,z;meta=json.loads(z['stdout']);print(meta,flush=True)
 with tempfile.TemporaryDirectory(prefix='xiaopang_g02_relay_') as tmp:
  local=Path(tmp)/'delta.zst'
  with source.open_sftp() as s:s.get(root+'/new_start_t01_tp_decode_build_v1/xllm.zstpatch',str(local));s.get(root+'/new_start_t01_tp_decode_build_v1/result.json',str(E/'02_代码/build_result.json'))
  assert local.stat().st_size==meta['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest()==meta['sha256']
  with a.ssh_client('sj9') as dest:
   incoming='/tmp/xiaopang_g02_'+meta['sha256'][:16]+'.zst'
   with dest.open_sftp() as s:s.put(str(local),incoming)
   code=r'''import pathlib,json,hashlib,subprocess
r=pathlib.Path(ROOT);incoming=pathlib.Path(INCOMING);meta=META
assert hashlib.sha256(incoming.read_bytes()).hexdigest()==meta['sha256']
o=r/'artifacts/native_t01_decode_tp_ar';o.mkdir(exist_ok=False);(o/'bin').mkdir()
base=r/'artifacts/native_g02_v2_timing/bin/xllm';assert hashlib.sha256(base.read_bytes()).hexdigest()=='896191bc0a14d8d0e7f5e96c15d090250b3a2dc637416cfec4c374f141d70cdf'
subprocess.run(['zstd','-q','-d','--long=29','--patch-from='+str(base),str(incoming),'-o',str(o/'bin/xllm')],check=True)
assert hashlib.sha256((o/'bin/xllm').read_bytes()).hexdigest()==meta['binary_sha256'];(o/'bin/xllm').chmod(0o755);(o/'lib').symlink_to('../native_stage_i/lib');(o/'manifest.json').write_text(json.dumps(meta,indent=2));incoming.unlink();print(json.dumps({'status':'VERIFIED_NEW_BINARY_SHARED_WITH_SJ10','artifact':str(o),**meta}))
'''.replace('ROOT',repr(root)).replace('INCOMING',repr(incoming)).replace('META',repr(meta))
   z=a.execute(dest,'sj9','python3 -',root=True,timeout=90,payload=code);(E/'02_代码/relay_receipt.json').write_text(json.dumps(z,ensure_ascii=False,indent=2));assert z['exit_status']==0,z;print(z['stdout'])
print('Local relay temporary files removed')
