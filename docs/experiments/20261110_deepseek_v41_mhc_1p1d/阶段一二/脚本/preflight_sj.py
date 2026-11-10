"""Validate the compiled operators on actual SJ hardware, only between runs."""
from pathlib import Path
import sys,shlex,json,time
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b.parents[2]/'sj9_12_登陆工具'));import sj_access
root='/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops'
script=r'''from pathlib import Path
import json,subprocess,re,hashlib
root=Path(ROOT);out=root/'02_correctness/sj9_operator_preflight';out.mkdir(exist_ok=False)
gmi=subprocess.check_output(['mthreads-gmi'],text=True);cards=re.findall(r'(\d+)%\s+(\d+)MiB\((\d+)MiB\)',gmi);assert len(cards)==8 and all(u=='0' and m=='0' for u,m,_ in cards),'All eight cards must be idle'
manifest=json.loads((root/'00_manifest/test_module.json').read_text());module=root/'artifacts/test_module/musa_python_kernel_test_module.cpython-310-x86_64-linux-gnu.so';assert hashlib.sha256(module.read_bytes()).hexdigest()==manifest['module_sha256']
(out/'gpu_before.log').write_text(gmi)
shell="""set -e
cd /work/01_sources/stage2_mhc
export PYTHONPATH=/work/artifacts/test_module:/work/01_sources/stage2_mhc
export MUSA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python3 -c 'import torch,torch_musa,json; print(json.dumps({"device":str(torch.musa.get_device_properties(0)),"capability":torch.musa.get_device_capability(0),"torch":torch.__version__},indent=2))' > /work/02_correctness/sj9_operator_preflight/hardware.json
python3 tools/musa_deepseek_v41/validate_shared_swiglu.py --output /work/02_correctness/sj9_operator_preflight/swiglu.json > /work/02_correctness/sj9_operator_preflight/swiglu.log 2>&1
python3 /work/scripts/validate_hyper_connection_final.py --checkpoint /work/artifacts/hc_1017_subset --output /work/02_correctness/sj9_operator_preflight/mhc.json > /work/02_correctness/sj9_operator_preflight/mhc.log 2>&1
"""
(root/'scripts/sj9_operator_preflight.sh').write_text(shell)
cmd=['docker','exec','-d','xiaopang1110_v41_prefill_sj9','bash','-c','bash /work/scripts/sj9_operator_preflight.sh > /work/02_correctness/sj9_operator_preflight/runner.log 2>&1; echo $? > /work/02_correctness/sj9_operator_preflight/exit']
subprocess.run(cmd,check=True);print('SJ operator preflight started with no inference ranks running')
'''.replace('ROOT',repr(root))
with sj_access.ssh_client('sj9') as c:
 r=sj_access.execute(c,'sj9','python3 -c '+shlex.quote(script),root=True,timeout=20);assert r['exit_status']==0,r;print(r['stdout'],flush=True)
for _ in range(60):
 with sj_access.ssh_client('sj9') as c:
  r=sj_access.execute(c,'sj9','python3 -c '+shlex.quote(f'from pathlib import Path;import json;p=Path({root!r})/"02_correctness/sj9_operator_preflight/exit";print(json.dumps(p.read_text() if p.exists() else None))'),root=True,timeout=15)
 assert r['exit_status']==0,r
 code=json.loads(r['stdout'])
 if code is not None:
  assert code.strip()=='0','SJ operator validation failed; inspect logs before full model launch'
  with sj_access.ssh_client('sj9') as c:
   with c.open_sftp() as f:
    out=b/'证据/sj9_operator_preflight';out.mkdir(exist_ok=True)
    for name in ['hardware.json','swiglu.json','swiglu.log','mhc.json','mhc.log','runner.log','exit','gpu_before.log']:f.get(root+'/02_correctness/sj9_operator_preflight/'+name,str(out/name),prefetch=False)
  assert json.loads((out/'swiglu.json').read_text())['status']==json.loads((out/'mhc.json').read_text())['status']=='PASS'
  print('SJ operator validation passed',flush=True);break
 time.sleep(3)
else:raise RuntimeError('SJ operator preflight exceeded three minutes; no forced termination')
