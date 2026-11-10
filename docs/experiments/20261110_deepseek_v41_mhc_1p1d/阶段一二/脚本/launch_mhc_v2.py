from pathlib import Path
import sys,shlex,json
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE.parents[2]/'sj9_12_登陆工具'));import sj_access
script=r'''from pathlib import Path
import subprocess,json,re
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离');task=root/'20261110_ops/stage2';name='xiaopang1110_v41_mhc_dev6_v2'
assert (task/'evidence/build.exit').read_text().strip()=='0'
gmi=subprocess.check_output(['mthreads-gmi'],text=True);cards=re.findall(r'(\d+)%\s+(\d+)MiB\((\d+)MiB\)',gmi);assert len(cards)==8 and cards[0][:2]==('0','0')
assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
shell="""cd /work/repos/xllm-musa
export PYTHONPATH=/work/20261110_ops/stage2/artifacts:/work/repos/xllm-musa
python3 /work/20261110_ops/stage2/scripts/diagnose_mhc.py > /work/20261110_ops/stage2/evidence/mhc_diag_v2.log 2>&1
python3 tools/musa_deepseek_v41/validate_hyper_connection.py --output /work/20261110_ops/stage2/evidence/mhc_v2.json > /work/20261110_ops/stage2/evidence/mhc_v2.log 2>&1
rc=$?
echo $rc > /work/20261110_ops/stage2/evidence/mhc_v2.exit
if [ $rc != 0 ]; then exit $rc; fi
for mode in reference post post_pre all; do
python3 tools/musa_deepseek_v41/validate_csa2_fixture.py --mhc "$mode" --output /work/20261110_ops/stage2/evidence/mhc_v2_${mode}_fixture.json --save-tensors /work/20261110_ops/stage2/evidence/mhc_v2_${mode}.pt > /work/20261110_ops/stage2/evidence/mhc_v2_${mode}_fixture.log 2>&1
rc=$?
echo $rc > /work/20261110_ops/stage2/evidence/mhc_v2_${mode}_fixture.exit
if [ $rc != 0 ]; then exit $rc; fi
done
"""
cmd=['docker','run','-d','--init','--pull=never','--runtime=mthreads','--network=none','--restart=no','--name',name,'--label','owner=xiaopang','--label','task=v41-mhc-1110','--cpus=4','--memory=24g','--shm-size=1g','--pids-limit=2048','--ulimit','core=0:0','--device=/dev/mtgpu.0','--device=/dev/dri/renderD128','--env','MTHREADS_VISIBLE_DEVICES=0','--env','MUSA_VISIBLE_DEVICES=0','--env','OMP_NUM_THREADS=2','--env','MKL_NUM_THREADS=2','--mount',f'type=bind,src={root},dst=/work','--entrypoint','/bin/bash','registry.mthreads.com/presale/devtech/xllm:1102-dev','-c',shell]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
record={'name':name,'command':cmd,'gpu_before':gmi,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr};(task/'evidence/mhc_v2_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps({'name':name,'exit':r.returncode,'id':r.stdout.strip(),'error':r.stderr}));r.check_returncode()
'''
with sj_access.ssh_client('dev6') as c:
 r=sj_access.execute(c,'dev6','python3 -c '+shlex.quote(script),root=True,timeout=50);(BASE/'证据/mhc_v2_launch.json').write_text(json.dumps(r,indent=2));print(r)
