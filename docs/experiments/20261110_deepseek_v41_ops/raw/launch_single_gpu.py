"""Run a finite, communication-free test on a currently unused dev7 GPU 0."""
from pathlib import Path
import argparse
import json
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--attempt', required=True)
parser.add_argument('--with-fixture', action='store_true')
args = parser.parse_args()
assert re.fullmatch(r'v[0-9]+', args.attempt)
root = Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离')
task = root / '20261110_ops'
name = 'xiaopang1110_v41_swiglu_dev7_' + args.attempt
gmi = subprocess.check_output(['mthreads-gmi'], text=True, timeout=15)
cards = re.findall(r'(\d+)%\s+(\d+)MiB\((\d+)MiB\)', gmi)
assert len(cards) == 8 and cards[0][:2] == ('0', '0'), 'GPU 0 is in use; no launch'
assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
prefix = '/work/20261110_ops/evidence/' + args.attempt
script = f'''cd /work/repos/xllm-musa
export PYTHONPATH=/work/20261110_ops/artifacts:/work/repos/xllm-musa
python3 tools/musa_deepseek_v41/validate_shared_swiglu.py --output {prefix}.json > {prefix}.log 2>&1
rc=$?
echo "$rc" > {prefix}.exit
if [ "$rc" != 0 ]; then exit "$rc"; fi
'''
if args.with_fixture:
    script += f'''for mode in reference v4_fused; do
python3 tools/musa_deepseek_v41/validate_csa2_fixture.py --output {prefix}_${{mode}}_fixture.json --shared-swiglu "$mode" --save-tensors {prefix}_${{mode}}.pt > {prefix}_${{mode}}_fixture.log 2>&1
rc=$?
echo "$rc" > {prefix}_${{mode}}_fixture.exit
if [ "$rc" != 0 ]; then exit "$rc"; fi
done
python3 tools/musa_deepseek_v41/compare_swiglu_fixtures.py {prefix}_reference.pt {prefix}_v4_fused.pt {prefix}_fixture_comparison.json > {prefix}_fixture_comparison.log 2>&1
rc=$?
echo "$rc" > {prefix}_fixture_comparison.exit
exit "$rc"
'''
command = [
    'docker', 'run', '-d', '--init', '--pull=never', '--runtime=mthreads',
    '--network=none', '--restart=no', '--name', name, '--label', 'owner=xiaopang',
    '--label', 'task=dsv41-1110-shared-swiglu', '--cpus=4', '--memory=24g',
    '--shm-size=1g', '--pids-limit=2048', '--ulimit', 'core=0:0',
    '--device=/dev/mtgpu.0', '--device=/dev/dri/renderD128',
    '--env', 'MTHREADS_VISIBLE_DEVICES=0', '--env', 'MUSA_VISIBLE_DEVICES=0',
    '--env', 'OMP_NUM_THREADS=2', '--env', 'MKL_NUM_THREADS=2',
    '--mount', f'type=bind,src={root},dst=/work', '--entrypoint', '/bin/bash',
    'registry.mthreads.com/presale/devtech/xllm:1102-dev', '-c', script,
]
result = subprocess.run(command, capture_output=True, text=True, timeout=45)
(task / 'evidence' / (args.attempt + '_launch.json')).write_text(json.dumps({
    'command': command, 'gpu_before': gmi, 'returncode': result.returncode,
    'stdout': result.stdout, 'stderr': result.stderr,
}, indent=2))
print(json.dumps({'name': name, 'rc': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}))
result.check_returncode()
