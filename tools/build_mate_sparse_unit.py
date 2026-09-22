#!/usr/bin/env python3
"""Build only the MATE sparse unit, reusing the local kernel archive read-only."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build/dspark-musa-local'
DEST = (ROOT / sys.argv[1]).resolve()
assert DEST.is_relative_to(ROOT / 'build') and not DEST.exists(), DEST
DEST.mkdir(parents=True)
(DEST / 'tmp').mkdir()
env = dict(os.environ, TMPDIR=str(DEST / 'tmp'))
source = ROOT / 'tests/core/kernels/musa/deepseek_v4_mate_sparse_attention_test.cpp'
template = ROOT / 'build/gsm8k_candidates/graph_q2_mate_per_token_diag_20260922/deepseek_v4_attention.cpp.command.txt'
command = shlex.split(template.read_text())
obj = DEST / 'test.o'
command[command.index('-o') + 1] = str(obj)
command[command.index('-c') + 1] = str(source)
commands = [command]
vcpkg = Path('/data/nfs_shared/jiaming.wang/xllmmerge/xllm/build/musa-standalone-db33797/vcpkg_installed/x64-linux/lib')
torchlib = Path('/usr/local/lib/python3.10/dist-packages/torch/lib')
musalib = Path('/usr/local/lib/python3.10/dist-packages/torch_musa/lib')
ffi = Path('/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib')
kernel = BUILD / 'xllm/core/kernels/musa/libcuda_kernels.a'
commands.append(['g++', str(obj), '-o', str(DEST / 'mate_sparse_test'),
    '-Wl,--start-group', str(kernel),
    str(BUILD / 'xllm/core/platform/libplatform.a'),
    str(vcpkg / 'manual-link/libgtest_main.a'), str(vcpkg / 'libgtest.a'),
    str(vcpkg / 'libglog.a'), str(vcpkg / 'libgflags.a'), '-Wl,--end-group',
    '-Wl,--no-as-needed', str(musalib / 'libmusa_python.so'),
    str(torchlib / 'libtorch.so'), str(torchlib / 'libtorch_cpu.so'),
    str(torchlib / 'libtorch_python.so'), str(torchlib / 'libc10.so'),
    '-L' + str(ffi), '-ltvm_ffi', '-L/usr/local/musa/lib', '-lmusa', '-lmusart',
    '-L/usr/local/mudnn/lib', '-lmudnn', '-lpython3.10', '-lpthread', '-ldl', '-lnuma',
    '-Wl,-rpath,' + ':'.join(map(str, [torchlib, musalib, ffi, '/usr/local/musa/lib', '/usr/local/mudnn/lib']))])
for stage, cmd in zip(['compile', 'link'], commands):
    (DEST / (stage + '.sh')).write_text(shlex.join(cmd) + '\n')
    with (DEST / (stage + '.log')).open('w') as log:
        print(stage, flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
(DEST / 'manifest.json').write_text(json.dumps({
    'head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'source': str(source), 'source_sha256': sha(source), 'kernel_archive': str(kernel),
    'kernel_sha256': sha(kernel), 'binary_sha256': sha(DEST / 'mate_sparse_test'),
}, indent=2) + '\n')
print(DEST / 'mate_sparse_test')
