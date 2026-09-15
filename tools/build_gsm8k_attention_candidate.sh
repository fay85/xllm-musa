#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
set -euo pipefail

# Reuse existing libraries; replace only this checkout's Attention object.
# Write a distinct executable instead of replacing the validated binary.
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_dir/build/dspark-musa-local"
output_bin=${1:?supply a new candidate executable under this checkout}
output_bin=$(realpath -m -- "$output_bin")
case "$output_bin" in "$repo_dir"/build/*) ;; *) exit 2 ;; esac
[[ ! -e "$output_bin" ]] || { echo 'Candidate already exists' >&2; exit 2; }
mkdir -p -- "$(dirname -- "$output_bin")"
export TMPDIR
TMPDIR=$(mktemp -d "$repo_dir/logs/gsm8k_compile_tmp.XXXXXX")
export PYTHONDONTWRITEBYTECODE=1

python3 - "$build_dir" "$repo_dir" <<'PY'
import json
import shlex
import subprocess
import sys
from pathlib import Path

build, root = map(Path, sys.argv[1:])
donor = Path('/data/nfs_shared/jiaming.wang/xllmmerge/xllm0814/xllm-dsv4-musa-dspark')
commands = json.loads((build / 'compile_commands.json').read_text())
source = 'xllm/core/layers/musa/deepseek_v4/deepseek_v4_attention.cpp'
command = next(item['command'] for item in commands if item['file'].endswith(source))
command = command.replace('/opt/xllm-build-deps/compiler_launcher.sh ', '')
command = command.replace('-fplugin=/usr/local/musa/tools/musamapping/libMusaMapping.so', '')
command = command.replace(' -x musa ', ' ')
command = command.replace(str(root / 'third_party'), str(donor / 'third_party'))
args = ['g++', '-I/usr/local/musa/include',
        '-I/usr/local/lib/python3.10/dist-packages',
        '-I/usr/local/lib/python3.10/dist-packages/tvm_ffi/include']
subprocess.run(args + shlex.split(command)[1:], cwd=build, check=True)
PY
ar rcs "$build_dir/xllm/core/layers/musa/libmusa_layers.a" \
  "$build_dir/xllm/core/layers/musa/CMakeFiles/musa_layers.dir/deepseek_v4/deepseek_v4_attention.cpp.o"
ranlib "$build_dir/xllm/core/layers/musa/libmusa_layers.a"
link_command=$(ninja -C "$build_dir" -t commands xllm | tail -n 1)
link_command=${link_command/\/opt\/xllm-build-deps\/compiler_launcher.sh /}
link_command=${link_command//\/usr\/local\/lib\/libzstd.so/\/usr\/lib\/x86_64-linux-gnu\/libzstd.so.1}
[[ "$link_command" == *" -o xllm/xllm "* ]] || exit 2
link_command=${link_command/ -o xllm\/xllm / -o "$output_bin" }
(cd "$build_dir" && eval "$link_command")
sha256sum "$output_bin"
