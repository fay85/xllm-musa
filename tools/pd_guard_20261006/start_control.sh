#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
set -euo pipefail
run=${1:?usage inside 92/cmz: start_control.sh NEW_RUN_DIR}
[[ "$(hostname -s)" == worker38092 && "$run" == /* && ! -e "$run" ]]
binary=/workspace/xllm-service/build-cmake331-native/xllm_service/xllm_master_serving
printf '%s  %s\n' cd18acb99e2f6415a2a83c3791e9bee93a888e132956ca63a54aed0bb7a0f627 "$binary" | sha256sum -c -
curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:2489/health
python3 - <<'PY'
import socket
for port in (29888, 29889):
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', port))
PY
mkdir -p -- "$(dirname -- "$run")"
mkdir -- "$run"
unset XLLM_SERVICE_ASSUME_PRELINKED_PD LD_PRELOAD
export LD_LIBRARY_PATH=/workspace/xllm-service/build-cmake331-native/xllm_service/tokenizer/tokenizers/x86_64-unknown-linux-gnu/release:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/openmpi/lib
echo "$$" > "$run/control.pid"
sha256sum "$binary" > "$run/binary.sha256"
exec "$binary" --etcd_addr=127.0.0.1:2489 --http_server_port=29888 --rpc_server_port=29889 \
  --tokenizer_path=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt \
  --logtostderr=true > "$run/control.log" 2>&1
