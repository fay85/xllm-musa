#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
set -euo pipefail

# Reproduce the pinned N6/q7 candidate. Acceptance status lives in the report,
# not in this filename. Never stop another server/holder or reset any GPU.
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$project_root"
run_dir=${1:?provide a NEW logs/run directory}
http_port=${2:-19466}
master_port=${3:-31230}
binary=build/gsm8k_candidates/grouped_state_model_20260926/xllm
base_run=logs/goal_gsm95_tpot10_short_c4_20260926

check_hash() {
  local expected=$1 path=$2 actual
  read -r actual _ < <(sha256sum -- "$path")
  [[ "$actual" == "$expected" ]] || {
    echo "Recorded artifact changed: $path" >&2
    exit 2
  }
}
check_hash 8228891e3c9a4994c38131a869738a477a6fdef8a581c846ee693a26f77b5c87 "$binary"
check_hash 63a6d253bf5b0ef0334f0663cd3d22658e13df1671c392219314235042f9b62c "$base_run/launch_manifest.txt"
check_hash ac89bf0343c4e72fff711e066568079c135d710519d90fb4d39aee6a5c804377 tools/launch_dspark_tuning.py
check_hash a42b73cd84d2f1a467865520385b4bb270e1e24586616384c0a4329d66aa5a98 tools/start_dspark_custom.sh
ops_manifest=logs/goal_gsm96_tpot9_n6_groupedstate_20260926/ops_artifacts.sha256
check_hash c9f9a0b32cb87c33e3edcce537c42f82cb3ec05d3c6bac82659151f35df29d21 "$ops_manifest"
sha256sum --quiet -c "$ops_manifest"

# These two host thread overrides were separately tested, not part of the
# selected configuration. The launcher clears inherited XLLM tuning controls.
unset OMP_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
exec python3 tools/launch_dspark_tuning.py \
  --run-dir "$run_dir" --binary "$binary" --base-run "$base_run" \
  --port "$http_port" --master-port "$master_port" --draft-tokens 6 \
  --set-env XLLM_DSV4_FUSED_PAGED_SAS=1 \
  --set-env XLLM_DSV4_HC_PRE_SMALL_TILE=1 \
  --set-env XLLM_DSV4_FAST_GREEDY_VERIFY=1 \
  --set-env XLLM_DSV4_GRAPH_JOINT_COMPRESSOR_PROJECTION=1 \
  --set-env XLLM_DSV4_GROUPED_STATE_GATHER=1
