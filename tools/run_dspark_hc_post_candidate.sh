#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
run_dir=${1:?Usage: bash tools/run_dspark_hc_post_candidate.sh logs/NEW_RUN_NAME}

# Reuse the isolated candidate and installed dependencies. Never stop another
# service, release somebody else's reservation, rebuild donors, or reset GPUs.
export DSPARK_BIN=${DSPARK_BIN:-$project_root/build/gsm8k_candidates/hc_post_orientation_model_20260925/xllm}
export MODEL_PATH=${MODEL_PATH:-/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt}
export XLLM2_EXTERNAL_TP8_HTTP_PORT=${XLLM2_EXTERNAL_TP8_HTTP_PORT:-19447}
export XLLM2_EXTERNAL_TP8_MASTER_ADDR=${XLLM2_EXTERNAL_TP8_MASTER_ADDR:-127.0.0.1:30710}
export XLLM2_EXTERNAL_TP8_EP_SIZE=1
export XLLM2_EXTERNAL_TP8_DP_SIZE=1
export XLLM2_EXTERNAL_TP8_NUM_SPECULATIVE_TOKENS=5
export XLLM2_EXTERNAL_TP8_DISABLE_DRAFT=false
export XLLM2_EXTERNAL_TP8_SPECULATIVE_ALGORITHM=DSpark
export XLLM2_EXTERNAL_TP8_GRAPH=true
export XLLM2_EXTERNAL_TP8_PREFIX_CACHE=true
export XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY=1
export XLLM_DSV4_ENABLE_GRAPH_BATCHED_MATE_VERIFY=1
export XLLM_DSV4_GRAPH_SHORT_FULLKV_EAGER=0
export XLLM_DSV4_DISABLE_GRAPH_Q6_FREEZE=1
export XLLM_DSV4_DYNAMIC_GRAPH_STATE=1
export XLLM_DSV4_GRAPH_DYNAMIC_INDEXER=1
export XLLM_DSV4_CLEAR_STATE_CACHE_ON_REQUEST=1
export XLLM_DSV4_CLEAR_DECODE_WORKSPACES_ON_REQUEST=1
export XLLM_DSV4_GRAPH_KEY_TRACE=1

# Fail closed for old diagnostic controls. Report names only, never unrelated
# environment values. Use start_dspark_custom.sh directly for deliberate A/Bs.
while IFS= read -r variable; do
  case "$variable" in
    XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY|XLLM_DSV4_ENABLE_GRAPH_BATCHED_MATE_VERIFY|\
    XLLM_DSV4_GRAPH_SHORT_FULLKV_EAGER|XLLM_DSV4_DISABLE_GRAPH_Q6_FREEZE|\
    XLLM_DSV4_DYNAMIC_GRAPH_STATE|XLLM_DSV4_GRAPH_DYNAMIC_INDEXER|\
    XLLM_DSV4_CLEAR_STATE_CACHE_ON_REQUEST|XLLM_DSV4_CLEAR_DECODE_WORKSPACES_ON_REQUEST|\
    XLLM_DSV4_GRAPH_KEY_TRACE)
      ;;
    XLLM_DSV4_*|XLLM_GRAPH_*|XLLM_MUSA_GRAPH_*|XLLM_KEEP_HOST_MIRROR_REFRESH)
      echo "Unset inherited diagnostic control before acceptance: $variable" >&2
      exit 2
      ;;
  esac
done < <(compgen -e)

exec bash tools/start_dspark_custom.sh "$run_dir" 10240 0.8
