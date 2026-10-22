#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$project_root"
run_dir=$(realpath -m -- "${1:?provide a NEW logs/run directory}")
case "$run_dir" in
  "$project_root"/logs/*) ;;
  *) echo "Run records must stay inside this checkout's logs" >&2; exit 2 ;;
esac

export DSPARK_BIN="$project_root/build/gsm8k_candidates/short_c4_causal_model_20260926/xllm"
export MODEL_PATH=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt
export XLLM2_EXTERNAL_TP8_HTTP_PORT=${XLLM2_EXTERNAL_TP8_HTTP_PORT:-19465}
export XLLM2_EXTERNAL_TP8_MASTER_ADDR=${XLLM2_EXTERNAL_TP8_MASTER_ADDR:-127.0.0.1:30910}
export XLLM2_EXTERNAL_TP8_OPS_PATH=/data/nfs_shared/jiaming.wang/test/workspace_fp8_28/cached_ops
export XLLM2_EXTERNAL_TP8_CUSTOM_AR_SO=/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/xllm0814_perf_goal_20260817/custom_ar_current/libxllm_custom_ar_sglang_current.so
export XLLM2_EXTERNAL_TP8_KV_CACHE_DTYPE=auto
export XLLM2_EXTERNAL_TP8_HOST_BLOCKS_FACTOR=0
export XLLM2_EXTERNAL_TP8_EP_SIZE=1
export XLLM2_EXTERNAL_TP8_DP_SIZE=1
export XLLM2_EXTERNAL_TP8_NUM_SPECULATIVE_TOKENS=5
export XLLM2_EXTERNAL_TP8_DISABLE_DRAFT=false
export XLLM2_EXTERNAL_TP8_SPECULATIVE_ALGORITHM=DSpark
export XLLM2_EXTERNAL_TP8_GRAPH=true
export XLLM2_EXTERNAL_TP8_PREFIX_CACHE=true
export XLLM2_EXTERNAL_TP8_IN_BATCH_PREFIX_CACHE=false
export XLLM2_EXTERNAL_TP8_CHUNKED_PREFILL=true
export XLLM2_EXTERNAL_TP8_PREFILL_THRESHOLD=0.95
export XLLM2_EXTERNAL_TP8_PREFILL_CHUNK=10240
export XLLM2_EXTERNAL_TP8_MAX_GRAPH_TOKENS=2048
export XLLM2_EXTERNAL_TP8_GRAPH_VMM_POOL=true
export XLLM2_EXTERNAL_TP8_ONLINE_PROFILE=false
export XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY=1
export XLLM_DSV4_ENABLE_GRAPH_BATCHED_MATE_VERIFY=1
export XLLM_DSV4_GRAPH_SHORT_FULLKV_EAGER=0
export XLLM_DSV4_DISABLE_GRAPH_Q6_FREEZE=1
export XLLM_DSV4_DYNAMIC_GRAPH_STATE=1
export XLLM_DSV4_GRAPH_DYNAMIC_INDEXER=1
export XLLM_DSV4_GRAPH_BATCHED_INDEXER=1
export XLLM_DSV4_C128_CONTEXT_SPLITS=2
export XLLM_DSV4_C128_CONTEXT_UNROLL=4
export XLLM_DSV4_STATE_COPY_BLOCKS=8
export XLLM_DSV4_INDEXER_SPLIT_SCORE_TOPK=1
export XLLM_DSV4_INDEXER_SCORE_THREADS=128
export XLLM_DSV4_C128_MATE_VERIFY=1
export XLLM_DSV4_MATE_GROUPED_METADATA=1
export XLLM_DSV4_CLEAR_STATE_CACHE_ON_REQUEST=1
export XLLM_DSV4_CLEAR_DECODE_WORKSPACES_ON_REQUEST=1
export XLLM_DSV4_GRAPH_KEY_TRACE=1
export XLLM_DSV4_GRAPH_CAUSAL_SHORT_C4=1

expected_sha=493424fd4b553463be2fe8f3ab8e2169e911e0532c36864b743c5158c388f133
read -r binary_sha _ < <(sha256sum -- "$DSPARK_BIN")
[[ "$binary_sha" == "$expected_sha" ]] || {
  echo "Candidate binary changed; refusing a mixed-version reproduction" >&2; exit 2;
}

# A diagnostic-only switch can silently change both accuracy and performance.
# Reject inherited overrides; deliberate A/B runs use start_dspark_custom.sh.
while IFS= read -r variable; do
  case "$variable" in
    XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY|XLLM_DSV4_ENABLE_GRAPH_BATCHED_MATE_VERIFY|\
    XLLM_DSV4_GRAPH_SHORT_FULLKV_EAGER|XLLM_DSV4_DISABLE_GRAPH_Q6_FREEZE|\
    XLLM_DSV4_DYNAMIC_GRAPH_STATE|XLLM_DSV4_GRAPH_DYNAMIC_INDEXER|\
    XLLM_DSV4_GRAPH_BATCHED_INDEXER|XLLM_DSV4_C128_CONTEXT_SPLITS|\
    XLLM_DSV4_C128_CONTEXT_UNROLL|XLLM_DSV4_STATE_COPY_BLOCKS|\
    XLLM_DSV4_INDEXER_SPLIT_SCORE_TOPK|XLLM_DSV4_INDEXER_SCORE_THREADS|\
    XLLM_DSV4_C128_MATE_VERIFY|XLLM_DSV4_MATE_GROUPED_METADATA|\
    XLLM_DSV4_CLEAR_STATE_CACHE_ON_REQUEST|XLLM_DSV4_CLEAR_DECODE_WORKSPACES_ON_REQUEST|\
    XLLM_DSV4_GRAPH_KEY_TRACE|XLLM_DSV4_GRAPH_CAUSAL_SHORT_C4) ;;
    XLLM_DSV4_*|XLLM_GRAPH_*|XLLM_MUSA_GRAPH_*|XLLM_KEEP_HOST_MIRROR_REFRESH)
      echo "Unset inherited diagnostic control: $variable" >&2; exit 2 ;;
  esac
done < <(compgen -e)

# Existing processes and reservations are never stopped or reset by this script.
exec bash tools/start_dspark_custom.sh "$run_dir" 10240 0.8
