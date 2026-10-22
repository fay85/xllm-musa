#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
set -euo pipefail
role=${1:?usage: start_role.sh PREFILL|DECODE NEW_RUN_DIR}
run=${2:?pass an absolute, previously nonexistent run directory}
[[ "$run" == /* && ! -e "$run" && ! -e "$run.launcher.log" ]]
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
artifacts=/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005
mkdir -p -- "$(dirname -- "$run")"
case "$role" in
  PREFILL)
    [[ "$(hostname -s)" == worker38094 ]]
    export DSPARK_BIN="$artifacts/candidates/indexer_mask_once/xllm"
    export DSPARK_BIN_SHA256=caf3db8f775224045f9b8143f1b8a3491fb4637229fd5ab18194e00b60506b43
    export MCK_DATA_PORTS=16397,15901,15773,16486,16158,15036,15583,15672
    export DIAG_COMPACT_PREFILL_FP8=1 DIAG_PREFILL_INDEXER_BF16_FP32=1 DIAG_PREFILL_PREPARE_ONCE=1
    export DIAG_HC_TILED_PREFILL=1 DIAG_PREFILL_GATE_BF16_FP32=1 DIAG_LONG_MOE_VECTOR_RESTORE=1
    export DIAG_PREFILL_REDUCE_SCATTER=1 DIAG_LONG_MOE_WARP_SWIGLU=1
    export DIAG_INDEXER_ROPE_COPY=1 DIAG_MAIN_ROPE_COPY=1 DIAG_INDEXER_MASK_ONCE=1
    export MAX_TOKENS_PER_BATCH=16384 PREFILL_CHUNK=16384
    launcher="$here/frozen/pd-role-indexer-mask-once.sh"
    ;;
  DECODE)
    [[ "$(hostname -s)" == worker38090 ]]
    export DSPARK_BIN="$artifacts/candidates/reopen_decode/xllm"
    export DSPARK_BIN_SHA256=5ea86017debbd94ba751a8ce38c74c1e2e9d39487b4ef3fcf662a24cedbaac93
    export MCK_DATA_PORTS=32300,32301,32302,32303,32304,32305,32306,32307
    export MAX_TOKENS_PER_BATCH=10032 PREFILL_CHUNK=8192
    launcher="$here/frozen/pd-role-reopen-decode.sh"
    ;;
  *) echo 'role must be PREFILL or DECODE' >&2; exit 2 ;;
esac
export GRAPH=true PREFIX_CACHE=true OVERLAP=true
export MAX_SEQS_PER_BATCH=16 MAX_CONCURRENT_REQUESTS=16 REQUEST_THREADS=4
set -C
exec bash "$launcher" "$role" "$run" > "$run.launcher.log" 2>&1
