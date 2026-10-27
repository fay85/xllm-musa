# Settings for the Qwen3.8-27B-FP8 VL PD benchmark on MUSA.
# Source of truth for retunable knobs: /workspace/config.json
#   - binary_default_on: already ON in the wave2 binary — do not re-export
#   - explicit: values this stack still applies (edit config, then restart)
# One-off overrides: export VL_<NAME> before sourcing (e.g. VL_MTP_K=1).

KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$KIT_DIR/../.." && pwd)

PROFILE=${VL_PROFILE:-large_kv_rr}
case "$PROFILE" in
  large_kv_rr|baseline) source "$KIT_DIR/profiles/$PROFILE.env" ;;
  *) echo "ERROR: unsupported VL_PROFILE=$PROFILE" >&2; return 1 ;;
esac

HOST=${VL_HOST:-$(hostname -I | awk '{print $1}')}

MODEL=${VL_MODEL:-/data/nfs_shared/models/Qwen3.8-27B-FP8}
MODEL_ID=${VL_MODEL_ID:-Qwen3.8-27B-FP8}

XLLM_BIN=${VL_XLLM_BIN:-$REPO_DIR/build/lib.linux-x86_64-cpython-310/xllm/xllm}
XLLM_LIB_DIR=${VL_XLLM_LIB_DIR:-$REPO_DIR/build/cmake.linux-x86_64-cpython-310/mooncake-common}

MASTER_BIN=${VL_MASTER_BIN:-$PROFILE_MASTER_BIN}
RUST_TOKENIZERS_LIB=${VL_RUST_TOKENIZERS_LIB:-/opt/xllm_service/lib/librust_tokenizers.so}
ETCD_BIN=${VL_ETCD_BIN:-/opt/xllm_service/bin/etcd}

MUSA_PRELOAD=${VL_MUSA_PRELOAD:-/workspace/artifacts/libmusa_backend_init.so}
MUSA_OPS_DIR=${VL_MUSA_OPS_DIR:-/opt/mate027_cache/mp31/cached_ops_jd_k8d}

DATA_DIR=${VL_DATA_DIR:-/workspace/xllm_vl_bench_ref}
WORK_DIR=${VL_WORK_DIR:-$HOME/xllm_vl_bench}
MASTER_TOKENIZER=${VL_MASTER_TOKENIZER:-$WORK_DIR/master_tokenizer}
PROMPTS=${VL_PROMPTS:-$WORK_DIR/vl_prompts_isl40960_n68.json}
RUNS_DIR=${VL_RUNS_DIR:-$WORK_DIR/runs}
RESULTS_DIR=${VL_RESULTS_DIR:-$WORK_DIR/results}

PREFILL_GPUS=${VL_PREFILL_GPUS:-"0,1 2,3 4,5"}
DECODE_GPUS=${VL_DECODE_GPUS:-"6 7"}

CONC=${VL_CONC:-16}
MTP_K=${VL_MTP_K:-3}
ENABLE_MTP=${VL_ENABLE_MTP:-true}

MAX_MEMORY_UTILIZATION=${VL_MAX_MEMORY_UTILIZATION:-$PROFILE_MEMORY_UTILIZATION}
ENABLE_GRAPH_VMM_POOL=${VL_ENABLE_GRAPH_VMM_POOL:-$PROFILE_GRAPH_VMM_POOL}
PREFILL_BATCH_WAIT_MS=${VL_PREFILL_BATCH_WAIT_MS:-0}
PREFILL_CHUNK_TOKENS=${VL_PREFILL_CHUNK_TOKENS:-8192}
ENABLE_IN_BATCH_PREFIX_CACHE=${VL_ENABLE_IN_BATCH_PREFIX_CACHE:-false}
MCCL_MIN_NCHANNELS_PIN=${VL_MCCL_MIN_NCHANNELS:-${MCCL_MIN_NCHANNELS:-}}

SERVICE_VMODULE=${VL_SERVICE_VMODULE:-}

BATCH_CREDIT_MAX_GRANT_UNITS=${VL_BATCH_CREDIT_MAX_GRANT_UNITS:-$PROFILE_CREDIT_CAP}
BATCH_CREDIT_MAX_GRANT_UNITS_PER_ROUTER_TENANT=${VL_BATCH_CREDIT_MAX_GRANT_UNITS_PER_ROUTER_TENANT:-$PROFILE_CREDIT_CAP}
BATCH_CREDIT_SAFETY_PERMYRIAD=${VL_BATCH_CREDIT_SAFETY_PERMYRIAD:-10000}

ETCD_CLIENT_PORT=${VL_ETCD_CLIENT_PORT:-64379}
ETCD_PEER_PORT=${VL_ETCD_PEER_PORT:-64380}
MASTER_HTTP_PORT=${VL_MASTER_HTTP_PORT:-64888}
MASTER_RPC_PORT=${VL_MASTER_RPC_PORT:-64889}
MASTER_URL=${VL_MASTER_URL:-http://$HOST:$MASTER_HTTP_PORT}

FOUR_ROUNDS_DATASET=${VL_FOUR_ROUNDS_DATASET:-/opt/xllm/releases/musa-large-kv-rr-20261025/benchmark/four_rounds_prompts.json}
FOUR_ROUNDS_RUNNER=${VL_FOUR_ROUNDS_RUNNER:-/opt/xllm/releases/musa-large-kv-rr-20261025/benchmark/run_four_rounds.py}

# Apply /workspace/config.json (wins over profile defaults; VL_* still wins if
# re-exported after this source). Skip with VL_SKIP_CONFIG=1.
if [[ ${VL_SKIP_CONFIG:-0} != 1 ]]; then
  # Preserve intentional VL_* one-offs set before source.
  _vl_mtp=${VL_MTP_K-}
  _vl_conc=${VL_CONC-}
  _vl_mem=${VL_MAX_MEMORY_UTILIZATION-}
  _vl_vmm=${VL_ENABLE_GRAPH_VMM_POOL-}
  _vl_model=${VL_MODEL-}
  _vl_bin=${VL_XLLM_BIN-}
  _vl_pg=${VL_PREFILL_GPUS-}
  _vl_dg=${VL_DECODE_GPUS-}
  _vl_enable_mtp=${VL_ENABLE_MTP-}
  # shellcheck disable=SC1091
  source "$KIT_DIR/load_config.sh"
  [[ -n $_vl_mtp ]] && MTP_K=$_vl_mtp
  [[ -n $_vl_conc ]] && CONC=$_vl_conc
  [[ -n $_vl_mem ]] && MAX_MEMORY_UTILIZATION=$_vl_mem
  [[ -n $_vl_vmm ]] && ENABLE_GRAPH_VMM_POOL=$_vl_vmm
  [[ -n $_vl_model ]] && MODEL=$_vl_model
  [[ -n $_vl_bin ]] && XLLM_BIN=$_vl_bin
  [[ -n $_vl_pg ]] && PREFILL_GPUS=$_vl_pg
  [[ -n $_vl_dg ]] && DECODE_GPUS=$_vl_dg
  [[ -n $_vl_enable_mtp ]] && ENABLE_MTP=$_vl_enable_mtp
  unset _vl_mtp _vl_conc _vl_mem _vl_vmm _vl_model _vl_bin _vl_pg _vl_dg _vl_enable_mtp
fi

# Defaults for explicit knobs if config omitted them.
ENABLE_DISAGG_PD=${ENABLE_DISAGG_PD:-true}
ENABLE_NATIVE_SERVICE_ADAPTER=${ENABLE_NATIVE_SERVICE_ADAPTER:-true}
KV_CACHE_TRANSFER_MODE=${KV_CACHE_TRANSFER_MODE:-PULL}
KV_SPLIT_SIZE=${KV_SPLIT_SIZE:-1}
ENABLE_PREFILL_PIECEWISE_GRAPH_PREFILL=${ENABLE_PREFILL_PIECEWISE_GRAPH_PREFILL:-true}
ENABLE_PREFILL_PIECEWISE_GRAPH_DECODE=${ENABLE_PREFILL_PIECEWISE_GRAPH_DECODE:-false}
MAX_TOKENS_PER_BATCH_PREFILL=${MAX_TOKENS_PER_BATCH_PREFILL:-8192}
MAX_TOKENS_PER_BATCH_DECODE=${MAX_TOKENS_PER_BATCH_DECODE:-32768}
MAX_TOKENS_FOR_GRAPH_MODE_PREFILL=${MAX_TOKENS_FOR_GRAPH_MODE_PREFILL:-8192}
MAX_TOKENS_FOR_GRAPH_MODE_DECODE=${MAX_TOKENS_FOR_GRAPH_MODE_DECODE:-4096}
MAX_TOKENS_PER_CHUNK_FOR_PREFILL_DECODE=${MAX_TOKENS_PER_CHUNK_FOR_PREFILL_DECODE:-2048}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
SPECULATIVE_ALGORITHM=${SPECULATIVE_ALGORITHM:-MTP}
ENABLE_MTP=${ENABLE_MTP:-true}

MASTER_URL=${VL_MASTER_URL:-http://$HOST:$MASTER_HTTP_PORT}
