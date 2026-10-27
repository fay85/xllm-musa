# Settings for the Qwen3.8-27B-FP8 VL PD benchmark on MUSA. Every script in
# this directory sources this file. The defaults are the paths inside the
# benchmark container image; edit them here, or override a setting by
# exporting its name with a VL_ prefix, e.g. VL_MODEL=/models/Qwen3.8-27B-FP8.
# Unprefixed names are not read from the environment because container images
# predefine some of them (XLLM_BIN).

KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$KIT_DIR/../.." && pwd)

# One canonical launcher consumes named defaults. Historical settings remain
# available as VL_PROFILE=baseline, while the image defaults to large_kv_rr.
PROFILE=${VL_PROFILE:-large_kv_rr}
case "$PROFILE" in
  large_kv_rr|baseline) source "$KIT_DIR/profiles/$PROFILE.env" ;;
  *) echo "ERROR: unsupported VL_PROFILE=$PROFILE" >&2; return 1 ;;
esac

# Address the instances advertise to each other and to the master.
HOST=${VL_HOST:-$(hostname -I | awk '{print $1}')}

# Qwen3.8-27B-FP8 checkpoint, also used as the MTP draft model.
MODEL=${VL_MODEL:-/data/nfs_shared/models/Qwen3.8-27B-FP8}
MODEL_ID=${VL_MODEL_ID:-Qwen3.8-27B-FP8}

# xllm built from this repository by build.sh; the Mooncake transfer engine
# loads libasio.so from the CMake build tree.
XLLM_BIN=${VL_XLLM_BIN:-$REPO_DIR/build/lib.linux-x86_64-cpython-310/xllm/xllm}
XLLM_LIB_DIR=${VL_XLLM_LIB_DIR:-$REPO_DIR/build/cmake.linux-x86_64-cpython-310/mooncake-common}

# xllm-service master, the librust_tokenizers.so it links against, and etcd.
MASTER_BIN=${VL_MASTER_BIN:-$PROFILE_MASTER_BIN}
RUST_TOKENIZERS_LIB=${VL_RUST_TOKENIZERS_LIB:-/opt/xllm_service/lib/librust_tokenizers.so}
ETCD_BIN=${VL_ETCD_BIN:-/opt/xllm_service/bin/etcd}

# MUSA backend shim preloaded into every xllm process and the prebuilt Mate
# 0.2.7 kernels xllm loads from FLASHINFER_OPS_PATH.
MUSA_PRELOAD=${VL_MUSA_PRELOAD:-/workspace/artifacts/libmusa_backend_init.so}
MUSA_OPS_DIR=${VL_MUSA_OPS_DIR:-/opt/mate027_cache/mp31/cached_ops_jd_k8d}

# Articles and photos (the image ships the reference set), then the generated
# prompts, run logs and results. The prefill instances read the photos from
# DATA_DIR, so keep it on the serving machine.
DATA_DIR=${VL_DATA_DIR:-/workspace/xllm_vl_bench_ref}
WORK_DIR=${VL_WORK_DIR:-$HOME/xllm_vl_bench}
MASTER_TOKENIZER=${VL_MASTER_TOKENIZER:-$WORK_DIR/master_tokenizer}
PROMPTS=${VL_PROMPTS:-$WORK_DIR/vl_prompts_isl40960_n68.json}
RUNS_DIR=${VL_RUNS_DIR:-$WORK_DIR/runs}
RESULTS_DIR=${VL_RESULTS_DIR:-$WORK_DIR/results}

# Three CP2 PREFILL instances and two TP1 DECODE instances.
PREFILL_GPUS=${VL_PREFILL_GPUS:-"0,1 2,3 4,5"}
DECODE_GPUS=${VL_DECODE_GPUS:-"6 7"}

# Per-instance request and batch limit, and MTP draft tokens per step.
CONC=${VL_CONC:-16}
MTP_K=${VL_MTP_K:-2}

# Per-deployment defaults; VL_ overrides remain explicit and are logged.
MAX_MEMORY_UTILIZATION=${VL_MAX_MEMORY_UTILIZATION:-$PROFILE_MEMORY_UTILIZATION}
ENABLE_GRAPH_VMM_POOL=${VL_ENABLE_GRAPH_VMM_POOL:-$PROFILE_GRAPH_VMM_POOL}
PREFILL_BATCH_WAIT_MS=${VL_PREFILL_BATCH_WAIT_MS:-0}
PREFILL_CHUNK_TOKENS=${VL_PREFILL_CHUNK_TOKENS:-8192}
ENABLE_IN_BATCH_PREFIX_CACHE=${VL_ENABLE_IN_BATCH_PREFIX_CACHE:-false}
# This MCCL library reads MCCL_MIN_NCHANNELS even though its diagnostic log
# prints IMCCL_MIN_NCHANNELS. Preserve a direct SDK override as well.
MCCL_MIN_NCHANNELS_PIN=${VL_MCCL_MIN_NCHANNELS:-${MCCL_MIN_NCHANNELS:-}}

# Optional Service diagnostics, recorded with the effective stack settings.
SERVICE_VMODULE=${VL_SERVICE_VMODULE:-}

# A 40k-token prompt costs about 640 KV blocks. The service default of
# 1024 blocks per Router/tenant allows only one request per decode Engine.
# This dedicated benchmark Router may use the full per-resource grant;
# Coordinator capacity observations and the safety margin still bound it.
BATCH_CREDIT_MAX_GRANT_UNITS=${VL_BATCH_CREDIT_MAX_GRANT_UNITS:-$PROFILE_CREDIT_CAP}
BATCH_CREDIT_MAX_GRANT_UNITS_PER_ROUTER_TENANT=${VL_BATCH_CREDIT_MAX_GRANT_UNITS_PER_ROUTER_TENANT:-$PROFILE_CREDIT_CAP}
# Fraction of the Engine-reported KV budget exposed as soft Service credit.
# The Engine already reserves memory via max_memory_utilization.
# Expose its full KV budget here; real Engine admission remains authoritative.
BATCH_CREDIT_SAFETY_PERMYRIAD=${VL_BATCH_CREDIT_SAFETY_PERMYRIAD:-10000}

ETCD_CLIENT_PORT=${VL_ETCD_CLIENT_PORT:-64379}
ETCD_PEER_PORT=${VL_ETCD_PEER_PORT:-64380}
MASTER_HTTP_PORT=${VL_MASTER_HTTP_PORT:-64888}
MASTER_RPC_PORT=${VL_MASTER_RPC_PORT:-64889}
MASTER_URL=${VL_MASTER_URL:-http://$HOST:$MASTER_HTTP_PORT}

# The frozen image carries the exact four-round workload and archived runner.
# Short bench.sh runs remain diagnostics and are not four-round qualification.
FOUR_ROUNDS_DATASET=${VL_FOUR_ROUNDS_DATASET:-/opt/xllm/releases/musa-large-kv-rr-20261025/benchmark/four_rounds_prompts.json}
FOUR_ROUNDS_RUNNER=${VL_FOUR_ROUNDS_RUNNER:-/opt/xllm/releases/musa-large-kv-rr-20261025/benchmark/run_four_rounds.py}
