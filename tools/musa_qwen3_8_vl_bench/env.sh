# Settings for the Qwen3.8-27B-FP8 VL PD benchmark on MUSA. Every script in
# this directory sources this file. The defaults are the paths inside the
# benchmark container image; edit them here, or override a setting by
# exporting its name with a VL_ prefix, e.g. VL_MODEL=/models/Qwen3.8-27B-FP8.
# Unprefixed names are not read from the environment because container images
# predefine some of them (XLLM_BIN).

KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$KIT_DIR/../.." && pwd)

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
MASTER_BIN=${VL_MASTER_BIN:-/opt/xllm_service/bin/xllm_master_serving}
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

ETCD_CLIENT_PORT=${VL_ETCD_CLIENT_PORT:-64379}
ETCD_PEER_PORT=${VL_ETCD_PEER_PORT:-64380}
MASTER_HTTP_PORT=${VL_MASTER_HTTP_PORT:-64888}
MASTER_RPC_PORT=${VL_MASTER_RPC_PORT:-64889}
MASTER_URL=${VL_MASTER_URL:-http://$HOST:$MASTER_HTTP_PORT}
