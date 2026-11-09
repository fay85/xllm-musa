#!/usr/bin/env bash
# Settings for the GLM-5.3-Flash VL benchmark on MUSA (default: 2k input / 256 output).
# Override any setting by exporting GLM5_<NAME> or VL_<NAME> before running bench.sh.

KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$KIT_DIR/../.." && pwd)

HOST=${GLM5_HOST:-${VL_HOST:-127.0.0.1}}
PORT=${GLM5_PORT:-${VL_MASTER_HTTP_PORT:-18901}}
MASTER_URL=${GLM5_URL:-${VL_MASTER_URL:-http://$HOST:$PORT}}

MODEL=${GLM5_MODEL_PATH:-${VL_MODEL:-/data/nfs_shared/models/GLM-5.3-Flash}}
MODEL_ID=${GLM5_MODEL_ID:-${VL_MODEL_ID:-glm-5.3-flash}}

DATA_DIR=${GLM5_DATA_DIR:-${VL_DATA_DIR:-/workspace/xllm_vl_bench_ref}}
WORK_DIR=${GLM5_WORK_DIR:-${VL_WORK_DIR:-/tmp/glm5_3_vl_bench}}

ISL=${GLM5_ISL:-${VL_ISL:-2048}}
SHARED=${GLM5_SHARED:-${VL_SHARED:-1024}}
OSL=${GLM5_OSL:-${VL_OSL:-256}}
NUM_IMAGES=${GLM5_NUM_IMAGES:-${VL_NUM_IMAGES:-1}}

PROMPTS=${GLM5_PROMPTS:-${VL_PROMPTS:-$WORK_DIR/vl_prompts_isl${ISL}_img${NUM_IMAGES}.json}}
LOG_DIR=${GLM5_LOG_DIR:-${VL_SERVER_LOG_DIR:-/tmp/glm5_3_flash_run}}
RESULTS_DIR=${GLM5_RESULTS_DIR:-${VL_RESULTS_DIR:-$WORK_DIR/results}}

CONC=${GLM5_CONC:-${VL_CONC:-1}}
REQUESTS=${GLM5_REQUESTS:-${VL_REQUESTS:-}}
WARM=${GLM5_WARM:-${VL_WARM:-1}}
WARM_MASTER=${GLM5_WARM_MASTER:-${VL_WARM_MASTER:-0}}
WARM_CLOSED=${GLM5_WARM_CLOSED:-${VL_WARM_CLOSED:-0}}
