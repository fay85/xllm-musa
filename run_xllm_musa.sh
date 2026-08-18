#!/usr/bin/env bash
# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start the MUSA-only xLLM build with the validated graph/FA3 defaults.
# Qwen3 graph capture additionally requires the explicit experimental opt-in.

set -euo pipefail

ulimit -c 0 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${MODEL_ROOT:-/workspace/model_weights}"
if [[ -n "${MODEL_NAME:-}" ]]; then
  MODEL_NAME_EXPLICIT=1
else
  MODEL_NAME_EXPLICIT=0
fi
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B-FP8}"
MODEL_PATH="${MODEL_PATH:-}"
PORT="${PORT:-8092}"
MASTER_NODE_ADDR="${MASTER_NODE_ADDR:-127.0.0.1:9748}"
XLLM_RANDOM_SEED="${XLLM_RANDOM_SEED:--1}"
MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/log}"
BACKGROUND=0

# Validated MUSA defaults. Packed prefill stays disabled because FP8 B2 is not
# yet a correctness-safe default. Set any value explicitly to run an A/B arm.
ENABLE_GRAPH="${ENABLE_GRAPH:-1}"
ENABLE_GRAPH_DECODE_NO_PADDING="${ENABLE_GRAPH_DECODE_NO_PADDING:-1}"
ENABLE_GRAPH_VMM_POOL="${ENABLE_GRAPH_VMM_POOL:-0}"
ENABLE_PREFILL_PIECEWISE_GRAPH="${ENABLE_PREFILL_PIECEWISE_GRAPH:-1}"
ENABLE_PACKED_PREFILL="${ENABLE_PACKED_PREFILL:-0}"
MAX_TOKENS_FOR_GRAPH_MODE="${MAX_TOKENS_FOR_GRAPH_MODE:-8192}"

XLLM_USE_FA3_DECODE_EXPLICIT="${XLLM_USE_FA3_DECODE+x}"
XLLM_MUSA_POOL_COMPUTE_STREAM_EXPLICIT="${XLLM_MUSA_POOL_COMPUTE_STREAM+x}"
export XLLM_USE_FA3="${XLLM_USE_FA3:-1}"
export XLLM_USE_FA3_DECODE="${XLLM_USE_FA3_DECODE:-${XLLM_USE_FA3}}"
export XLLM_MUSA_POOL_COMPUTE_STREAM="${XLLM_MUSA_POOL_COMPUTE_STREAM:-1}"
export XLLM_PIECEWISE_CAPTURE_FA3="${XLLM_PIECEWISE_CAPTURE_FA3:-1}"
export XLLM_GDN_DECODE_BACKEND="mate"
export XLLM_PACKED_PREFILL_PIECEWISE="${XLLM_PACKED_PREFILL_PIECEWISE:-0}"
if [[ "$ENABLE_PACKED_PREFILL" == 1 ]]; then
  DEFAULT_MAX_PACKED_PREFILL_SEQS=2
else
  DEFAULT_MAX_PACKED_PREFILL_SEQS=1
fi
export XLLM_MAX_PACKED_PREFILL_SEQS="${XLLM_MAX_PACKED_PREFILL_SEQS:-${DEFAULT_MAX_PACKED_PREFILL_SEQS}}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --background              Run in the background under LOG_DIR
  --model-path PATH         Hugging Face model directory
  --model-name NAME         Model id and folder name under MODEL_ROOT
  --port PORT               HTTP port (default: 8092)
  --musa-visible-devices N  Physical device(s) exposed to the process
  -h, --help                Show this help

Important overrides:
  XLLM_BIN, ENABLE_GRAPH, ENABLE_GRAPH_VMM_POOL,
  ENABLE_PREFILL_PIECEWISE_GRAPH, ENABLE_PACKED_PREFILL,
  XLLM_USE_FA3_DECODE, XLLM_MUSA_POOL_COMPUTE_STREAM,
  XLLM_RANDOM_SEED,
  XLLM_QWEN3_ENABLE_GRAPH_EXPERIMENTAL,
  XLLM_MAX_PACKED_PREFILL_SEQS, MAX_CONCURRENT_REQUESTS,
  MAX_SEQS_PER_BATCH, MAX_TOKENS_PER_BATCH
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "${option} requires a non-empty value." >&2
    usage >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background)
      BACKGROUND=1
      shift
      ;;
    --model-path)
      require_value "$1" "${2:-}"
      MODEL_PATH="$2"
      shift 2
      ;;
    --model-name)
      require_value "$1" "${2:-}"
      MODEL_NAME="$2"
      MODEL_NAME_EXPLICIT=1
      shift 2
      ;;
    --port)
      require_value "$1" "${2:-}"
      PORT="$2"
      shift 2
      ;;
    --musa-visible-devices)
      require_value "$1" "${2:-}"
      MUSA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$MODEL_PATH" && "$MODEL_NAME_EXPLICIT" == 0 ]]; then
  MODEL_NAME="$(basename -- "${MODEL_PATH%/}")"
fi

resolve_model_path() {
  if [[ -n "$MODEL_PATH" ]]; then
    printf '%s\n' "$MODEL_PATH"
    return
  fi

  local candidate="${MODEL_ROOT}/${MODEL_NAME}"
  if [[ -f "${candidate}/config.json" ]]; then
    printf '%s\n' "$candidate"
    return
  fi

  echo "Model not found: ${candidate}. Set MODEL_PATH or --model-path." >&2
  exit 1
}

resolve_xllm_bin() {
  if [[ -n "${XLLM_BIN:-}" ]]; then
    if [[ ! -x "$XLLM_BIN" ]]; then
      echo "XLLM_BIN is not executable: ${XLLM_BIN}" >&2
      exit 1
    fi
    printf '%s\n' "$XLLM_BIN"
    return
  fi

  local candidate="${SCRIPT_DIR}/build/lib.linux-x86_64-cpython-310/xllm/xllm"
  if [[ -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return
  fi

  candidate="$(find "${SCRIPT_DIR}/build" -path '*/xllm/xllm' -type f \
    -perm -u+x -print -quit 2>/dev/null || true)"
  if [[ -n "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return
  fi

  echo "No xLLM binary found below ${SCRIPT_DIR}/build." >&2
  exit 1
}

resolve_backend_init() {
  if [[ -n "${MUSA_BACKEND_INIT_SO:-}" ]]; then
    [[ -f "$MUSA_BACKEND_INIT_SO" ]] || {
      echo "MUSA_BACKEND_INIT_SO does not exist: ${MUSA_BACKEND_INIT_SO}" >&2
      exit 1
    }
    printf '%s\n' "$MUSA_BACKEND_INIT_SO"
    return
  fi

  local candidate
  for candidate in \
    /workspace/artifacts/libmusa_backend_init.so \
    "${SCRIPT_DIR}/artifacts/libmusa_backend_init.so"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "libmusa_backend_init.so was not found; set MUSA_BACKEND_INIT_SO." >&2
  return 1
}

setup_runtime_env() {
  export MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
  export CUDA_HOME="${CUDA_HOME:-${MUSA_HOME}}"
  export MUSAMAPPING_PATH="${MUSAMAPPING_PATH:-${MUSA_HOME}/tools/musamapping}"
  export MUDNN_HOME="${MUDNN_HOME:-/workspace/mudnn_3.4.0}"
  export MCCL_HOME="${MCCL_HOME:-/workspace/MCCL_2.3.0/mccl}"
  export MUSA_LAUNCH_BLOCKING="${MUSA_LAUNCH_BLOCKING:-0}"
  export XLLM_TILELANG_LIB="${XLLM_TILELANG_LIB:-/usr/local/lib/python3.10/dist-packages/tilelang/lib/libtilelang.so}"
  local mate_home="${MATE_HOME:-}"
  local candidate
  if [[ -z "${mate_home}" ]]; then
    for candidate in /workspace/mate_0.2.5 /data/feihu/mate_0.2.5; do
      if [[ -f "${candidate}/version.txt" ]]; then
        mate_home="${candidate}"
        break
      fi
    done
  fi
  [[ -n "${mate_home}" && -f "${mate_home}/version.txt" ]] || {
    echo "Mate 0.2.5 source tree not found; set MATE_HOME." >&2
    exit 1
  }
  local mate_version
  mate_version="$(tr -d '[:space:]' < "${mate_home}/version.txt")"
  [[ "${mate_version}" == "0.2.5" ]] || {
    echo "Expected Mate 0.2.5, found ${mate_version} at ${mate_home}" >&2
    exit 1
  }
  export MATE_HOME="${mate_home}"
  export MATE_MUSA_ARCH_LIST="${MATE_MUSA_ARCH_LIST:-3.1}"
  export PYTHONPATH="${MATE_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ -z "${MATE_WORKSPACE_BASE:-}" ]]; then
    if [[ -d /data/feihu/mate025_cache ]]; then
      MATE_WORKSPACE_BASE=/data/feihu/mate025_cache
    else
      MATE_WORKSPACE_BASE="${MATE_HOME}"
    fi
  fi
  export MATE_WORKSPACE_BASE

  export MATE_MUBIN_DIR="${MATE_MUBIN_DIR:-${MATE_WORKSPACE_BASE}/mubin}"
  export FLASHINFER_OPS_PATH="${MATE_WORKSPACE_BASE}/.cache/mate/0.2.5/mp31/cached_ops"

  local -a library_dirs=(
    "${MUDNN_HOME}/lib"
    "/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib"
    "/usr/local/lib/python3.10/dist-packages/torch_musa/lib"
    "/usr/local/lib/python3.10/dist-packages/torch/lib"
    "${MUSA_HOME}/lib"
    "${MCCL_HOME}/lib"
    "/usr/local/openmpi/lib"
    "/opt/intel/oneapi/mkl/lib/intel64"
  )
  local dir runtime_library_path=""
  for dir in "${library_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
      runtime_library_path="${runtime_library_path:+${runtime_library_path}:}${dir}"
    fi
  done
  export LD_LIBRARY_PATH="${runtime_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

  local -a preload=()
  local musa_python="/usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so"
  local backend_init
  backend_init="$(resolve_backend_init)"
  [[ -f "$musa_python" ]] && preload+=("$musa_python")
  [[ -n "$backend_init" ]] && preload+=("$backend_init")
  [[ -f /opt/intel/oneapi/mkl/lib/intel64/libmkl_core.so.2 ]] && \
    preload+=(/opt/intel/oneapi/mkl/lib/intel64/libmkl_core.so.2)
  [[ -n "${LD_PRELOAD:-}" ]] && preload+=("$LD_PRELOAD")
  if [[ ${#preload[@]} -gt 0 ]]; then
    local IFS=:
    export LD_PRELOAD="${preload[*]}"
  fi

  export MUSA_VISIBLE_DEVICES
}

set_model_defaults() {
  local default_max_memory_utilization="0.70"
  BLOCK_SIZE="${BLOCK_SIZE:-64}"
  MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-4}"
  MAX_SEQS_PER_BATCH="${MAX_SEQS_PER_BATCH:-${MAX_CONCURRENT_REQUESTS}}"
  case "${MODEL_NAME,,}" in
    qwen3.5-35b-a3b* | qwen3.6-35b-a3b*)
      default_max_memory_utilization="0.90"
      ;;
  esac
  MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-${default_max_memory_utilization}}"
}

detect_attention_shape() {
  local model_path="$1"
  python3 - "${model_path}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = json.load(config_file)
text_config = config.get("text_config", config)
hidden_size = int(text_config.get("hidden_size", 0) or 0)
num_heads = int(text_config.get("num_attention_heads", 0) or 0)
num_kv_heads = int(text_config.get("num_key_value_heads", 0) or 0)
head_dim = int(text_config.get("head_dim", 0) or 0)
if head_dim == 0 and num_heads > 0:
    head_dim = hidden_size // num_heads
gqa_ratio = 0
if num_kv_heads > 0 and num_heads % num_kv_heads == 0:
    gqa_ratio = num_heads // num_kv_heads
dtype = (
    text_config.get("torch_dtype")
    or text_config.get("dtype")
    or config.get("torch_dtype")
    or config.get("dtype")
    or "unknown"
)
model_type = str(
    text_config.get("model_type") or config.get("model_type") or "unknown"
)
quantization_config = (
    text_config.get("quantization_config")
    or config.get("quantization_config")
    or {}
)
quant_method = str(quantization_config.get("quant_method") or "none").lower()
print(head_dim, gqa_ratio, str(dtype).lower(), model_type, quant_method)
PY
}

preflight() {
  local model_path="$1"
  local xllm_bin="$2"
  [[ -f "${model_path}/config.json" ]] || {
    echo "Missing ${model_path}/config.json" >&2
    exit 1
  }
  [[ -x "$xllm_bin" ]] || {
    echo "xLLM binary is not executable: ${xllm_bin}" >&2
    exit 1
  }
  [[ -d "$FLASHINFER_OPS_PATH" ]] || {
    echo "Mate 0.2.5 ops directory does not exist: ${FLASHINFER_OPS_PATH}" >&2
    exit 1
  }
  if ! find "$FLASHINFER_OPS_PATH" -name '*.so' -print -quit | grep -q .; then
    echo "Mate 0.2.5 ops directory contains no shared objects: ${FLASHINFER_OPS_PATH}" >&2
    exit 1
  fi

  local head_dim gqa_ratio dtype model_type quant_method
  read -r head_dim gqa_ratio dtype model_type quant_method < <(
    detect_attention_shape "$model_path"
  )
  case "$model_type" in
    qwen3 | qwen3_5*)
      if [[ "$BLOCK_SIZE" != 64 ]]; then
        echo "==> MUSA Qwen3 attention uses effective BLOCK_SIZE=64; " \
          "normalizing requested BLOCK_SIZE=${BLOCK_SIZE}."
        BLOCK_SIZE=64
      fi
      ;;
  esac

  if [[ "$model_type" == qwen3_5_moe_text ]]; then
    local gemm_artifact_id=fbd1a6df72047350033cb4c524dd8145b8f18698
    local gemm_mubin_root="$MATE_MUBIN_DIR/gemm/$gemm_artifact_id"
    [[ -f "$gemm_mubin_root/kernel_map.json" ]] || {
      echo "Missing Mate 0.2.5 GEMM kernel map: $gemm_mubin_root/kernel_map.json" >&2
      exit 1
    }
    local gemm_kernel_prefix
    case "$quant_method" in
      fp8)
        gemm_kernel_prefix=e4m3e4m3bf16bf16ssgemm
        ;;
      none)
        gemm_kernel_prefix=bf16bf16bf16bf16ssgemm
        ;;
      *)
        echo "Unsupported Mate 0.2.5 MoE quantization: $quant_method" >&2
        exit 1
        ;;
    esac
    local -a moe_gemm_launchers
    mapfile -t moe_gemm_launchers < <(
      find "$FLASHINFER_OPS_PATH" -mindepth 2 -maxdepth 2 -type f \
        -name "run_${gemm_kernel_prefix}_gm[134]_*.so" | sort
    )
    [[ ${#moe_gemm_launchers[@]} -eq 10 ]] || {
      echo "Mate 0.2.5 MoE requires 10 ${gemm_kernel_prefix} launchers; " \
        "found ${#moe_gemm_launchers[@]}." >&2
      exit 1
    }
    local kernel launcher_path launcher_uri
    for launcher_path in "${moe_gemm_launchers[@]}"; do
      launcher_uri="${launcher_path%/*}"
      launcher_uri="${launcher_uri##*/}"
      kernel="${launcher_uri#run_}"
      [[ -f "$gemm_mubin_root/mubin/$kernel.o" ]] || {
        echo "Missing Mate 0.2.5 MoE GEMM object: $kernel.o" >&2
        exit 1
      }
    done
  fi

  local fa3_shape_supported=1
  if [[ "$dtype" != bfloat16 && "$dtype" != bf16 &&
        "$dtype" != torch.bfloat16 ]]; then
    fa3_shape_supported=0
  elif [[ "$head_dim" == 128 &&
          ("$gqa_ratio" == 2 || "$gqa_ratio" == 4) ]]; then
    :
  elif [[ "$head_dim" == 256 &&
          ("$gqa_ratio" == 6 || "$gqa_ratio" == 8) ]]; then
    :
  else
    fa3_shape_supported=0
  fi

  if [[ "$fa3_shape_supported" == 0 ]]; then
    echo "Mate 0.2.5 FA3 artifacts are unavailable for dtype=${dtype}, " \
      "head_dim=${head_dim}, GQA=${gqa_ratio}." >&2
    exit 1
  fi

  if [[ "$XLLM_USE_FA3" != 1 ]]; then
    echo "This Mate 0.2.5 deployment requires XLLM_USE_FA3=1." >&2
    exit 1
  fi

  if [[ "$XLLM_USE_FA3_DECODE" == 1 && "$BLOCK_SIZE" != 64 ]]; then
    if [[ -n "$XLLM_USE_FA3_DECODE_EXPLICIT" ]]; then
      echo "FA3 decode requires BLOCK_SIZE=64; got ${BLOCK_SIZE}." >&2
      exit 1
    fi
    export XLLM_USE_FA3_DECODE=0
    echo "==> FA3 decode disabled for BLOCK_SIZE=${BLOCK_SIZE}; " \
      "prefill remains on FA3 and decode uses FA2."
  fi

  if [[ "$model_type" == qwen3 &&
        -z "$XLLM_MUSA_POOL_COMPUTE_STREAM_EXPLICIT" ]]; then
    export XLLM_MUSA_POOL_COMPUTE_STREAM=0
    echo "==> Qwen3 FA2/FA3 stream policy: using the default compute stream."
  fi

  if [[ "$XLLM_USE_FA3" == 1 || "$XLLM_USE_FA3_DECODE" == 1 ]]; then
    [[ -d "$FLASHINFER_OPS_PATH" ]] || {
      echo "Mate 0.2.5 FA3 ops directory does not exist: ${FLASHINFER_OPS_PATH}" >&2
      exit 1
    }
    local prefill_hash decode_hash metadata_ratio
    if [[ "$head_dim" == 128 && "$gqa_ratio" == 2 ]]; then
      prefill_hash="782ec5133834c8e9cec31f987596832dccb087fca95a59dfa17b7f4125ac1108"
      decode_hash="00c10b1f3506da04ed725b335bb7cf0bb3f3d52f636ebbe42666a64d56c62054"
      metadata_ratio=2
    elif [[ "$head_dim" == 128 && "$gqa_ratio" == 4 ]]; then
      prefill_hash="b6883be7b4cfcc0200f994cdd9ec9844d8ec6175c27a5ea5488723c9a43164bf"
      decode_hash="752d744e3cb4c86cb68d87b6765dad16ac8be82118ea254af186d3219d1cba13"
      metadata_ratio=4
    elif [[ "$head_dim" == 256 && "$gqa_ratio" == 6 ]]; then
      prefill_hash="e140040cf5f04fdf94b0c9d28900659052de34f380b6ecea23afd16d3691d4ea"
      decode_hash="fde7210c2a9ea763861b4f8358dc6a87e6a8eea0fb36bd5bb0477a9f356e69b8"
      metadata_ratio=6
    else
      prefill_hash="236afb906e2625d51106085cc57d954f7e5b1440b41da68c782fc894801386ee"
      decode_hash="fa49af2cdea283ee74e581849904e618b9eb550b9a6ab8ab04aa0693292b08f0"
      metadata_ratio=8
    fi
    local metadata_warp metadata_uri
    for metadata_warp in $(seq 1 17); do
      metadata_uri="fmha_get_metadata_${metadata_ratio}x${metadata_warp}_bf16_ragged_q_padded_k_causal_packgqa"
      [[ -f "$FLASHINFER_OPS_PATH/$metadata_uri/$metadata_uri.so" ]] || {
        echo "Missing Mate 0.2.5 FA3 metadata op: $FLASHINFER_OPS_PATH/$metadata_uri" >&2
        exit 1
      }
    done
    if [[ "$XLLM_USE_FA3" == 1 ]]; then
      local prefill_uri="fmha_fwd_${prefill_hash}"
      [[ -f "$FLASHINFER_OPS_PATH/$prefill_uri/$prefill_uri.so" ]] || {
        echo "Missing Mate 0.2.5 FA3 prefill op for GQA=$gqa_ratio: $prefill_uri" >&2
        exit 1
      }
    fi
    if [[ "$XLLM_USE_FA3_DECODE" == 1 ]]; then
      local decode_uri="fmha_fwd_${decode_hash}"
      [[ -f "$FLASHINFER_OPS_PATH/$decode_uri/$decode_uri.so" ]] || {
        echo "Missing Mate 0.2.5 FA3 decode op for GQA=$gqa_ratio: $decode_uri" >&2
        exit 1
      }
    fi
    local combine_size combine_uri
    for combine_size in 16 32 64; do
      combine_uri="fmha_fwd_combine_bf16_16x64x${combine_size}_ragged_q_metadata"
      [[ -f "$FLASHINFER_OPS_PATH/$combine_uri/$combine_uri.so" ]] || {
        echo "Missing Mate 0.2.5 FA3 combine op: $combine_uri" >&2
        exit 1
      }
    done
  fi
}

run_xllm() {
  local model_path="$1"
  local xllm_bin="$2"
  local -a cmd=(
    "$xllm_bin"
    "--model=${model_path}"
    "--model_id=${MODEL_NAME}"
    "--backend=llm"
    "--port=${PORT}"
    "--master_node_addr=${MASTER_NODE_ADDR}"
    "--random_seed=${XLLM_RANDOM_SEED}"
    "--nnodes=1"
    "--node_rank=0"
    "--block_size=${BLOCK_SIZE}"
    "--max_memory_utilization=${MAX_MEMORY_UTILIZATION}"
    "--max_concurrent_requests=${MAX_CONCURRENT_REQUESTS}"
    "--max-seqs-per-batch=${MAX_SEQS_PER_BATCH}"
    "--enable_prefix_cache=${ENABLE_PREFIX_CACHE:-false}"
    "--enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-true}"
    "--max_tokens_per_chunk_for_prefill=${MAX_TOKENS_PER_CHUNK_FOR_PREFILL:-8192}"
    "--enable_schedule_overlap=${ENABLE_SCHEDULE_OVERLAP:-true}"
    "--enable_graph=$([[ "$ENABLE_GRAPH" == 1 ]] && echo true || echo false)"
    "--enable_graph_mode_decode_no_padding=$([[ "$ENABLE_GRAPH_DECODE_NO_PADDING" == 1 ]] && echo true || echo false)"
    "--enable_graph_vmm_pool=$([[ "$ENABLE_GRAPH_VMM_POOL" == 1 ]] && echo true || echo false)"
    "--enable_prefill_piecewise_graph=$([[ "$ENABLE_PREFILL_PIECEWISE_GRAPH" == 1 ]] && echo true || echo false)"
    "--enable_packed_prefill=$([[ "$ENABLE_PACKED_PREFILL" == 1 ]] && echo true || echo false)"
    "--max_tokens_for_graph_mode=${MAX_TOKENS_FOR_GRAPH_MODE}"
    "--num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-0}"
  )

  [[ -n "${MAX_TOKENS_PER_BATCH:-}" ]] && \
    cmd+=("--max_tokens_per_batch=${MAX_TOKENS_PER_BATCH}")
  if [[ "${ENABLE_PROFILE:-0}" == 1 ]]; then
    cmd+=("--enable_online_profile=true")
    [[ -n "${PROFILE_DIR:-}" ]] && cmd+=("--profile_dir=${PROFILE_DIR}")
  fi
  if [[ "${NUM_SPECULATIVE_TOKENS:-0}" -gt 0 ]]; then
    local speculative_algorithm="${SPECULATIVE_ALGORITHM:-MTP}"
    cmd+=("--speculative_algorithm=${speculative_algorithm}")
    if [[ "${speculative_algorithm,,}" != suffix ]]; then
      if [[ -z "${DRAFT_MODEL_PATH:-}" ||
            ! -f "${DRAFT_MODEL_PATH}/config.json" ]]; then
        echo "${speculative_algorithm} requires DRAFT_MODEL_PATH with config.json." >&2
        exit 1
      fi
      cmd+=("--draft_model=${DRAFT_MODEL_PATH}")
    fi
  fi

  echo "==> xLLM binary: ${xllm_bin}"
  echo "==> model: ${model_path} (${MODEL_NAME})"
  echo "==> random seed: ${XLLM_RANDOM_SEED}"
  echo "==> device: logical musa:0, visible=${MUSA_VISIBLE_DEVICES}"
  echo "==> graph=${ENABLE_GRAPH}, piecewise=${ENABLE_PREFILL_PIECEWISE_GRAPH}, vmm=${ENABLE_GRAPH_VMM_POOL}, packed=${ENABLE_PACKED_PREFILL}"
  echo "==> FA3=${XLLM_USE_FA3}, FA3 decode=${XLLM_USE_FA3_DECODE}, pool stream=${XLLM_MUSA_POOL_COMPUTE_STREAM}, GDN=${XLLM_GDN_DECODE_BACKEND}"
  echo "==> max concurrency=${MAX_CONCURRENT_REQUESTS}, max seqs=${MAX_SEQS_PER_BATCH}, memory utilization=${MAX_MEMORY_UTILIZATION}"

  if [[ "$BACKGROUND" == 1 ]]; then
    mkdir -p "$LOG_DIR"
    local log_file="${LOG_DIR}/xllm_${MODEL_NAME//\//_}.log"
    nohup "${cmd[@]}" >"$log_file" 2>&1 &
    echo "==> PID: $!, log: ${log_file}"
    return
  fi
  exec "${cmd[@]}"
}

main() {
  local model_path xllm_bin
  model_path="$(resolve_model_path)"
  xllm_bin="$(resolve_xllm_bin)"
  setup_runtime_env
  set_model_defaults
  preflight "$model_path" "$xllm_bin"
  run_xllm "$model_path" "$xllm_bin"
}

main "$@"
