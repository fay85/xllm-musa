#!/bin/bash
# Start xLLM on MUSA (MUSA-as-CUDA build + mate FlashInfer URI ops).
#
# Tested models on musa5.1_torchmusa2.7.1_0527 (container mount: /data/feihu -> /workspace):
#   Qwen3-8B        head_dim=128  (inference OK)
#   Qwen3.5-27B     head_dim=256  (loads + HTTP up; first-token still blocked on GDN path)
#   Qwen3.5-0.8B    head_dim=256
#
# Usage (inside container):
#   bash run_xllm_musa.sh
#   MODEL_NAME=Qwen3.5-27B MUSA_VISIBLE_DEVICES=3 bash run_xllm_musa.sh
#   bash run_xllm_musa.sh --model-name Qwen3.5-27B --background
#
# Usage (from host):
#   docker exec -it musa5.1_torchmusa2.7.1_0527 \
#     bash /workspace/xllm_qwen3.5/xllm_0526/xllm/run_xllm_musa.sh

set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/workspace/model_weights}"
# Hardcoded to the Qwen3.5-27B graph-mode bring-up target so naive `bash
# run_xllm_musa.sh` (no env) picks up the correct model + concurrency caps.
# Env can still override (e.g. `MODEL_NAME=Qwen3-8B bash run_xllm_musa.sh`).
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
# Pin to the per-model concurrency cap up front so the GDN linear-state cache
# sizing (max_concurrent_requests + 2 slots) stays satisfiable, matching the
# correctness_check.sh / conc_eval.sh defaults and avoiding the "Please reduce
# max_concurrent_requests to less than 81" failure that fires when the engine
# default (200) is left intact.
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-4}"
PORT="${PORT:-8092}"
MASTER_NODE_ADDR="${MASTER_NODE_ADDR:-127.0.0.1:9748}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
LOG_DIR="${LOG_DIR:-log}"
XLLM_TILELANG_LIB="${XLLM_TILELANG_LIB:-/usr/local/lib/python3.10/dist-packages/tilelang/lib/libtilelang.so}"
export XLLM_TILELANG_LIB
BACKGROUND=0
MODEL_PATH="${MODEL_PATH:-}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --background              Run xLLM in the background (logs under LOG_DIR)
  --model-path PATH         HuggingFace model directory
  --model-name NAME         Model folder name under MODEL_ROOT (default: Qwen3-8B)
  --port PORT               HTTP port (default: 8010)
  --device INDEX            Logical musa device index passed to xLLM (default: 0)
  --musa-visible-devices N  Physical GPU(s) for MUSA_VISIBLE_DEVICES (optional)
  -h, --help                Show this help

Environment overrides:
  MODEL_ROOT, MODEL_NAME, MODEL_PATH, XLLM_BIN, FLASHINFER_OPS_PATH,
  MUSA_BACKEND_INIT_SO, MUSA_VISIBLE_DEVICES, LD_PRELOAD, LOG_DIR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background) BACKGROUND=1; shift ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --device) DEVICE_INDEX="$2"; shift 2 ;;
    --musa-visible-devices) MUSA_VISIBLE_DEVICES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

resolve_model_path() {
  if [[ -n "$MODEL_PATH" ]]; then
    echo "$MODEL_PATH"
    return
  fi

  local candidates=(
    "${MODEL_ROOT}/${MODEL_NAME}"
    "${MODEL_ROOT}/Qwen3-8B"
    "${MODEL_ROOT}/Qwen3.5-27B"
    "${MODEL_ROOT}/Qwen3.5-0.8B"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/config.json" ]]; then
      echo "$candidate"
      return
    fi
  done

  echo "Could not find model weights under ${MODEL_ROOT} (MODEL_NAME=${MODEL_NAME})." >&2
  echo "Set --model-path or MODEL_PATH explicitly." >&2
  exit 1
}

detect_head_dim() {
  local model_path="$1"
  python3 - "$model_path/config.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
tc = cfg.get("text_config", cfg)
hs = int(tc.get("hidden_size", 0) or 0)
nh = int(tc.get("num_attention_heads", 0) or 0)
hd = tc.get("head_dim")
if hd is None and nh:
    hd = hs // nh
print(int(hd or 0))
PY
}

resolve_musa_backend_init_so() {
  if [[ -n "${MUSA_BACKEND_INIT_SO:-}" && -f "$MUSA_BACKEND_INIT_SO" ]]; then
    echo "$MUSA_BACKEND_INIT_SO"
    return
  fi

  local candidate
  for candidate in \
    /workspace/libmusa_backend_init.so \
    /workspace/artifacts/libmusa_backend_init.so \
    /workspace/xllm_qwen3.5/xllm_0526/artifacts/libmusa_backend_init.so \
    /workspace/xllm_0526/artifacts/libmusa_backend_init.so; do
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
}

resolve_flashinfer_ops_path() {
  local head_dim="$1"
  local candidate
  for candidate in \
    "${FLASHINFER_OPS_PATH:-}" \
    /workspace/mate_cached_ops; do
    [[ -n "$candidate" && -d "$candidate" ]] || continue
    local probe="${candidate}/batch_decode_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_${head_dim}_head_dim_vo_${head_dim}_posenc_0_use_swa_False_use_logits_cap_False/batch_decode_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_${head_dim}_head_dim_vo_${head_dim}_posenc_0_use_swa_False_use_logits_cap_False.so"
    if [[ -f "$probe" ]]; then
      echo "$candidate"
      return
    fi
  done

  # Last resort: first existing directory (preflight will warn about missing URIs).
  for candidate in \
    "${FLASHINFER_OPS_PATH:-}" \
    /workspace/mate_cached_ops; do
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo "/workspace/mate_cached_ops"
}

setup_runtime_env() {
  export MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/musa}"
  export MUSAMAPPING_PATH="${MUSAMAPPING_PATH:-/usr/local/musa/tools/musamapping}"
  export MUDNN_HOME="${MUDNN_HOME:-/usr/local/mudnn}"
  export MCCL_HOME="${MCCL_HOME:-/workspace/MCCL_2.3.0/mccl}"
  export MUSA_LAUNCH_BLOCKING="${MUSA_LAUNCH_BLOCKING:-0}"

  export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/musa/lib:/usr/local/mudnn/lib:${MCCL_HOME}/lib:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"

  # Set after head_dim is known in main(); placeholder here for help text only.
  export FLASHINFER_OPS_PATH="${FLASHINFER_OPS_PATH:-/workspace/mate_cached_ops}"
  export MATE_HOME="${MATE_HOME:-/workspace/xllm_qwen3.5/mate_feihu}"
  MATE_FFI_HD="${MATE_FFI_HD:-256}"
  MATE_FFI_ROOT="${MATE_HOME}/build/flashinfer_ffi_hd${MATE_FFI_HD}"
  export LD_LIBRARY_PATH="${MATE_FFI_ROOT}/mate_flashinfer_prefill_ffi:${MATE_FFI_ROOT}/mate_flashinfer_batch_attention_ffi:${MATE_FFI_ROOT}/mate_flashinfer_decode_ffi:/workspace/MTTOplib/lib:/opt/intel/oneapi/mkl/lib/intel64:/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/musa/lib:/usr/local/mudnn/lib:${MCCL_HOME}/lib:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"
  # libtorch_cpu -> MKL thread DSO needs libmkl_core loaded first (oneAPI layout).
  export LD_PRELOAD="/opt/intel/oneapi/mkl/lib/intel64/libmkl_core.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
  local shim
  shim="$(resolve_musa_backend_init_so || true)"
  local musa_python="/usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so"
  if [[ -f "$musa_python" && -n "$shim" ]]; then
    export LD_PRELOAD="${musa_python}:${shim}${LD_PRELOAD:+:$LD_PRELOAD}"
  elif [[ -f "$musa_python" ]]; then
    export LD_PRELOAD="${musa_python}${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "WARNING: libmusa_backend_init.so not found; model load may fail." >&2
  elif [[ -n "$shim" ]]; then
    export LD_PRELOAD="${shim}${LD_PRELOAD:+:$LD_PRELOAD}"
  else
    echo "WARNING: neither libmusa_python.so nor libmusa_backend_init.so found." >&2
  fi

  if [[ -n "$MUSA_VISIBLE_DEVICES" ]]; then
    export MUSA_VISIBLE_DEVICES
  fi
}

resolve_xllm_bin() {
  if [[ -n "${XLLM_BIN:-}" && -x "$XLLM_BIN" ]]; then
    echo "$XLLM_BIN"
    return
  fi

  local candidate
  for candidate in \
    /workspace/xllm-git-master/build/lib.linux-x86_64-cpython-310/xllm/xllm \
    /workspace/xllm_0526/xllm/build/lib.linux-x86_64-cpython-310/xllm/xllm; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done

  echo "xLLM binary not found. Build with _build_in_container_t27.sh first." >&2
  exit 1
}

check_flashinfer_ops() {
  local head_dim="$1"
  local ops_root="$2"
  local missing=0

  local uris=(
    "batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_${head_dim}_head_dim_vo_${head_dim}_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False"
    "batch_decode_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_${head_dim}_head_dim_vo_${head_dim}_posenc_0_use_swa_False_use_logits_cap_False"
    "batch_attention_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_${head_dim}_head_dim_vo_${head_dim}_posenc_0_use_logits_soft_cap_false_use_profiler_false"
  )

  for uri in "${uris[@]}"; do
    local so="${ops_root}/${uri}/${uri}.so"
    if [[ ! -f "$so" ]]; then
      echo "WARNING: missing mate FlashInfer op: ${so}" >&2
      missing=1
    fi
  done

  if [[ "$missing" -eq 1 ]]; then
    echo "         Rebuild + deploy mate ops for head_dim=${head_dim}, e.g.:" >&2
    echo "           cd /workspace/xllm_qwen3.5/mate_feihu" >&2
    echo "           python scripts/build_flashinfer_ffi.py --head-dim ${head_dim}" >&2
    echo "           python scripts/deploy_flashinfer_ops.py --ops-root ${ops_root} --head-dim-qk ${head_dim} --head-dim-vo ${head_dim}" >&2
    echo "         Or deploy both Qwen presets:" >&2
    echo "           bash scripts/deploy_qwen_flashinfer_ops.sh --ops-root ${ops_root}" >&2
  fi
}

model_runtime_flags() {
  local model_name="$1"
  BLOCK_SIZE="${BLOCK_SIZE:-64}"
  MAX_SEQS_PER_BATCH=""
  MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-}"
  MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-0.70}"

  case "$model_name" in
    Qwen3.5-27B|Qwen3.5-27b|qwen3.5-27b|Qwen3.5-27B-FP8|Qwen3.5-27B-fp8|qwen3.5-27b-fp8)
      # Block-wise FP8 (Qwen3.5-27B-FP8) shares the exact architecture/caps of
      # the BF16 Qwen3.5-27B; only the linear weights are FP8 (dequant to BF16
      # at compute time in the step-1 semi-FP8 path). Same concurrency caps.
      # Default raised from 1 to 4 so multi-concurrency perf runs (conc_eval.sh
      # C>=2) don't see HTTP 500 rejections out of the box. Override with the
      # env var (e.g. MAX_CONCURRENT_REQUESTS=16) for higher concurrency sweeps.
      MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-4}"
      # Cap max_seqs_per_batch to the linear-state concurrency limit. Qwen3.5
      # has hybrid GDN layers whose linear state cache holds
      # (max_concurrent_requests + 2) slots, so the actual batch can never
      # exceed that. Keeping max_seqs_per_batch=64 also breaks graph-mode
      # decode warmup, which builds buckets up to max_seqs_per_batch (e.g.
      # {1,2,4,8,16,32,48,64}) and tries to allocate that many sequences from
      # the 6-slot linear pool, crashing with "Not enough blocks".
      MAX_SEQS_PER_BATCH="${MAX_SEQS_PER_BATCH:-${MAX_CONCURRENT_REQUESTS}}"
      ;;
    Qwen3.5-0.8B|Qwen3.5-0.8b)
      # 0.8B shares the hybrid GDN + full-attn architecture with 27B, so its
      # linear-state cache is also (max_concurrent_requests + 2) slots. Cap
      # max_seqs_per_batch to that to avoid graph-mode warmup OOM (`Not enough
      # blocks, total length: 16`). Allow env override for graph-debug runs.
      MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-4}"
      MAX_SEQS_PER_BATCH="${MAX_SEQS_PER_BATCH:-${MAX_CONCURRENT_REQUESTS}}"
      ;;
    *)
      ;;
  esac
}

preflight() {
  local model_path="$1"
  local xllm_bin="$2"
  local head_dim="$3"

  echo "==> xLLM binary:           $xllm_bin"
  echo "==> Model path:           $model_path"
  echo "==> Model id:             $MODEL_NAME"
  echo "==> Attention head_dim:   $head_dim"
  echo "==> Device:               musa:${DEVICE_INDEX}"
  echo "==> MUSA_VISIBLE_DEVICES: ${MUSA_VISIBLE_DEVICES:-<unset>}"
  echo "==> HTTP port:            ${PORT}"
  echo "==> FLASHINFER_OPS_PATH:  ${FLASHINFER_OPS_PATH}"
  echo "==> LD_PRELOAD:           ${LD_PRELOAD:-<unset>}"

  if [[ ! -f "${model_path}/config.json" ]]; then
    echo "ERROR: missing ${model_path}/config.json" >&2
    exit 1
  fi

  if [[ "$head_dim" -le 0 ]]; then
    echo "ERROR: could not determine head_dim from ${model_path}/config.json" >&2
    exit 1
  fi

  if [[ ! -d "${FLASHINFER_OPS_PATH}" ]]; then
    echo "WARNING: FLASHINFER_OPS_PATH does not exist: ${FLASHINFER_OPS_PATH}" >&2
  else
    check_flashinfer_ops "$head_dim" "$FLASHINFER_OPS_PATH"
  fi

  if [[ "$MODEL_NAME" == Qwen3.5-* || "$MODEL_NAME" == qwen3.5-* ]]; then
    echo "NOTE: Qwen3.5 hybrid (GDN + full-attn) may still fail on first token" >&2
    echo "      after startup (torch_musa IndexSelect on linear-attn path)." >&2
  fi
}

run_xllm() {
  local model_path="$1"
  local xllm_bin="$2"
  local log_file="$3"

  model_runtime_flags "$MODEL_NAME"

  local -a cmd=(
    "$xllm_bin"
    "--model=${model_path}"
    "--model_id=${MODEL_NAME}"
    "--backend=llm"
    "--devices=musa:${DEVICE_INDEX}"
    "--port=${PORT}"
    "--master_node_addr=${MASTER_NODE_ADDR}"
    "--nnodes=1"
    "--node_rank=0"
    "--block_size=${BLOCK_SIZE}"
    "--max_memory_utilization=${MAX_MEMORY_UTILIZATION}"
    "--enable_prefix_cache=${ENABLE_PREFIX_CACHE:-false}"
    "--enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-true}"
    "--enable_schedule_overlap=${ENABLE_SCHEDULE_OVERLAP:-true}"
  )

  cmd+=("--max_tokens_per_chunk_for_prefill=${MAX_TOKENS_PER_CHUNK_FOR_PREFILL:-8192}")

  if [[ -n "${MAX_TOKENS_PER_BATCH:-}" ]]; then
    cmd+=("--max_tokens_per_batch=${MAX_TOKENS_PER_BATCH}")
  fi

  if [[ -n "$MAX_SEQS_PER_BATCH" ]]; then
    cmd+=("--max-seqs-per-batch=${MAX_SEQS_PER_BATCH}")
  fi

  if [[ -n "$MAX_CONCURRENT_REQUESTS" ]]; then
    cmd+=("--max_concurrent_requests=${MAX_CONCURRENT_REQUESTS}")
  fi

  # Online profiling (Kineto torch.profiler -> Chrome trace JSON).
  # Pair with /start_profile + /stop_profile HTTP endpoints.
  #   ENABLE_PROFILE=1 PROFILE_DIR=/tmp/xllm_profile bash run_xllm_musa.sh ...
  if [[ "${ENABLE_PROFILE:-0}" == "1" ]]; then
    cmd+=("--enable_online_profile=true")
    if [[ -n "${PROFILE_DIR:-}" ]]; then
      cmd+=("--profile_dir=${PROFILE_DIR}")
    fi
    echo "==> Online profile: enabled (dir=${PROFILE_DIR:-cwd})"
  fi

  # Graph mode (MUSA Graph / CUDA Graph API via torch_musa). Decode-only.
  #   ENABLE_GRAPH=1 bash run_xllm_musa.sh ...
  # Disable VMM pool on first bring-up if capture fails:
  #   ENABLE_GRAPH=1 ENABLE_GRAPH_VMM_POOL=0 bash run_xllm_musa.sh ...
  if [[ "${ENABLE_GRAPH:-1}" == "1" ]]; then
    cmd+=("--enable_graph=true")
    if [[ "${ENABLE_GRAPH_DECODE_NO_PADDING:-1}" == "1" ]]; then
      cmd+=("--enable_graph_mode_decode_no_padding=true")
    fi
    if [[ "${ENABLE_GRAPH_VMM_POOL:-0}" == "1" ]]; then
      cmd+=("--enable_graph_vmm_pool=true")
    else
      cmd+=("--enable_graph_vmm_pool=false")
    fi
    cmd+=("--max_tokens_for_graph_mode=${MAX_TOKENS_FOR_GRAPH_MODE:-8192}")
    if [[ "${ENABLE_PREFILL_PIECEWISE_GRAPH:-1}" == "1" ]]; then
      cmd+=("--enable_prefill_piecewise_graph=true")
    fi
    if [[ "${ENABLE_PACKED_PREFILL:-0}" == "1" ]]; then
      cmd+=("--enable_packed_prefill=true")
    fi
    echo "==> Graph mode: enable_graph=true vmm_pool=${ENABLE_GRAPH_VMM_POOL:-0} prefill_piecewise=${ENABLE_PREFILL_PIECEWISE_GRAPH:-1} packed_prefill=${ENABLE_PACKED_PREFILL:-0}"
  fi

  # Speculative decoding. Defaults: off. Qwen3.5 MTP requires exporting draft
  # weights first:
  #   python3 tools/export_mtp.py --input-dir Qwen3.5-27B --output-dir Qwen3.5-27B-mtp
  # Then launch with separate draft checkpoint (graph MTP is not ready yet):
  #   NUM_SPECULATIVE_TOKENS=1 SPECULATIVE_ALGORITHM=MTP \
  #   DRAFT_MODEL_PATH=/path/to/Qwen3.5-27B-mtp \
  #   DRAFT_DEVICES="musa:${DEVICE_INDEX}" \
  #     ENABLE_GRAPH=0 bash run_xllm_musa.sh ...
  # Suffix decoding needs no draft model; just set NUM_SPECULATIVE_TOKENS>0
  # and SPECULATIVE_ALGORITHM=Suffix.
  local num_spec="${NUM_SPECULATIVE_TOKENS:-0}"
  if [[ "${ENABLE_GRAPH:-1}" == "1" && "$num_spec" -gt 0 ]]; then
    echo "==> WARNING: speculative decoding (MTP) is disabled while ENABLE_GRAPH=1"
    echo "==>          (graph-mode MTP is not ready). Set ENABLE_GRAPH=0 to use MTP."
    num_spec=0
  fi
  if [[ "$num_spec" -gt 0 ]]; then
    cmd+=("--num_speculative_tokens=${num_spec}")
    cmd+=("--speculative_algorithm=${SPECULATIVE_ALGORITHM:-MTP}")
    if [[ -n "${DRAFT_MODEL_PATH:-}" ]]; then
      cmd+=("--draft_model=${DRAFT_MODEL_PATH}")
    fi
    cmd+=("--draft_devices=${DRAFT_DEVICES:-musa:${DEVICE_INDEX}}")
    echo "==> Speculative decoding: algo=${SPECULATIVE_ALGORITHM:-MTP} k=${num_spec}"
    if [[ -n "${DRAFT_MODEL_PATH:-}" ]]; then
      echo "==> Draft model:           ${DRAFT_MODEL_PATH}"
      echo "==> Draft devices:         ${DRAFT_DEVICES:-musa:${DEVICE_INDEX}}"
    fi
  fi

  if [[ "$MODEL_NAME" == Qwen3.5-* || "$MODEL_NAME" == qwen3.5-* ]]; then
    echo "==> Qwen3.5 GDN: fused decode enabled by default (chunk_gated_delta_rule prefill)"
  fi

  if [[ "$BACKGROUND" -eq 1 ]]; then
    mkdir -p "$LOG_DIR"
    echo "==> Starting xLLM in background; log: ${log_file}"
    nohup "${cmd[@]}" >"$log_file" 2>&1 &
    echo "==> PID: $!"
    echo "==> Tail log: tail -f ${log_file}"
    echo "==> After 'Application startup complete', test:"
    echo "curl http://127.0.0.1:${PORT}/v1/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"model\":\"${MODEL_NAME}\",\"prompt\":\"Hello\",\"max_tokens\":16,\"temperature\":0}'"
  else
    echo "==> Starting xLLM in foreground (model loads now; Ctrl+C to stop)"
    exec "${cmd[@]}"
  fi
}

main() {
  setup_runtime_env
  local model_path head_dim xllm_bin log_slug
  model_path="$(resolve_model_path)"
  head_dim="$(detect_head_dim "$model_path")"
  export FLASHINFER_OPS_PATH="$(resolve_flashinfer_ops_path "$head_dim")"
  xllm_bin="$(resolve_xllm_bin)"
  preflight "$model_path" "$xllm_bin" "$head_dim"
  log_slug="$(echo "$MODEL_NAME" | tr '/ ' '__')"
  run_xllm "$model_path" "$xllm_bin" "${LOG_DIR}/xllm_${log_slug}.log"
}

main "$@"
