#!/usr/bin/env bash
# E005 Decode：DP8/EP8、DSPARK5、full graph。

set -euo pipefail

export PATH=/opt/venv/bin:/usr/local/musa/bin:${PATH}

unset LD_PRELOAD MUSA_DEVICE_PAGE_SIZE SGLANG_USE_MUSA_FUSED_KERNEL

# 代码目录：使用镜像内的固定目录。
export SGLANG_DIR=/workspace/sglang
# 镜像固定代码目录；使用本轮优化前，须将最新源码同步到该目录。

MATE_ROOT=$(python -I -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("mate").origin).parent.parent)')
export MATE_ROOT
export JIT=/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/jit/decode
export DSV41_MUTLASS_INCLUDE=${MATE_ROOT}/mate/data/mutlass/include
export DSV41_MTSHMEM_ROOT=/usr/local/mtshmem

test -d "${SGLANG_DIR}/python/sglang"
test -f "${DSV41_MUTLASS_INCLUDE}/mutlass/mutlass.h"
test -d "${MATE_ROOT}/tvm_ffi/lib"
test -d "${DSV41_MTSHMEM_ROOT}/lib"

mkdir -p \
    "/opt/jit/tilelang" \
    "/opt/jit/tmp" \
    "/opt/jit/sglang" \
    "/opt/jit/musa" \
    "/opt/jit/tvm_ffi"

export TILELANG_CACHE_DIR=/opt/jit/tilelang
export TILELANG_CLEANUP_TEMP_FILES=1
export TILELANG_TMP_DIR=/opt/jit/tmp
export SGLANG_CACHE_DIR=/opt/jit/sglang
export SGLANG_MUSA_JIT_CACHE_DIR=/opt/jit/musa
export TVM_FFI_CACHE_DIR=/opt/jit/tvm_ffi
export TL_MUTLASS_PATH=${DSV41_MUTLASS_INCLUDE}

export DSV41_ENABLE_CP=1
export DSV41_ENABLE_DEEPEP=1
# 保留原环境；D 的实际模式由下面 CLI low_latency 指定。
export DSV41_DEEPEP_MODE=normal

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_USE_DEEPGEMM_BMM=0
export SGLANG_MOE_PADDING=1
export MCCL_MAX_NCHANNELS=16
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:False

export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_DSV4_FP4_DEQUANT=0
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private
export SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN=1
export SGLANG_MUSA_PIN_MEMORY=1

export SGLANG_OPT_MHC_PRENORM_BACKEND=deepgemm
export SGLANG_OPT_MHC_PRENORM_SPLIT_K=32
export SGLANG_OPT_FUSE_WQA_WKV=1
export SGLANG_OPT_USE_FUSED_HASH_TOPK=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=1
export SGLANG_OPT_FP8_WO_A_GEMM=0

export SGLANG_OPT_MUSA_MHC_MIX_STATS_FALLBACK=0
export SGLANG_OPT_MUSA_MHC_MIX_DOT=1
export SGLANG_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS=0
export SGLANG_MUSA_MHC_COMBINE_NORM=1
export SGLANG_MUSA_CANDIDATE_GRAPH_VARIANTS=1

# E005：有效长度评分与 TopK 流水线。
export SGLANG_OPT_MUSA_INDEXER_LENGTH_PIPELINE=1
export SGLANG_OPT_MUSA_INDEXER_PAIRED_ROWS=1
export SGLANG_MUSA_MATE_DECODE_BLOCK128=1
export SGLANG_MUSA_MOE_ROUTER_FP32_OUTPUT=1

export SGLANG_W4AFP8_MUSA_PER_TOKEN_FP8_COMM=1
export SGLANG_MUSA_CP_DEQUANT_GEMM=1
export SGLANG_MUSA_W4A8_TAIL_MASKED=1
export SGLANG_MUSA_W4A8_PREFILL_QUANT_FUSION=1
export SGLANG_MUSA_W4A8_ROUTING_METADATA_FUSION=1

export SGLANG_MUSA_W4A8_DEEPEP_LL_GATEUP_BACKEND=mutlass
export SGLANG_MUSA_W4A8_DEEPEP_LL_DOWN_BACKEND=mutlass
export SGLANG_MUSA_W4A8_MUTLASS_INCLUDE=${DSV41_MUTLASS_INCLUDE}
export SGLANG_MUSA_DEEPEP_PREFILL_ONLY=0
export SGLANG_MUSA_DEEPEP_FP8_SINGLE_STAGE=1
export SGLANG_MUSA_DEEPEP_TRITON_CHUNK_SIZE=2048
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
export SGLANG_MUSA_DEEPEP_INDEXER_SCORE_BUDGET_MB=256
export SGLANG_MUSA_DEEPEP_NORMAL_SKIP_HOT_BARRIERS=1

export SGLANG_MUSA_MOE_GROUPED_GEMM=1
export SGLANG_MUSA_DECODE_MOE_SINGLE_STAGE=1
export SGLANG_MUSA_DECODE_ALIGN_M6=1
export SGLANG_MUSA_DECODE_ROUTER_SELECT=1
export SGLANG_MUSA_DECODE_INDEXER_TAIL=1
export SGLANG_MUSA_FP8_LINEAR_OPT=1
export SGLANG_MUSA_FP8_GEMM_SPECIALIZE=1
export SGLANG_MUSA_ENGRAM_PREFILL_TP=1
export SGLANG_MUSA_PREFILL_Q_ROPE_STORE=1
export SGLANG_MUSA_PREFILL_INV_ROPE=1

export SGLANG_MUSA_INDEXER_RELU_WEIGHT=1
export SGLANG_MUSA_INDEXER_FULL_SCORER=1
export SGLANG_MUSA_INDEXER_RADIX_TOPK=1
export SGLANG_MUSA_INDEXER_GEOMETRY_REUSE=1
export SGLANG_OPT_DSV4_MUSA_TILELANG_ROPE=1
export SGLANG_MUSA_FUSED_SILU_QUANT_V2=1

export SGLANG_MUSA_MHC_PREFILL_PROJECTION=1
export SGLANG_MUSA_MHC_PREFILL_CAST_RMS=1
export SGLANG_MUSA_MHC_PREFILL_CAST_MEAN=1
export SGLANG_MUSA_MHC_PREFILL_TP=1
export SGLANG_MUSA_MHC_PREFILL_TP_PROJECTION=1
export SGLANG_MUSA_MHC_PREFILL_TAIL_CHAIN=1
export SGLANG_MUSA_MHC_PREFILL_BF16X3=1
export SGLANG_MUSA_MHC_PREFILL_POST_COMBINE=1
export SGLANG_MUSA_MHC_PREFILL_CP=1
export SGLANG_MUSA_PREFILL_CANDIDATE_FULL_COVERAGE=1
export SGLANG_MUSA_PREFILL_TAIL_GRAPH=all
export SGLANG_MUSA_PREFILL_TAIL_GRAPH_MAX_CONTEXT=-1
export SGLANG_MUSA_PREFILL_TAIL_GRAPH_VALIDATE=0
export SGLANG_MUSA_DENSE_8K_TILE=1
export SGLANG_MUSA_PREFILL_8K_UE8M0_QUANT=1

export MCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

export PYTHONPATH=${SGLANG_DIR}/python:${MATE_ROOT}
export LD_LIBRARY_PATH=${DSV41_MTSHMEM_ROOT}/lib:${MATE_ROOT}/tvm_ffi/lib:/usr/local/musa/lib

export TVM_FFI_GPU_BACKEND=musa
export TVM_FFI_MUSA_ARCH_LIST=31

export SGLANG_DEEPEP_DISABLE_RECV_HOOK=1
export SGLANG_MUSA_DEEPEP_DP_DECODE=1
export SGLANG_RAGGED_VERIFY_MODE=static

export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=0

export SGLANG_MUSA_W4A8_DEEPEP_LL_MASKED_PREP=1
export SGLANG_MUSA_DEEPEP_MASK_PADDING=1
export SGLANG_MUSA_FP8_TP1_LINEAR_OPT=1
export SGLANG_MUSA_MOE_ROUTER_TWO_WARPS=1
export SGLANG_MUSA_W4A8_LL_SCALE_REUSE=1
export SGLANG_MUSA_DSV41_TP1_ROPE=1
export SGLANG_MUSA_FP8_TP1_PREQUANT=1
export SGLANG_MUSA_W4A8_LL_PUBLIC_MUTLASS=1
export SGLANG_MUSA_DSPARK_TARGET_TP_HEAD=1
export SGLANG_MUSA_FP8_TP1_VERIFY_SPLIT4=1

export MCCL_GRAPH_REGISTER=0
export PYTHONUNBUFFERED=1

SGLANG_IB_DEVICES=${SGLANG_IB_DEVICES:-$(grep -lE '200 Gb|400 Gb' /sys/class/infiniband/*/ports/*/rate | cut -d/ -f5 | sort -Vu | paste -sd, -)}


cd /workspace/sglang


# 本轮新增：隔离 JIT/记录环境；以下为最终覆盖值。
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_PROFILE_V2=0
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export SGLANG_TORCH_PROFILER_DIR=/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/B_recheck_20261117/traces/decode


exec python -m sglang.launch_server \
  --model-path /data/nfs_shared/models/DeepSeek-V4.1-Flash-1017-W4AFP8-mt \
  --tokenizer-path /data/nfs_shared/models/DeepSeek-V4.1-Flash-1017-W4AFP8-mt \
  --served-model-name deepseek-v4.1-flash-1017-w4afp8-mt \
  --trust-remote-code \
  --device musa \
  --language-model-only \
  --tp-size 8 \
  --ep-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --enable-tp-lm-head-all-to-all \
  --attention-backend dsv4 \
  --moe-runner-backend triton \
  --disable-shared-experts-fusion \
  --mem-fraction-static 0.93 \
  --chunked-prefill-size 256 \
  --context-length 4096 \
  --max-total-tokens 8192 \
  --max-running-requests 8 \
  --watchdog-timeout 1800 \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode 8 \
  --disable-overlap-schedule \
  --disable-radix-cache \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --random-seed 42 \
  --disable-attn-tp-gather \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --deepep-dispatcher-output-dtype bf16 \
  --deepep-config '{"normal_dispatch":{"num_sms":60,"num_max_nvl_chunked_send_tokens":26},"normal_combine":{"num_sms":60,"num_max_nvl_chunked_send_tokens":16}}' \
  --host 0.0.0.0 \
  --port 28588 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 28998 \
  --disaggregation-ib-device "${SGLANG_IB_DEVICES}"



