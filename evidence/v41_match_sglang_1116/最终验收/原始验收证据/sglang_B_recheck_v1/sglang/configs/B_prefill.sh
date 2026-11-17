#!/usr/bin/env bash
set -euo pipefail

# 代码目录：使用镜像内的固定目录。
export SGLANG_DIR=/workspace/sglang

# Python 与动态库：复用容器现有依赖，只优先选择当前代码和已安装的 MATE。
export PATH=/opt/venv/bin:/root/.local/bin:/usr/local/musa/bin:/usr/local/musa/mudnn/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export MUSA_HOME=/usr/local/musa
MATE_ROOT=$(python -I -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("mate").origin).parent.parent)')
export MATE_ROOT
export PYTHONPATH="${MATE_ROOT}:${SGLANG_DIR}/python"
export DSV41_MTSHMEM_ROOT=/usr/local/mtshmem
# full graph 依赖已验证的兼容驱动库，不能直接改用容器默认的系统 libmusa。
MUSA_DRIVER_LIB_DIR=/opt/venv/lib/python3.10/site-packages/tvm_ffi/lib
export LD_LIBRARY_PATH="${MUSA_DRIVER_LIB_DIR}:${DSV41_MTSHMEM_ROOT}/lib:/usr/local/musa/lib:/usr/local/musa/compat:/usr/local/openmpi/lib:/usr/local/musa/mudnn/lib"
export TVM_FFI_GPU_BACKEND=musa
export TVM_FFI_MUSA_ARCH_LIST=31
unset LD_PRELOAD MUSA_DEVICE_PAGE_SIZE SGLANG_USE_MUSA_FUSED_KERNEL
# 清除先前实验留下的诊断/预热开关。
unset SGLANG_LOG_GC SGLANG_DIAG_BYPASS_HEALTH_GENERATE
unset SGLANG_MUSA_PD_PREFILL_WARMUP_SHAPES PREFILL_STARTUP_SHAPES

# JIT 缓存与编译头文件：复用同一缓存，避免重复编译；保持已验证的 MUTLASS 版本。
JIT=/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/jit/prefill
export TILELANG_CACHE_DIR=${JIT}/tilelang
export TILELANG_TMP_DIR=${JIT}/tmp
export TILELANG_CLEANUP_TEMP_FILES=1
export SGLANG_CACHE_DIR=${JIT}/sglang
export SGLANG_MUSA_JIT_CACHE_DIR=${JIT}/musa
export TVM_FFI_CACHE_DIR=${JIT}/tvm_ffi
export TL_MUTLASS_PATH="${MATE_ROOT}/mate/data/mutlass/include"
export DSV41_MUTLASS_INCLUDE=${TL_MUTLASS_PATH}
export SGLANG_MUSA_W4A8_MUTLASS_INCLUDE=${TL_MUTLASS_PATH}
mkdir -p "${TILELANG_CACHE_DIR}" "${TILELANG_TMP_DIR}" "${SGLANG_CACHE_DIR}" "${SGLANG_MUSA_JIT_CACHE_DIR}" "${TVM_FFI_CACHE_DIR}"

# 并行、通信与显存：TP8/EP8/CP8，DeepEP normal，保留原通信调优值。
export DSV41_ENABLE_CP=1
export DSV41_ENABLE_DEEPEP=1
export DSV41_DEEPEP_MODE=normal
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:False
export SGLANG_MUSA_PIN_MEMORY=1
export MCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export MCCL_MAX_NCHANNELS=16
export SGLANG_MUSA_DEEPEP_PREFILL_ONLY=1
export SGLANG_MUSA_DEEPEP_FP8_SINGLE_STAGE=1
export SGLANG_MUSA_DEEPEP_TRITON_CHUNK_SIZE=2048
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_MUSA_DEEPEP_INDEXER_SCORE_BUDGET_MB=256
export SGLANG_MUSA_DEEPEP_NORMAL_SKIP_HOT_BARRIERS=1

# Dense/FP8 路径：优先 Block128，未覆盖形状保留 BF16 回退，不启用 Block32。
export SGLANG_USE_DEEPGEMM_BMM=0
export SGLANG_DEEP_GEMM_BLOCK_M=128
export SGLANG_MUSA_MATE_DENSE_BLOCK128=1
export SGLANG_MUSA_MATE_JIT_BLOCK32=0
export SGLANG_MUSA_MATE_MUBIN_BLOCK32=0
export SGLANG_MUSA_CP_DEQUANT_GEMM=1
export SGLANG_MUSA_FP8_LINEAR_OPT=1
export SGLANG_MUSA_FP8_GEMM_SPECIALIZE=1
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_MUSA_DENSE_8K_TILE=1
export SGLANG_MUSA_PREFILL_8K_UE8M0_QUANT=1
export SGLANG_MUSA_PREFILL_8K_UE8M0_QUANT_M4096=1

# W4A8/MoE：大 M 预填充 M256、尾块掩码、量化及路由元数据融合。
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_DSV4_FP4_DEQUANT=0
export SGLANG_MOE_PADDING=1
export SGLANG_W4AFP8_MUSA_PER_TOKEN_FP8_COMM=1
export SGLANG_MUSA_W4A8_PREFILL_M256=1
export SGLANG_MUSA_W4A8_TAIL_MASKED=1
export SGLANG_MUSA_W4A8_PREFILL_QUANT_FUSION=1
export SGLANG_MUSA_W4A8_ROUTING_METADATA_FUSION=1
export SGLANG_MUSA_W4A8_DEEPEP_LL_GATEUP_BACKEND=mutlass
export SGLANG_MUSA_W4A8_DEEPEP_LL_DOWN_BACKEND=mutlass
export SGLANG_MUSA_MOE_GROUPED_GEMM=1
export SGLANG_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS=0
export SGLANG_MUSA_FUSED_SILU_QUANT_V2=1

# Attention/Indexer：scorer+mask 融合、候选索引裁剪和 mask 展开融合。
export SGLANG_OPT_FUSE_WQA_WKV=1
export SGLANG_OPT_USE_FUSED_HASH_TOPK=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_OPT_DSV4_MUSA_TILELANG_ROPE=1
export SGLANG_MUSA_PREFILL_Q_ROPE_STORE=1
export SGLANG_MUSA_PREFILL_INV_ROPE=1
export SGLANG_MUSA_INDEXER_RELU_WEIGHT=1
export SGLANG_MUSA_INDEXER_FULL_SCORER=1
export SGLANG_MUSA_INDEXER_RADIX_TOPK=1
export SGLANG_MUSA_INDEXER_RADIX_TOPK_ANY_SHAPE=0
export SGLANG_MUSA_INDEXER_GEOMETRY_REUSE=1
export SGLANG_MUSA_INDEXER_SCORER_MASK_FUSION=1
export SGLANG_MUSA_PREFILL_CANDIDATE_FULL_COVERAGE=1
export SGLANG_MUSA_PREFILL_CANDIDATE_SKIP_COMPACT_INDICES=1
export SGLANG_MUSA_PREFILL_CANDIDATE_MASK_EXPAND_FUSION=1
export SGLANG_MUSA_CP_LOCAL_COMPRESSOR=1

# MHC/CP：投影与尾段融合，只计算 M16 / M>=1024 路径实际需要的 RMS。
export SGLANG_OPT_MHC_PRENORM_BACKEND=deepgemm
export SGLANG_OPT_MHC_PRENORM_SPLIT_K=32
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=1
export SGLANG_OPT_MUSA_MHC_MIX_STATS_FALLBACK=0
export SGLANG_OPT_MUSA_MHC_MIX_DOT=1
# Keep MoE logits in FP32 before correction bias and top-k.
export SGLANG_MUSA_MOE_ROUTER_FP32_OUTPUT=1
export SGLANG_MUSA_MHC_PREFILL_PROJECTION=1
export SGLANG_MUSA_MHC_PREFILL_CAST_RMS=1
export SGLANG_MUSA_MHC_PREFILL_CAST_MEAN=1
export SGLANG_MUSA_MHC_PREFILL_TP=1
export SGLANG_MUSA_MHC_PREFILL_TP_PROJECTION=1
export SGLANG_MUSA_MHC_PREFILL_TAIL_CHAIN=1
export SGLANG_MUSA_MHC_PREFILL_BF16X3=1
export SGLANG_MUSA_MHC_PREFILL_POST_COMBINE=1
export SGLANG_MUSA_MHC_PREFILL_CP=1
export SGLANG_MUSA_MHC_CP_RMS_ONLY_M16=1
export SGLANG_MUSA_MHC_CP_RMS_ONLY_M1024_PLUS=1

# 图与解码侧配套：tail graph 全开，保留 full decode graph 与自动 batch 配置。
export SGLANG_MUSA_PREFILL_TAIL_GRAPH=all
export SGLANG_MUSA_PREFILL_TAIL_GRAPH_MAX_CONTEXT=-1
export SGLANG_MUSA_PREFILL_TAIL_GRAPH_VALIDATE=0
export SGLANG_MUSA_DECODE_MOE_SINGLE_STAGE=1
export SGLANG_MUSA_DECODE_ALIGN_M6=1
export SGLANG_MUSA_DECODE_ROUTER_SELECT=1
export SGLANG_MUSA_DECODE_INDEXER_TAIL=1

# Engram：使用独立、固定内存的 host table 和 TP 预填充路径。
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private
export SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN=1
export SGLANG_MUSA_ENGRAM_PREFILL_TP=1

# PD 传输：仅 IB 设备按机器自动检测，固定参数直接写在启动命令中。
unset SGLANG_IB_DEVICES
SGLANG_IB_DEVICES=$(grep -lE '200 Gb|400 Gb' /sys/class/infiniband/*/ports/*/rate | cut -d/ -f5 | sort -Vu | paste -sd, -)

# 启动服务：性能实验配置；本脚本不表示完整模型精度认证。
# 核对实际源码与关键优化开关，防止旧代码静默忽略脚本设置。

# 本轮新增：隔离 JIT/记录环境；以下为最终覆盖值。
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_PROFILE_V2=0
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export SGLANG_TORCH_PROFILER_DIR=/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/B_recheck_20261117/traces/prefill


exec python -u -m sglang.launch_server \
  --model-path /data/nfs_shared/models/DeepSeek-V4.1-Flash-1017-W4AFP8-mt \
  --tokenizer-path /data/nfs_shared/models/DeepSeek-V4.1-Flash-1017-W4AFP8-mt \
  --served-model-name deepseek-v4.1-flash-1017-w4afp8-mt \
  --trust-remote-code --device musa --language-model-only \
  --tp-size 8 --ep-size 8 --dp-size 1 \
  --attention-backend dsv4 --moe-runner-backend triton \
  --disable-radix-cache --disable-shared-experts-fusion \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 256 \
  --context-length 4096 \
  --max-total-tokens 8192 \
  --max-running-requests 8 \
  --watchdog-timeout 600 \
  --tokenizer-backend fastokens \
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
  --enable-decoder-swa-bounded-replay \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --random-seed 42 \
  --disable-attn-tp-gather --moe-a2a-backend deepep \
  --deepep-mode normal --deepep-dispatcher-output-dtype bf16 \
  --deepep-config '{"normal_dispatch":{"num_sms":32,"num_max_nvl_chunked_send_tokens":26},"normal_combine":{"num_sms":32,"num_max_nvl_chunked_send_tokens":16}}' \
  --enable-prefill-cp --cp-strategy interleave --enable-cp-decode-attn-tp --attn-cp-size 8 \
  --host 0.0.0.0 --port 28586 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 28998 --disaggregation-ib-device "${SGLANG_IB_DEVICES}"


