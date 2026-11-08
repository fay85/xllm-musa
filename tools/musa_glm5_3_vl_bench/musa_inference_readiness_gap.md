# GLM-5.3-Flash (MUSA) 推理跑通差距清单

更新日期：2026-11-07
分支基线：`albert/glm5.3-flash` (`3a13ce4ec`)

## 1. 当前已完成模块

- **Linear Attention/KDA 算子与整层**：`xllm/core/kernels/musa/kda.cpp`、`xllm/core/layers/musa/glm5_next/glm5_next_linear_attention.cpp`（含输入投影、causal conv1d、`flash_kda`、output gate、输出投影与状态索引管理）。
- **MLA + Indexer 算子与整层**：`xllm/core/kernels/musa/glm5_next_indexer.cu`、`xllm/core/kernels/musa/flash_mla_sparse.cpp`、`xllm/core/layers/musa/glm5_next/glm5_next_mla_attention.cpp`（含 FP8 block-scale 权重反量化、Indexer TopK 选择、MLA KV cache 写入与稀疏注意力计算）。
- **mHC 算子与整层**：`xllm/core/kernels/musa/mhc_*.cu`、`xllm/core/layers/musa/hyper_connection.cpp`（含 `mhc_expand`、`MHCPre`、`MHCPost`、跨层 `PendingMHC` 融合与 `mhc_contract_mean`）。
- **Decoder 与 CausalLM 骨架**：`xllm/core/layers/musa/glm5_next/glm5_next_decoder_layer.cpp`、`xllm/models/llm/musa/glm5_next.h`（已串联 KDA / Sparse Attention、mHC、Dense MLP 与 Sparse MoE）。

---

## 2. 距离端到端跑通还差的项（按阻塞顺序）

### 2.1 纯文本推理（`--backend=llm`）必补项（已全部打通并完成真机 8 卡端到端推理验证）

1. **[已完成] 模型参数解析器（Args Loader）在 MUSA 分支注册**
   - **修复**：在 `xllm/models/models.h` 的 `USE_MUSA` 分支引入 `llm/glm5_next.h`，并在 `xllm/models/llm/glm5_next.h` 中同时兼容 `text_config.dtype` 与 `text_config.torch_dtype`。
   - **验证**：`hf_model_loader_test --gtest_filter=*Glm5NextRegistersArgsLoaderAndParsesConfig*` 通过，`xllm` 二进制与 `libmodels.a` 均已包含 `glm5_next_args_loader_registered` 与 `glm5_next_registered`。

2. **[已完成] Decoder 接入 Sparse MoE（第 3~44 层）**
   - **修复**：
     - 在 `Glm5NextDecoderLayerImpl` 与 `Glm5NextModelImpl` 中打通 `sparse` MLP 分支，新增通用 MUSA `RoutedMoeGate` + `RoutedMoe`（`xllm/core/layers/musa/routed_moe_gate.{h,cpp}`、`xllm/core/layers/musa/routed_moe.{h,cpp}`，不改动 `deepseek_v4` 任何文件）（对齐 FP32 sigmoid router、`norm_topk_prob=true`、`routed_scaling_factor=2.5`、`swiglu_limit=10.0`、shared expert 分支、分片增量权重加载、`weight_scale_inv` FP8 128x128 block-scale 权重加载及 `verify_loaded_weights` 校验）；
     - 将 `xllm/core/kernels/cuda/moe/moe_compute_index.cu` 的 `moe_prefix_sum_kernel` 扩展至支持最多 1024 个专家（支持 GLM-5.3-Flash 的 `n_routed_experts=288` 单卡 EP=1 运行）。
   - **验证**：`glm5_next_mla_attention_test`（含新增的 288 专家 FP8 `weight_scale_inv` Prefill/Decode 整层测试 `DecoderLayerSparseMoeFp8ScaleInvMatchesReferenceInPrefillAndDecode` 及双卡 MCCL TP 测试）全部通过。

3. **[已完成] Engine KV/Index Cache 分配与 MUSA 算子契约对齐**
   - **修复**：
     - 在 `xllm/core/platform/platform.h` 中为 MUSA 开启 `uses_compressed_kpool_cache()` 与 `supports_dsa_indexer_cache_elision()`；
     - 在 `xllm/core/framework/model/model_args.h` 中增加 `uses_musa_compressed_kpool_tail`，并在 `kv_cache_estimation.cpp`、`kv_cache_shape.cpp`、`kv_cache_utils.cpp` 中对齐 MUSA Rate-4 压缩池契约（`index_cache` 为 `{pages, 64, 1, 132}` `torch::kUInt8`，`kpool_tail` 为 `{state_slots, 2, 4, 128}` `torch::kBFloat16`）；
     - 在 `xllm/models/llm/glm5_next.h`、`xllm/core/runtime/executor.cpp`、`xllm/core/distributed_runtime/master.cpp` 中对齐 `enable_mla=true`、`block_size=256` 及 MUSA eager 执行约束（`requires_eager_execution=true` / `enable_graph=false`）。
   - **验证**：`kv_cache_estimation_test`、`kv_cache_test` 及 MUSA 整层缓存分配测试 `GlmSparseTest.EngineKVCacheAndSharedSelectionCrossLayerForward` 全部通过。

4. **[已完成] Shared Indexer（`shared_selection`）跨层复用接线**
   - **修复**：
     - 在 `xllm/core/layers/common/dsa_topk_share_plan.h` 中支持 `args.indexer_types()`（`"full"` / `"shared"`）计划解析；
     - 在 `Glm5NextDecoderLayerImpl` 中通过 `DsaTopkSharePlan(args).layer_uses_indexer(layer_id)` 决定每层 `enable_indexer_`，并在 `Glm5NextDecoderLayerImpl::forward` 与 `Glm5NextModelImpl::forward` 中跨层保存和透传 `kernel::musa::GlmKPoolSelection`。
   - **验证**：`GlmSparseTest.EngineKVCacheAndSharedSelectionCrossLayerForward` 验证了 `"full"` + `"shared"` 两层 DSA 在 Prefill / Decode 下跨层复用 `shared_selection` 且 `"shared"` 层跳过 `index_cache` / `kpool_tail` 分配。

5. **[已完成] 端到端 LLM 推理真机验证（`/data/nfs_shared/models/GLM-5.3-Flash`）**
   - **运行时边界修复**：
     - `xllm/core/layers/musa/glm5_next/glm5_next_linear_attention.cpp`：兼容 `BatchInputBuilder` 默认填充的就地状态槽位（`linear_state_read_ids == linear_state_ids`）；
     - `xllm/models/llm/musa/glm5_next.h`：当 MUSA `BatchInputBuilder` 预置仅含 `fa3_metadata` 的 `params.attn_metadata`（`!q_cu_seq_lens.defined()`）时，在 `prepare_attention_metadata` 中补齐完整 `AttentionMetadataBuilder::build` 并同步就地 `linear_state_read_ids`。
   - **8x MTT S5000 (TP=8) 端到端验证结果**：
     - 62 个 safetensors 分片（45 层混合 KDA + DSA + Dense MLP + 288 专家 FP8 MoE）全部加载并通过 `verify_loaded_weights()`；
     - 引擎分配缓存：`k_cache [4426, 256, 1, 512]`、`index_cache [4426, 64, 1, 132]`、`kpool_tail [640, 2, 4, 128]`、`conv_cache [640, 3072, 3]`、`ssm_cache [640, 8, 128, 128]`；
     - `/v1/completions` 与 `/v1/chat/completions`（256 tokens 连续生成）Prefill + Decode 端到端推理输出语义正常。

---

### 2.2 多模态推理（`--backend=vlm`）必补项（已全部打通并完成真机 8 卡端到端 VLM 推理验证）

6. **[已完成] VLM Processor 注册与 MUSA 原生 Vision Tower（`Glm5NextForConditionalGeneration`）**
   - **修复**：
     - 在 `xllm/models/llm/glm5_next.h` 中补齐 `vision_config.projection_intermediate_size`（`mm_projector_hidden_size=10240`）与 `mm_head_dim=64` 参数解析；
     - 新增 `xllm/models/vlm/musa/glm5_next_vlm.h` 实现 MUSA 原生 `Glm5NextVisionTransformer`（`Conv3d` PatchEmbed 带 bias、2D Vision RoPE、24 层带 per-head `q_norm`/`k_norm` 与 `swiglu_limit=10.0` 的 `Glm5NextVisionBlock`、`Conv2d` 空间下采样、带 `post_projection_norm` LayerNorm 的 `Glm5NextVisionPatchMerger`）与 `Glm5NextForConditionalGeneration`，并在 `xllm/models/models.h` 的 `USE_MUSA` 分支引入 `vlm/glm5_next_multimodal.h` 与 `vlm/musa/glm5_next_vlm.h`；
     - 在 `xllm/models/model_registry.cpp` 中保留双注册模型的 `llm` 默认后端映射，并在 `xllm/core/distributed_runtime/master.cpp` 中修正 `EngineType::VLM` 下的 `model_type` 解析与 `options_.block_size(256)` 传递。
   - **验证**：
     - 单元测试 `hf_model_loader_test --gtest_filter=*Glm5Next*`（含真实 `model-00062-of-00062.safetensors` 347 个视觉权重加载、与 PyTorch 参考数值对齐 `< 5e-3`、多图 varlen 视觉编码及 `merge_multimodal_embeddings` 验证）全部通过；
     - 8x MTT S5000 (`--backend=vlm`) 真机服务启动成功，纯文本与多模态图文请求（几何图形空间位置/颜色识别、非正方形动态分辨率 OCR 识别）端到端推理输出准确。
