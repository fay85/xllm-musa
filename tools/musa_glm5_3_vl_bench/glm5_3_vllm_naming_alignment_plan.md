<!-- Copyright 2026 The xLLM Authors. All Rights Reserved. -->

# GLM-5.3-Flash 命名对齐方案

本文比较本工作区的 xLLM MUSA backend 与本机 vLLM checkout 中
GLM-5.3-Flash 的层、kernel、文件和主要符号命名，并记录已经落地的命名迁移方案。
源码迁移保留了旧 C++ 符号的兼容别名，避免影响现有调用方；checkpoint key、注册 key
和 cache layout 均保持不变。

## 1. 对比基线和结论

| 项目 | 基线 |
| --- | --- |
| xLLM MUSA | `b85269417`，模型入口为 `glm5_next` |
| vLLM | `ebac032b92`，目录为 `vllm/models/glm5next/` |
| vLLM 主模型文件 | `common/model.py`、`common/attention.py`、`common/kda.py`、`common/multimodal.py`、`common/mtp.py` |
| vLLM MUSA 对应物 | 当前 xLLM MUSA C++ 层和 kernel，主要位于 `core/layers/musa/`、`core/kernels/musa/` |

当前两边的模型级名称已经基本一致：`Glm5NextModel`、
`Glm5NextDecoderLayer`、`Glm5NextForCausalLM` 等名称可直接互认。主要差异在
注意力和 IndexPool：xLLM 使用 `KDA`、`SparseAttention`、`sparse_mla` 和
`glm5_next_kpool`，而 vLLM 使用 `LinearAttention`、`MLAAttention`、
`SparseAttnIndexerKpool`、`sparse_indexer` 和 `kpool_compress`。

建议采用下面的边界：

1. 保留 checkpoint、Hugging Face 和注册表可见的 `glm5_next`。它是模型架构和
   权重前缀的一部分，不能因为 vLLM 的 Python 包目录写成 `glm5next` 就改成
   `glm5next`。
2. 对齐模型层和职责名称：`Glm5NextKDA` 改为
   `Glm5NextLinearAttention`，`Glm5NextSparseAttention` 改为
   `Glm5NextMLAAttention`。
3. 将当前混合在 `glm5_next_kpool.cu` 中的 IndexPool 逻辑拆成
   `indexer` 和 `kpool_compress` 两个职责，与 vLLM 的文件边界一致。
4. `MHC`、`KDA`、`RoutedMoe` 是可复用的 MUSA 公共算子/层，保留通用命名；只在
   GLM 层的成员和调用位置使用 vLLM 的模型级语义名称。
5. 所有 C++ 文件名继续遵守 xLLM 的 `snake_case` 规则。因此 vLLM 的
   `glm5next`、`flashmla_sparse.py` 在 C++ 中分别表达为 `glm5_next`、
   `flash_mla_sparse`，不直接复制 Python 文件名。

## 2. 模型层和主要符号映射

### 2.1 已经对齐、建议保留的名称

| vLLM | 当前 xLLM MUSA | 处理建议 |
| --- | --- | --- |
| `Glm5NextModel` | `Glm5NextModelImpl` / `Glm5NextModel` | 保留。`Impl` 是 LibTorch 的现有封装约定，`TORCH_MODULE` 导出的 `Glm5NextModel` 已对齐 |
| `Glm5NextForCausalLM` | `Glm5NextForCausalLMImpl` / `Glm5NextForCausalLM` | 保留 |
| `Glm5NextDecoderLayer` | `Glm5NextDecoderLayerImpl` / `Glm5NextDecoderLayer` | 保留 |
| `Glm5NextMLP` | `DenseMLP` | 见下方：`DenseMLP` 是通用层，可保留；GLM 层成员建议语义化为 `mlp_` |
| `Glm5NextMoE` | `RoutedMoe` | 见下方：公共实现保留 `RoutedMoe`，不要为了单模型改名 |
| `MHCPreOp` / `MHCPostOp` | `MHCPre` / `MHCPost` | 保留现有 xLLM LibTorch wrapper；`MHCPreImpl` / `MHCPostImpl` 与 vLLM 的职责一一对应 |

### 2.2 建议重命名的模型层

| 当前 xLLM 名称 | 对齐后的名称 | 对齐原因 |
| --- | --- | --- |
| `Glm5NextKDAImpl` / `Glm5NextKDA` | `Glm5NextLinearAttentionImpl` / `Glm5NextLinearAttention` | vLLM 在 `common/kda.py` 中公开的层名是 `Glm5NextLinearAttention`；`KDA` 是实现算法名，不是 GLM 层职责名 |
| `Glm5NextKDAInputProjectionImpl` / `Glm5NextKDAInputProjection` | `Glm5NextLinearAttentionInputProjectionImpl` / `Glm5NextLinearAttentionInputProjection` | 与父层保持一致；仍可在注释和 kernel API 中保留 KDA |
| `Glm5NextSparseAttentionImpl` / `Glm5NextSparseAttention` | `Glm5NextMLAAttentionImpl` / `Glm5NextMLAAttention` | vLLM 的 `common/attention.py` 使用 `Glm5NextMLAAttention`；该层实际包含 MLA 投影、Indexer、候选选择、cache 写入和 sparse MLA 输出 |
| 当前 `Glm5NextSparseAttentionImpl` 内部的 IndexPool 代码 | `Glm5NextIndexerImpl` / `Glm5NextIndexer` | vLLM 把 Indexer 作为独立职责；拆出后 `Glm5NextMLAAttention` 只组合 Indexer 和 MLA backend |
| `GlmKPoolSelection` | `Glm5NextIndexerSelection` | 该结构不是通用 KPool 结果，而是 GLM Indexer 的 token candidates、lengths、rows；名称应表达产生它的职责 |
| `shared_selection` | `shared_indexer_selection` | 明确这是跨 DSA 层复用的 Indexer 选择，避免与普通 KPool 临时结果混淆 |

推荐的 MUSA decoder 成员最终形态如下：

```text
Glm5NextDecoderLayer
  ├── MHCPre attn_hc_pre
  ├── Glm5NextLinearAttention self_attn       # linear_attention 层
  ├── Glm5NextMLAAttention self_attn          # DSA/MLA 层，二选一
  ├── DenseMLP mlp                            # dense 层
  ├── RoutedMoeGate gate                      # sparse MoE router
  └── RoutedMoe moe                           # sparse MoE experts
```

上面的 `self_attn` 是语义示意。当前 LibTorch module registration 使用同一个
checkpoint 前缀 `self_attn.`，重命名 C++ 成员不能改变权重路径。

## 3. 文件名映射

### 3.1 第一优先级：模型层文件

| 当前文件 | 推荐文件 | 变更 |
| --- | --- | --- |
| `xllm/core/layers/musa/glm5_next_kda.h` | `xllm/core/layers/musa/glm5_next/glm5_next_linear_attention.h` | 移入 GLM 专属目录，并按 vLLM 层名命名 |
| `xllm/core/layers/musa/glm5_next_kda.cpp` | `xllm/core/layers/musa/glm5_next/glm5_next_linear_attention.cpp` | 同上 |
| `xllm/core/layers/musa/glm5_next_sparse_attention.h` | `xllm/core/layers/musa/glm5_next/glm5_next_mla_attention.h` | 以 vLLM `Glm5NextMLAAttention` 为准 |
| `xllm/core/layers/musa/glm5_next_sparse_attention.cpp` | `xllm/core/layers/musa/glm5_next/glm5_next_mla_attention.cpp` | 同上 |
| `xllm/core/layers/musa/glm5_next/glm5_next_decoder_layer.h` | 保留 | 该路径已经是 GLM 专属且职责对应 vLLM `Glm5NextDecoderLayer` |
| `xllm/core/layers/musa/glm5_next/glm5_next_decoder_layer.cpp` | 保留 | 同上 |

建议先完成文件和类名迁移，再进行 Indexer 的职责拆分。这样可以把“命名对齐”
和“执行逻辑重构”分开审查。

### 3.2 第二优先级：Indexer、KPool 和 sparse MLA kernel

| 当前文件 | 推荐文件 | vLLM 对应职责 |
| --- | --- | --- |
| `xllm/core/kernels/musa/glm5_next_kpool.h` | `xllm/core/kernels/musa/glm5_next_indexer.h` + `xllm/core/kernels/musa/glm5_next_kpool_compress.h` | `common/sparse_indexer.py` / `nvidia/sparse_indexer.py` 与 `nvidia/ops/kpool_compress.py` |
| `xllm/core/kernels/musa/glm5_next_kpool.cu` | `xllm/core/kernels/musa/glm5_next_indexer.cu` + `xllm/core/kernels/musa/glm5_next_kpool_compress.cu` | 将 top-k、candidate expansion、metadata 与 pool compression、Hadamard、FP8 cache write 分开 |
| `xllm/core/kernels/musa/sparse_mla.h` | `xllm/core/kernels/musa/flash_mla_sparse.h` | 对齐 vLLM `flashmla_sparse.py` 的 sparse MLA backend 语义 |
| `xllm/core/kernels/musa/sparse_mla.cpp` | `xllm/core/kernels/musa/flash_mla_sparse.cpp` | 同上；C++ 文件使用 `flash_mla_sparse` 的 snake case |
| `xllm/core/kernels/musa/sparse_mla_cache.cu` | `xllm/core/kernels/musa/flash_mla_sparse_cache.cu` | 明确这是 sparse MLA 的 cache write，而非通用 cache API |

推荐拆分后的公开接口命名：

```text
kernel::musa::glm5_next_indexer(...)
kernel::musa::glm5_next_kpool_compress(...)
kernel::musa::flash_mla_sparse_bf16(...)
kernel::musa::flash_mla_sparse_fp8(...)
kernel::musa::flash_mla_sparse_cache_write(...)
```

如果后续确认 `flash_mla_sparse` 只服务 GLM-5.3-Flash，应在接口注释中写清
MP31、NoPE、cache layout 和 shape 限制；如果要复用于其他模型，再把模型无关部分
下沉到 `sparse_mla` 公共层，并保留 `Glm5NextMLAAttention` 作为适配层。

### 3.3 应保持通用名称的 MUSA 文件

| 文件 | vLLM 参考 | 建议 |
| --- | --- | --- |
| `xllm/core/kernels/musa/kda.h/.cpp` | `glm5next/nvidia/ops/third_party/kda/` | 保留 `kda`；它是 `flash_kda` 的公共 MUSA 封装，GLM 层通过 `Glm5NextLinearAttention` 调用 |
| `xllm/core/kernels/musa/mhc_ops.h/.cpp` | `model_executor/kernels/mhc/` | 保留 `mhc_ops`；不要改成 `glm5_next_mhc` |
| `xllm/core/kernels/musa/mhc_pre.cu` | vLLM MHC pre kernel | 保留 |
| `xllm/core/kernels/musa/mhc_post.cu` | vLLM MHC post kernel | 保留 |
| `xllm/core/kernels/musa/mhc_head.cu` | vLLM `HCHeadOp` | 保留 |
| `xllm/core/layers/musa/routed_moe.h/.cpp` | `Glm5NextMoE` 的专家执行部分 | 保留；该实现已经是可复用的 MUSA MoE 层 |
| `xllm/core/layers/musa/routed_moe_gate.h/.cpp` | `Glm5NextMoE` 的 router | 保留；GLM 通过参数配置 sigmoid、top-k 和 scaling factor |
| vLLM `nvidia/mega_moe.py:Glm5NextMegaMoEExperts` | 当前没有同名 MUSA 文件；专家执行位于 `routed_moe.*` | 不新增同名薄封装；如果后续接入专用 MUSA mega-MoE，再新增 `glm5_next_mega_moe.*` |
| vLLM `nvidia/ops/fused_eh_norm.py:fused_eh_norm` | 当前 MUSA 没有对应 kernel | 只有 native MTP 需要该融合时才新增 `glm5_next_fused_eh_norm.*`；不要把普通 RMSNorm 改名为该名称 |

这里的原则是：文件名描述可复用的执行组件，类名在 GLM decoder 组合处描述
模型语义。避免为同一个公共实现同时维护 `RoutedMoe` 和
`Glm5NextMoE` 两套实现。

### 3.4 vLLM 有而当前 MUSA 没有的缓存/辅助类

vLLM `common/attention.py` 中还有 `Glm5NextIndexerCache`、
`Glm5NextTailCache` 和 `Indexer`。当前 xLLM 将它们的职责分散在通用
`KVCache`、cache shape/estimation、`Glm5NextSparseAttentionImpl` 和
`glm5_next_kpool.cu` 中。因此建议按职责映射，而不是为了文件名创建空类：

| vLLM 名称 | 当前 xLLM 所在位置 | 推荐处理 |
| --- | --- | --- |
| `Glm5NextIndexerCache` | `KVCache` 的 `index_cache` 及 MUSA cache shape/estimation | 保持通用 `KVCache`；如果将来 cache backend 需要独立策略，再新增 `Glm5NextIndexerCache` |
| `Glm5NextTailCache` | `KVCache` 的 `kpool_tail` 及 MUSA cache shape/estimation | 保持通用 `KVCache`；名称通过 `kpool_tail` 字段表达 |
| `Indexer` | 当前合并在 `Glm5NextSparseAttentionImpl` 和 `glm5_next_kpool.cu` | 阶段 C 抽出 `Glm5NextIndexer` |
| `SparseAttnIndexerKpool` | 当前 `glm5_next_kpool(...)` | 推荐 wrapper 名为 `Glm5NextIndexer`，底层 kernel 入口为 `glm5_next_indexer` |

## 4. 多模态、Python 和 MTP 命名

### 4.1 多模态

vLLM 的视觉类在 `vllm/models/glm5next/common/multimodal.py` 中分成：

```text
Glm5NextVisionPatchEmbed
Glm5NextVisionMLP
Glm5NextVisionAttention
Glm5NextVisionBlock
Glm5NextPatchMerger
Glm5NextVisionTransformer
Glm5NextProcessingInfo
Glm5NextMultiModalProcessor
```

当前 MUSA 只有注册适配器
`xllm/models/vlm/glm5_next_vlm.h`，其中的别名是
`Glm5NextVLMultimodalProcessor`，视觉 tower 仍在 Python。建议如下：

| 当前 | 推荐 | 说明 |
| --- | --- | --- |
| `Glm5NextVLMultimodalProcessor` | `Glm5NextMultiModalProcessor` | 与 vLLM 完全对齐；`VL` 信息已由 VLM 注册位置表达 |
| `models/vlm/glm5_next_vlm.h` | `models/vlm/glm5_next_multimodal.h` | 文件名表达 processor/多模态职责；注册 key 继续为 `glm5_next` |
| `xllm/python/models/glm5_next_vl.py` | `xllm/python/models/glm5next/common/multimodal.py` | 仅在 Python MUSA/VLM 路径继续保留时迁移；当前 MUSA native 路径不依赖它 |
| 尚无 MUSA vision layer | 按 vLLM 名称新增 `Glm5NextVisionTransformer`、`Glm5NextVisionBlock` 等 | 只有真正实现原生 MUSA vision tower 时才新增，不要用 processor 名称冒充 vision layer |

`Glm5NextVLMultimodalProcessor` 的改名会影响注册宏和 include 路径，但不应影响
checkpoint 的 `model.visual.*` 权重前缀。

### 4.2 MTP

vLLM 的 MTP 名称为：

```text
Glm5NextMultiTokenPredictorLayer
Glm5NextMultiTokenPredictor
Glm5NextMTP
```

当前 MUSA 还没有对应的 native MTP layer；已有
`xllm/models/llm/glm5_next_mtp_args.h` 是参数加载，不是 MTP 执行层。建议：

- 参数文件继续叫 `glm5_next_mtp_args.h`，因为它描述 checkpoint/config loader；
- native MUSA MTP 实现新增到
  `xllm/core/layers/musa/glm5_next/glm5_next_mtp.h/.cpp`，类名采用
  `Glm5NextMultiTokenPredictorLayer` 和 `Glm5NextMultiTokenPredictor`；
- MTP 的 decoder attention 使用 `Glm5NextMLAAttention`，并显式命名
  `shared_indexer_selection`，对应 vLLM 的 `set_skip_topk` / top-k reuse；
- 在 MTP 接入完成前，不要把 `glm5_next_mtp_args.h` 重命名为 `mtp.h`，避免把
  loader 和执行层混为一谈。

## 5. 推荐的目标目录结构

完成第一、第二阶段后，MUSA 相关目录建议接近下面的结构：

```text
xllm/core/layers/musa/
├── glm5_next/
│   ├── glm5_next_decoder_layer.h/.cpp
│   ├── glm5_next_linear_attention.h/.cpp
│   ├── glm5_next_mla_attention.h/.cpp
│   └── glm5_next_mtp.h/.cpp                 # 后续实现
├── hyper_connection.h/.cpp                  # 公共 mHC layer
├── routed_moe.h/.cpp                         # 公共 MoE layer
└── routed_moe_gate.h/.cpp                    # 公共 MoE router

xllm/core/kernels/musa/
├── kda.h/.cpp
├── mhc_ops.h/.cpp
├── mhc_pre.cu
├── mhc_post.cu
├── mhc_head.cu
├── glm5_next_indexer.h/.cu
├── glm5_next_kpool_compress.h/.cu
├── flash_mla_sparse.h/.cpp
└── flash_mla_sparse_cache.cu

xllm/models/vlm/
└── glm5_next_multimodal.h                     # processor registration
```

## 6. 分阶段落地顺序与当前状态

### 阶段 A：建立兼容别名，已完成

1. 在原有头文件中增加目标名称的 `using` 或 wrapper alias：
   `Glm5NextLinearAttention`、`Glm5NextMLAAttention`、
   `Glm5NextIndexerSelection`。
2. 将新名称用于 decoder layer、model layer 和测试新代码；旧名称暂时保留。
3. 不改变 LibTorch module registration 名称：仍使用 `self_attn`、`mlp`、`gate`
   和 checkpoint 的权重前缀。
4. 让 `git grep` 能同时确认 vLLM 名称和兼容旧名称，减少一次性 rename 的风险。

### 阶段 B：迁移模型层文件，已完成

1. 迁移 `glm5_next_kda.*` 到
   `glm5_next/glm5_next_linear_attention.*`。
2. 迁移 `glm5_next_sparse_attention.*` 到
   `glm5_next/glm5_next_mla_attention.*`。
3. 已更新 `core/layers/musa/CMakeLists.txt`、include、model layer 成员和测试
   target；旧类名通过头文件兼容别名保留。
4. 对每个 touched C++/CUDA 文件运行 clang-format 20.1.6，并检查 include 去重。

### 阶段 C：Indexer、KPool 和 sparse MLA 命名迁移，已完成命名部分

1. pool compression、Hadamard、FP8 cache write 仍属于同一个现有实现单元；后续如需
   独立优化，再抽出 `glm5_next_kpool_compress.cu`，本次不制造空的拆分文件。
2. candidate metadata、paged MQA logits、top-k 和 candidate expansion 已迁移到
   `glm5_next_indexer.cu`。
3. 将 `GlmKPoolSelection` 改成 `Glm5NextIndexerSelection`，保留一段时间的
   类型别名；调用方参数改成 `shared_indexer_selection`。
4. 已将 `sparse_mla.*` 和 `sparse_mla_cache.cu` 改为
   `flash_mla_sparse*`，并在 layer 调用侧使用 `flash_mla_sparse_*` 统一 wrapper 前缀；
   旧低层函数名保留为 ABI 兼容实现。
5. 同步 `core/kernels/musa/CMakeLists.txt`、TileLang module URI、测试名称和
   bench 脚本；URI 若被外部预编译产物使用，应提供旧 URI fallback 或重新导出。

### 阶段 D：多模态命名迁移，已完成；MTP 保持后续实现

1. 已将 processor alias 改为 `Glm5NextMultiModalProcessor`，并迁移 VLM header。
2. 只有实现 MUSA vision tower 后，才按 vLLM 的
   `VisionPatchEmbed`、`VisionAttention`、`VisionBlock`、`PatchMerger`、
   `VisionTransformer` 命名新增层。
3. native MTP 接入时使用 vLLM 的 `MultiTokenPredictor` 命名，并复用
   `Glm5NextMLAAttention` 和 Indexer selection。

## 7. 必须同步的构建、注册和测试位置

每次文件 rename 至少检查下面的位置：

| 位置 | 检查内容 |
| --- | --- |
| `xllm/core/layers/musa/CMakeLists.txt` | 新旧头文件和源文件清单 |
| `xllm/core/kernels/musa/CMakeLists.txt` | 新 kernel wrapper、`.cu` 和 cache writer |
| `xllm/models/models.h` | MUSA model/VLM include 和注册可见性 |
| `xllm/models/llm/musa/glm5_next.h` | model class、layer type 和 selection 类型 |
| `xllm/models/vlm/glm5_next_vlm.h` | processor alias 和 `REGISTER_MULTIMODAL_PROCESSOR` |
| `xllm/core/layers/musa/glm5_next/glm5_next_decoder_layer.*` | layer member、权重前缀和 forward 参数 |
| `tools/musa_glm5_3_vl_bench/compile_musa_sparse_mla.py` | TileLang module 名称和导出 URI |
| `tools/musa_glm5_3_vl_bench/validate_musa_sparse_attention.py` | 测试可执行文件和日志名 |
| MUSA sparse attention tests | 测试 target、fixture、函数名和日志路径 |

必须特别区分两类名称：

- `glm5_next`、`model.language_model.layers.*`、`self_attn.*` 等是 checkpoint 或
  注册协议，保持不变；
- C++ 类型、文件、kernel wrapper 和测试名称是本方案的重命名范围。

## 8. 验证要求

命名迁移完成后按以下顺序验证：

```bash
git diff --check
rg -n "glm5_next_kda|glm5_next_sparse_attention|sparse_mla|GlmKPoolSelection" xllm tools
rg -n "Glm5NextLinearAttention|Glm5NextMLAAttention|Glm5NextIndexerSelection|flash_mla_sparse|glm5_next_indexer" xllm tools
cmake --build build/cmake.linux-x86_64-cpython-310 \
  --target glm5_next_mla_attention_test flash_mla_sparse_test -j 32
python tools/musa_glm5_3_vl_bench/validate_musa_sparse_attention.py
```

第一条 `rg` 在过渡期允许命中 forwarding header、兼容 alias 和旧 TileLang URI；
阶段 C 完成后应只剩明确标注的兼容路径。测试重点是：

- BF16 和 FP8 sparse MLA 输出；
- Indexer top-k、pool tail、cache write 和跨层 shared selection；
- prefill、chunked prefill、decode 和页边界；
- TP/MCCL 两卡路径；
- 权重加载、`verify_loaded_weights()` 和真实 GLM-5.3-Flash 请求。

## 9. 最终建议

优先执行阶段 A 至 C。它们能让源码浏览、日志、测试和 vLLM 对照都使用同一套
领域名称，同时不改变 `glm5_next` 注册 key、权重路径、cache layout 或 MUSA
算子语义。多模态文件和 MTP 名称随后跟随真实实现接入；processor、vision tower
和 native MTP 不应通过重命名文件提前制造“已实现”的假象。
