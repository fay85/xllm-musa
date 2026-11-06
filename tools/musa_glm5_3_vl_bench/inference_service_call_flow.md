<!-- Copyright 2026 The xLLM Authors. All Rights Reserved. -->

# GLM-5.3-Flash 在 xLLM/MUSA 上的推理服务完整调用链调研

调研日期：2026-11-06。源码基线：`88107189d53ad6f926bb6f1bb23ecfe4b3ccf64e`；模型：`/data/models/GLM-5.3-Flash/`。本文讨论当前工作区的 xLLM MUSA 后端，包含文本、图片和视频路径。除明确注明的历史测试记录外，结论来自本次配置、全部 safetensors 文件头和实际源码检查；本次没有启动完整模型服务、执行权重数值验证或测量吞吐。

## 1. 先回答：现在能否直接起服务？

**当前 checkout 还不能直接用这个 checkpoint 在 MUSA 上启动完整 GLM-5.3-Flash 服务。** 已有两个重要基础组件：MUSA KDA 算子、MUSA GLM sparse attention 整层，但它们还没有串成 GLM 模型和服务。不能把已有单测结果当作端到端支持。

| 路径 | 现状 | 直接依据 |
| --- | --- | --- |
| MUSA native 文本模型 | 缺少 GLM 模型/decoder 注册和组合 | `models/models.h` 的 `USE_MUSA` 分支没有包含 `llm/glm5_next.h` 或 GLM native 模型；当前 native GLM text 实现在 MLU |
| MUSA native 多模态模型 | 缺少 GLM VLM、视觉编码器和 processor 注册 | MUSA 分支没有包含 `vlm/glm5_next_vlm.h`；该头本身只注册 processor，没有 C++ VLM 模型 |
| 直接使用 `--model_impl=python` | 不能绕过上述缺口 | `create_llm_model()` 只在 CUDA/NPU 允许 Python；VLM factory 虽可尝试创建 `PyCausalLM`，但 `xllm.python.initialize_runtime()` 只选择 CUDA/NPU kernel 包，无 MUSA 包 |
| KDA | 有 `kernel::musa::flash_kda()`，尚无 MUSA GLM KDA layer 调用 | 全仓调用检索只发现 kernel 定义和 `kda_test.cpp` |
| 稀疏 MLA + IndexPool | 已有 `Glm5NextSparseAttentionImpl` 整层 | 服务模型中无该类的调用；当前调用方为整层测试 |
| FP8 权重 | sparse attention loader 已处理本模型这种 block scale | 不能据此推断全模型 MoE/MLP loader 也已经适配 |

更早的失败可能发生在 `HFModelLoader::load_model_args()`：它要先从模型注册表找到 args loader，而 MUSA 模型 include 分支尚未引入 GLM args 注册。不能断言当前运行一定首先打印某条 factory 错误；这里没有执行启动来定位第一条 fatal。

仓库 [GLM 部署示例](../../docs/src/content/docs/zh/cookbook/autoregressive_models/glm/glm_5_3_flash.md) 使用 Ascend、HCCL、Python 模型和 W8A8 checkpoint。其参数与设备 kernel 不能直接当作本机 MUSA FP8 的已验证方案。

## 2. 本地 checkpoint 的实际组成

模型配置和文件头的可复查快照见 [checkpoint_inventory.json](checkpoint_inventory.json)，其中包含配置 SHA-256、全部 shard 统计、逐层角色以及代表性权重的 shape/dtype。这里只读取文件头，没有将 306 GiB 权重整体搬进内存。

| 项目 | 实际值 |
| --- | --- |
| 外层架构 / model_type | `Glm5NextForConditionalGeneration` / `glm5_next` |
| 文本子模型 | `glm5_next_text`，45 层，hidden=4096，vocab=154880 |
| 注意力排布 | 34 KDA + 11 DSA/NoPE MLA；三层 KDA 后一层 DSA，最后还有第 44 层 KDA |
| MLP 排布 | 0–2 为 dense；3–44 为 MoE，共 42 层 |
| MoE | 288 routed experts，每 token 选 8；1 shared expert；expert intermediate=2048 |
| dense intermediate | 12288 |
| mHC | 4 条 residual stream，Sinkhorn=20，hc_eps=1e-6 |
| KDA | H=64，Dk=Dv=128，短卷积宽度4，gate 下界 -5 |
| MLA | 64 heads，Q rank=1536，KV rank=512，QK NoPE/V dim=256，RoPE dim=0 |
| Indexer | 32 heads × 128 dim，kPool 压缩率4，token Top-K=2048，所有 DSA 的 indexer 都是 `full` |
| 视觉 | 24 blocks，hidden=1024，16 heads，head_dim=64，MLP=4096 |
| patch / merge | 3×2×14×14，空间2×2 merge；输出4096，merger intermediate=10240 |
| 位置上限 | 配置1048576；不是已经验证可运行的服务长度 |
| MTP | 配置1层，文件头确有 `model.language_model.layers.45.*`；主干不应跑46层 |
| 精度 | text dtype=BF16；多数矩阵 FP8 E4M3，dynamic activation，weight block=128×128 |
| shard / tensor | 62 个 safetensors；76108 个 tensor |
| tensor payload | 328326771576 bytes = **305.78 GiB**；含文件头的物理大小328337455672 bytes |
| dtype 数量 | F8_E4M3=37338，BF16=1141，F32=37629（含大量 scale） |

权重空间下限与服务显存不是同一个数。305.78/8≈38.22 GiB/卡只是假定完全均分的文件 payload；实际还有 replicated 权重、cache、激活、workspace、通信 buffer 和加载峰值。若把全部 FP8 权重展开成 BF16，仅 payload 就约598.58 GiB；不能用“每 token 只激活8个专家”估算常驻权重显存。

典型真实权重（矩阵按 `[out,in]`）：

| 权重（省略 `model.language_model.layers.i.`） | shape / dtype | 含义 |
| --- | --- | --- |
| `self_attn.q_proj/k_proj/v_proj.weight`，i=0 | 每个[8192,4096] BF16 | KDA 的64×128通道 |
| `self_attn.b_proj.weight` | [64,4096] BF16 | beta 原始 logits |
| `self_attn.f_a_proj/g_a_proj.weight` | 每个[128,4096] BF16 | forget/output gate 低秩输入 |
| `self_attn.f_b_proj/g_b_proj.weight` | 每个[8192,128] BF16 | 两路 gate 展开 |
| `self_attn.q_conv1d/k_conv1d/v_conv1d.weight` | 每个[8192,1,4] BF16 | depthwise causal conv |
| `self_attn.A_log` / `dt_bias` | [64] / [8192] F32 | gate 参数，不应随全模型 cast 丢失精度 |
| `self_attn.q_a_proj.weight`，i=3 | [1536,4096] FP8；scale[12,32] F32 | MLA Q 压缩 |
| `self_attn.kv_a_proj_with_mqa.weight` | [512,4096] FP8；scale[4,32] | KV latent，无 rope 拼接 |
| `self_attn.q_b_proj.weight` | [16384,1536] FP8 | 展开64×256 Q |
| `self_attn.kv_b_proj.weight` | [32768,512] BF16 | 每 head 的 K/V 升维矩阵 |
| `self_attn.o_proj.weight` | [4096,16384] FP8 | MLA 输出投影 |
| `mlp.experts.0.gate_proj/up_proj.weight` | 每个[2048,4096] FP8 | 一个 expert 的两路输入 |
| `mlp.experts.0.down_proj.weight` | [4096,2048] FP8 | expert 输出 |
| `hc_attn_fn` / `hc_ffn_fn` | 每个[24,16384] BF16 | mHC 混合投影；scale/base 分别[3]/[24] F32 |

embedding 位于 `model.language_model.embed_tokens.weight`，LM head 位于 `lm_head.weight`，均为[154880,4096] BF16且不共享；视觉前缀为 `model.visual.`。多数 FP8 tensor 有对应 `.weight_scale_inv`，名字中的 inv 不意味着加载时取倒数：现有 sparse attention loader 是 FP32(weight) **乘** block scale，再转 BF16。

## 3. 完整服务的两阶段：启动与请求

下面是仓库已有服务框架的调用顺序。GLM 的 MUSA model/layer 组合位置标记为“待接入”；Python/NPU 路径用于说明已有的 GLM 语义参考。

### 3.1 服务启动、权重加载与 cache 初始化

```text
xllm/xllm.cpp::main
  → initialize_musa_backend（TorchMUSA PrivateUse1 hooks）
  → initialize_configs → run → create_options
  → 读取 config.json，选择 llm/vlm backend
  → create_master → LLMMaster / VLMMaster
      → LLMEngine / VLMEngine::init → init_model
      → HFModelLoader
          → load_model_args → ModelRegistry args loader（GLM/MUSA 待注册）
          → quant/tokenizer/config；safetensors StateDict
      → 各 rank WorkerImpl::init_model
          → create_llm_model / create_vlm_model（GLM/MUSA 待实现）
          → 构建 embedding、45个 decoder、final norm、lm_head
          → [VLM] vision tower
          → load_state_dict / load_weights → verify_loaded_weights
      → 初始化 block manager、KV/index/linear state、executor
      → [可选 graph warmup；本模型首版应关闭]
  → scheduler、JinjaChatTemplate、tokenizer、request factory
  → API service / brpc server
```

模型类型决定 model 类；`layer_types` 决定每层 attention；`mlp_layer_types` 决定 dense/MoE；线性层的量化配置、dtype、shape 和执行阶段决定 kernel。不是每个 HF tensor 名字都直接映射一个独立 kernel：QKV、gate/up 常被融合，KV 的 K/V 权重可被 absorb 成两次 BMM。

权重加载还要处理：外层 `model.language_model.` 前缀、mHC `hc_attn_*`/`hc_ffn_*`、跨 shard 到达的 FP8 weight/scale、TP 分片。现有 Python `Glm5NextForCausalLM.load_weights()` 支持真实 key alias，但主要是 BF16/W8A8 loader；`_load_qlinear()` 的非 INT8 分支直接加载浮点 tensor，`_load_experts_bf16()` 也没有这里需要的 FP8 block scale 处理。把 FP8 直接 `copy_` 到 BF16 会转换编码值而丢失 scale，不能认为它正确加载了本 checkpoint。

### 3.2 一次 HTTP 请求到第一个 token

```text
POST /v1/chat/completions
  → [VLM] MMChatServiceImpl::process_async_impl
  → VLMMaster::handle_request(messages, params, ...)
  → VLMRequestFactory::create
      → 解析/加载媒体，processor->process_multimodal
      → JinjaChatTemplate::apply(messages, tools, chat_template_kwargs)
      → process_prompt：展开 image/video placeholder → tokenizer
      → Request / Sequence：tokens、采样参数、媒体、输出回调
  → ContinuousScheduler::add_request
  → ContinuousScheduler::step → schedule_request
      → 分配 cache 页和 linear state slot
      → 组 Batch，生成 positions、query boundaries、block_table、slot_mapping
  → VLMEngine::step → 各 rank VLMWorkerImpl::step_internal
  → model_executor_->forward
      → [prefill有媒体] encode → vision_model → 媒体 embeddings
      → 与 token embeddings 合并
      → 45层 decoder forward（GLM/MUSA 待组合）
      → final hidden states
  → model_->logits（选定的最后有效token行）→ lm_head
  → Sampler::forward → token id / logprob
  → scheduler更新sequence → AsyncResponseProcessor → tokenizer decode
  → HTTP JSON / stream增量回调
```

纯文本使用 `ChatServiceImpl/LLMMaster/LLMEngine/LLMWorkerImpl` 的对应路径；VLM 服务也可接纯文本，请求没有媒体时跳过 vision。`backend=vlm` 是模型/服务类型，`USE_MUSA` 是编译设备后端，两者含义不同。

### 3.3 连续 decode

第一轮 prefill 消费 prompt，输出第一个生成 token。随后每轮将上一个 token 作为输入，每条 active sequence 通常只有1个 query token；position 增加，历史状态保留，45层再运行一次，再采样。它不会重算整段 prompt，也不需要每轮重跑视觉编码器。达到 EOS `[154820,154827,154829]`、`max_tokens`、stop 或取消条件后结束，释放/回收状态；开启 prefix cache 时还涉及状态快照恢复。

chunked prefill 是把同一个 prompt 分多轮，不等于多条新请求；每个 chunk 必须接上上一轮 KDA recurrent/conv 状态、MLA 历史页和 IndexPool tail。服务“第二次请求”也是重要测试：复用的 state slot 必须清理或恢复，不能串入上一条请求状态。

## 4. 每个 decoder 层内部的统一骨架

令 T 为本轮 packed token 数，D=4096，r=4。embedding `[T,D]` 扩成 `[T,r,D]`（4份初始相同的 residual streams）。每层包含两处 mHC：

```text
X[T,4,4096]
  → mHC-pre(attn)：得到 collapsed[T,4096]、post[T,4]、comb[T,4,4]
  → RMSNorm(eps=1e-5)
  → KDA 或 DSA/MLA
  → mHC-post(attn)：子层输出按post放回各stream，comb混合旧residual
  → mHC-pre(ffn)
  → RMSNorm(eps=1e-5)
  → Dense SwiGLU 或 MoE
  → mHC-post(ffn)
  → 下一层 X[T,4,4096]
```

最后4个 stream **取均值** → final RMSNorm → lm_head；不能把 mHC 简化成通常的 `x + attention(x)`。mHC 权重混合使用 FP32 参数/计算语义；源权重 `fn` 本身是 BF16，转 FP32 不能恢复其已舍入的信息。

语义参考：Python `Glm5NextDecoderLayer._forward_fused()`、`Glm5NextHyperConnection`、`Glm5NextModel.forward()`；native 结构参考 MLU `Glm5NextDecoderLayerImpl`。MUSA DeepSeek-V4 已有 `deepseek_v4_hc_pre_decode`、`deepseek_v4_hc_post` 和 final collapse 相关 kernel，可评估复用；**还没有在 GLM decoder 中连接和验证**。要核对 stream 混合方向、post 系数、Sinkhorn/eps、是否融合了后续 norm，以及 head 的均值尺度，不能只因接口名相似就宣称通用。

## 5. 45个主干层逐层对应

编号从0开始；K代表下节的 KDA pipeline，S代表 sparse MLA + 完整 IndexPool，D代表 dense SwiGLU，M代表 MoE。每行均带上述两处 mHC 和两处 RMSNorm；列出的 layer 类是现有语义参考，MUSA decoder 仍需组合。

| layer_id | attention | FFN | 现有 xLLM 语义类 |
| --- | --- | --- | --- |
| 0 | K | D | `Glm5NextKdaAttention` + `Glm5NextMLP` |
| 1 | K | D | 同0 |
| 2 | K | D | 同0 |
| 3 | S | M | `Glm5NextMlaAttention` + `Glm5NextMoE` |
| 4 | K | M | `Glm5NextKdaAttention` + `Glm5NextMoE` |
| 5 | K | M | 同4 |
| 6 | K | M | 同4 |
| 7 | S | M | 同3 |
| 8 | K | M | 同4 |
| 9 | K | M | 同4 |
| 10 | K | M | 同4 |
| 11 | S | M | 同3 |
| 12 | K | M | 同4 |
| 13 | K | M | 同4 |
| 14 | K | M | 同4 |
| 15 | S | M | 同3 |
| 16 | K | M | 同4 |
| 17 | K | M | 同4 |
| 18 | K | M | 同4 |
| 19 | S | M | 同3 |
| 20 | K | M | 同4 |
| 21 | K | M | 同4 |
| 22 | K | M | 同4 |
| 23 | S | M | 同3 |
| 24 | K | M | 同4 |
| 25 | K | M | 同4 |
| 26 | K | M | 同4 |
| 27 | S | M | 同3 |
| 28 | K | M | 同4 |
| 29 | K | M | 同4 |
| 30 | K | M | 同4 |
| 31 | S | M | 同3 |
| 32 | K | M | 同4 |
| 33 | K | M | 同4 |
| 34 | K | M | 同4 |
| 35 | S | M | 同3 |
| 36 | K | M | 同4 |
| 37 | K | M | 同4 |
| 38 | K | M | 同4 |
| 39 | S | M | 同3 |
| 40 | K | M | 同4 |
| 41 | K | M | 同4 |
| 42 | K | M | 同4 |
| 43 | S | M | 同3 |
| 44 | K | M | 同4 |

MUSA 对 S 已有可调用的 `layer::Glm5NextSparseAttention`；对 K 只有核心 `kernel::musa::flash_kda`，还缺投影/卷积/状态/norm 的模型层封装。`indexer_types` 全是 full，所以本 checkpoint 的11层 S 均应自己算选择，不在层间默认共享 topk；MTP iteration 的共享字段是另一件事。

## 6. KDA：从权重到 kernel

### 6.1 层语义及张量

用单卡全局尺寸说明；TP=p时按 head 拆分，local H=64/p，qkv_dim=8192/p。

1. 融合输入线性投影：按 **Q、K、V、beta、f_a、g_a** 顺序 pack，不能交换后两个同为128维的投影。Q/K/V 各8192，beta为64，f_a/g_a各128。
2. Q/K/V depthwise causal conv(width=4) + SiLU；prefill按sequence分界，decode读取并更新该sequence的conv state。
3. `f_b(f_a)` 产生 raw forget gate `[T,H,128]`；`g_b(g_a)` 产生输出 gate。同名 g 不是 KDA recurrence 的衰减值。
4. 对 Q/K 做 L2 normalize。beta_raw 在 kernel 内 sigmoid；衰减 log gate 为 `-5 * sigmoid(exp(A_log) * (raw_gate + dt_bias))`。
5. delta-rule recurrence 消费 Q/K/V、beta、gate 和初始 S，得到 O 和最终 S。V-first 状态为 `[N,H,128,128]`；不能按 GDN 的每head标量 gate 直接替代 KDA 的每channel gate。
6. `o_norm` 对128维做 RMSNorm，乘 **sigmoid(output_gate)**，不是 SiLU gate；flatten→`o_proj`→TP归约。

### 6.2 当前 MUSA 映射与缺口

| 子步骤 | 可参考 layer / wrapper | MUSA kernel / 路径 | 状态 |
| --- | --- | --- | --- |
| input/f_b/g_b/o投影 | Python `Glm5NextKdaAttention`；MLU `Glm5NextKDAImpl` | 公共 Linear → `kernel::matmul` → `musa::matmul` | primitives已存在，GLM封装待做 |
| causal conv | MLU KDA；MUSA Qwen GDN state组织 | `musa/causal_conv1d.cu` | 需核对缓存布局/sequence槽并接入 |
| chunk recurrence | `musa/kda.h` | `flash_kda` → TVM-FFI prepare + recurrence | 已有核心算子 |
| 单token decode | 同上 | 当前 `flash_kda` 也可处理T=1；不等于已接入专用state-pool decode | gather初始状态和scatter最终状态待做 |
| sigmoid-gated RMSNorm | `_RMSNormGated`；MLU `sigmoid_gate`参考 | 现有MUSA RMSNorm和sigmoid/mul可组合，或新增融合 | 不能误用SiLU门控的norm |
| state生命周期 | `linear_state_indices/read_indices`、conv/ssm cache | device gather/scatter/clear等 | 调度、槽复用、prefix/checkpoint语义待接入 |

KDA attention 选型已满足技能要求的 **MATE优先**：现有 `flash_kda.fwd` 对应的 MATE chunk KDA 有可用离线FFI接口，所以此处不需要降到 muDNN 或 muBLAS 拼 recurrence。投影 GEMM 属于独立非 attention 子算子。

`flash_kda` 当前支持FP16/BF16激活、128 head dim、packed varlen；`A_log/dt_bias` FP32；initial/final state 可用激活精度或FP32。模型应先用BF16激活+FP32 state验证。chunk=16是安装版MATE编译与workspace契约，不是prompt长度限制。workspace查询返回0不代表无临时显存，wrapper内部仍分配workspace。

模块路径：`$FLASHINFER_OPS_PATH/<function_name>/<function_name>.so`。核心函数名为 `prepare_kda_dtype_*_gate_*_varlen_1_normqk_1_seqlens_i64_arch_mp31` 和 `kda_recurrence_dtype_*_varlen_1_seqlens_i64_state_*_in_1_out_1_cs_0_arch_mp31`。由 [compile_musa_kda.py](compile_musa_kda.py) 离线生成；线上走device pointer、FFI和stream guard，不运行Python/JIT。

## 7. DSA / sparse MLA：现有最完整的 MUSA 层

入口：[Glm5NextSparseAttentionImpl](../../xllm/core/layers/musa/glm5_next_sparse_attention.cpp)。这里的“full_attn_layers”表示相对线性attention的层类型，长上下文实际仍由DSA选择稀疏token，并非总是dense全历史attention。

### 7.1 投影、Indexer、attention调用顺序

```text
hidden[T,4096]
  → fused_qkv_a_proj = concat(q_a,kv_a)：[T,1536+512]
  → q_a RMSNorm → q_b → Q[T,Hlocal,256]
  → kv_a RMSNorm → latent c[T,512]
  → Indexer：
       wq_b(q_a_norm) → indexQ[T,32,128]
       wk(hidden) → LayerNorm(weight+bias,eps=1e-6) → indexK[T,128]
       weights_proj(hidden) → FP32 head weights[T,32]
       compress_gate(hidden) + APE[4,128] → 4-token pooling
       cache/tail更新 → Hadamard/FP8 → MATE paged MQA logits
       causal mask → TorchMUSA topk(512 pools)
       展开成最多2048 raw token + 未完成pool的0~3个tail token
  → Qlatent = Q × W_K：[T,Hlocal,512]
  → 将当前c写入主paged cache
  → 候选逻辑token映射为物理slot → sparse MLA(scale=1/sqrt(256)=0.0625)
  → Z[T,Hlocal,512] × W_V → [T,Hlocal,256]
  → flatten → row-parallel o_proj → all_reduce → [T,4096]
```

`kv_b_proj` 被拆成每head的 W_K/W_V 供两次 `torch::bmm`，不用为全部历史token展开64头的完整K/V。MLA没有RoPE（qk_rope_head_dim=0），不要套用普通DeepSeek有rope分支；视觉RoPE仍需要运行。

| 子步骤 | 层内部实现 | kernel/库边界 |
| --- | --- | --- |
| q_a/kv_a融合 | `ReplicatedLinear` | 公共Linear、MUSA matmul |
| q_b/kv_b | `ColumnParallelLinear` | 按完整head分片；投影/权重reshape |
| q_a/kv_a norm | `RMSNorm` | `fused_layernorm`→共享移植的RMSNorm kernel |
| index K norm | `RMSNorm::set_layernorm_mode()` | MUSA分支改走`torch::layer_norm`，包含bias |
| pooling/tail、Hadamard/量化、候选展开 | `glm5_next_kpool` | `musa/glm5_next_kpool.cu`原生设备kernel |
| Indexer打分 | `glm_kpool_score_v1` | MATE FFI `fp8_paged_mqa_logits`，metadata模块为`glm_kpool_metadata_v1` |
| pool Top-K | `scores.topk` | TorchMUSA设备算子；不是host Top-K |
| cache写入/候选映射 | `sparse_mla_cache_write`等 | `musa/sparse_mla_cache.cu` |
| BF16主attention | `sparse_mla_bf16` | MATE导出`sparse_mla_bf16_v1_mp31_h{H}`的`main` |
| FP8主attention | `sparse_mla_nope_fp8` | MATE NoPE sparse MLA模块和`mla_ops` metadata |
| 输出投影 | `RowParallelLinear` | 公共Linear、进程组归约 |

attention选择：MATE已经提供匹配NoPE/512 latent/候选格式的BF16和FP8 kernel及scoring入口，因此不降级到muDNN SDPA，也不使用muBLAS GEMM+softmax冒充完整支持。BMM是absorb/恢复投影，不能据此把主attention标为muBLAS。

实现会在设备上把 FP8 attention矩阵按128×128 scale展开为BF16后加载并分片；这属于**加载格式支持**，当前整层不是所有矩阵直接做FP8 GEMM。加载器暂存跨shard未齐全的weight/scale，完成后释放。当前要求eager、MP31、BF16激活，local heads=8/16/32/64，页大小256，CP=1、有效KV split=1；明确拒绝stream capture。普通TP对64头对应p=1/2/4/8，仍需完整模型数值/显存验证。

### 7.2 三类状态，必须一起管理

| 状态 | 当前形状 / dtype | 生命周期 |
| --- | --- | --- |
| KDA recurrent | 每sequence每KDA层 `[Hlocal,128,128]` FP32（建议） | 每token推进；不随context长度线性增长 |
| KDA conv | 每sequence Q/K/V各通道最近width-1个有效输入；具体storage依wrapper而定 | chunk/decode衔接、槽复用与恢复 |
| MLA主cache BF16 | `[P,256,1,512]` BF16 | 每raw token一份latent，随历史增长 |
| MLA主cache FP8（可选） | `[P,256,1,528]` uint8 | 512个E4M3 + 4个group128 FP32 scale |
| IndexPool cache | 逻辑`[P,64,1,132]` uint8 | 每4token一条压缩index K |
| IndexPool tail | `[state_slots,2,4,128]` BF16 | raw K和gate ring；槽0保留padding |

注意IndexPool页内是**64×128 payload之后接64个FP32 scale**，不是64条逐记录132字节；不能只从逻辑shape推断layout。主cache与indexcache共享256-token block_table，候选buffer为2176列（2048+3向128对齐），无效位=-1；length包含候选前缀中的无效空洞。每query只看`floor((position+1)/4)`个完整pool，并追加当前tail，提前压缩chunk的未来pool不能泄漏给较早query。

`AttentionMetadata` 要有device block_table、slot_mapping、linear_state_indices；multi-token packed/chunked prefill还要q_cu_seq_lens。读槽和写槽可分开，position=0要清除复用tail。`create_cache()`是整层测试/调用接口，不说明engine已经按这个layout分配；接通时必须核对 `kv_cache_shape.cpp`、block manager和metadata producer。

内存量级：BF16主cache单DSA层每raw token1024 bytes，11层约11 KiB/token/持有完整cache的rank；1M token仅这项约11 GiB（按1048576计），另有index和workspace。34层KDA、global H=64时，FP32 recurrent约136 MiB/sequence/未分片模型，TP按head分片后约除以p；不包含conv/checkpoint副本。

## 8. Dense / MoE / 通用算子映射

| 模块 | 语义 | xLLM参考与MUSA接入 |
| --- | --- | --- |
| embedding | token→4096向量 | 公共`ParallelEmbedding`/`HiddenParallelEmbedding`或现有模型embedding模式；并行layout需与后续mHC一致 |
| dense FFN | gate/up 4096→12288，clamp+SiLU×up，down→4096 | Python`Glm5NextMLP`；MLU`DenseMLP`；MUSA Linear+activation可复用，FP8 loader及clamp待组合 |
| router | FP32 logits→sigmoid；correction bias只影响选expert；Top8 | `Glm5NextMoE._topk`；MUSA DeepSeek-V4 gate/topk原语需逐项验证 |
| routed experts | 288组SwiGLU，每token选8，inter=2048 | MUSA`DeepseekV4MoEImpl`、Qwen fused MoE可参考；不等于可原样替换 |
| shared expert | 每token都运行一个inter=2048的expert，再加到routed结果 | 本checkpoint未提供额外shared expert gate；不能带入其他模型的sigmoid shared gate |
| final norm/head | mHC均值→RMSNorm→vocab linear | 公共RMSNorm、ColumnParallelLinear、logits gather/采样 |
| sampling | temperature/top-p/greedy、停止条件 | 公共Sampler及MUSA random_sample/argmax等；与模型forward是分开的阶段 |

所有文本/视觉 SwiGLU 都要保留 `gate.clamp(max=10)` 与 `up.clamp(-10,10)`；普通SiluAndMul无此clamp不能直接替换。MoE权重取未加bias的sigmoid分数，对Top8归一化后乘2.5；n_group=1/topk_group=1不表示可以把sigmoid路由改成softmax。

FP8 MoE可评估现有MUSA `deepseek_v4_fp8_moe_quant_prepare`→grouped GEMM→clamped SwiGLU+quant→grouped GEMM→combine链路，或保留设备BF16计算作正确性基线；须验证router精度、expert布局/顺序、scale分片、输出权重和shared分支。没有在本次报告中认定该链路已支持GLM。muDNN专用算子/现有封装应优先复用；MATE grouped GEMM、muDNN groupwise GEMM等具体分支由已有FP8 wrapper条件决定，不能只根据`ldd`出现某库就推断某层调用它。

通用BF16 Linear目前是 `common/linear.cpp`→`ops_api.cpp::matmul`→`musa/activation.cu::matmul`：小形状可选native skinny GEMM，有输出buffer时走`mm_out/addmm_out`，一般走`torch::nn::functional::linear`。之后的底层库分发由TorchMUSA决定；本次没用profiler确认某一次GEMM到底命中muBLAS还是muDNN。FP8公共路径可参考`musa_fp8_linear.cpp`和`fp8_block_gemm.cpp`，不要把checkpoint存储精度、输入量化和计算精度混为一谈。

TP的基本边界：Q/K/V、q_b和FFN gate/up按输出维拆；o_proj和down_proj按输入维拆并归约；MLA latent和indexer在当前层实现中replicated。router/专家并行须单独设计；不能把`--ep_size`当成等价TP。首版应CP=DP=EP=1，先用普通TP容纳权重，再扩展其他并行方式。

## 9. 图片/视频进入文本主干前的分支

已有Python VLM类是 `Glm5NextVLModel(Glm5NextForCausalLM)`。C++ bridge在`is_vlm && model_type==glm5_next`时选内部registry key `glm5_next_vl`；这不是checkpoint的model_type，需要由bridge转换。

```text
URL/data URL/视频 → 媒体解码
  → resize、rescale、normalize、patchify → pixel_values + grid_thw
  → VisionPatchEmbed Conv3d [1024,3,2,14,14] + bias
  → 24 × VisionBlock：
       RMSNorm → QKV linear
       → Q/K head RMSNorm → 二维视觉RoPE
       → 图像/帧内部non-causal varlen attention → output linear → residual
       RMSNorm → clamped SwiGLU MLP → residual
  → post RMSNorm
  → Conv2d downsample [4096,1024,2,2] + bias
  → merger：Linear4096 → LayerNorm → GELU
            → clamped SwiGLU(4096→10240→4096)
  → image/video embeddings [Nvisual,4096]
  → scatter到placeholder对应的token embedding行
  → 文本45层主干
```

| 视觉步骤 | 现有实现 | MUSA现状/选型 |
| --- | --- | --- |
| 图片processor | `Glm5NextVLMultimodalProcessor`组合`GLM4VPromptProcessor`、`Qwen2VLImageProcessor` | Python模型路径委托`PyImagePreprocess`/HF AutoImageProcessor；MUSA注册待接 |
| 视频processor | `Glm4VVideoProcessor` | 已有采帧/normalize/patchify参考；仍需GLM服务验证 |
| patch conv/downsample | Python `nn.Conv3d/Conv2d` | MUSA native tower待实现，先评估muDNN/现有设备conv |
| norm、projection、RoPE | `VisionRMSNorm/ColumnParallelLinear/RowParallelLinear` | 可参考MUSA Qwen3.5 vision primitives，核对bias/shape/RoPE |
| vision attention | Python`VisionAttention`按cu_seqlens分段调用`F.scaled_dot_product_attention`，non-causal | 尚未映射成GLM/MUSA kernel；先查MATE匹配varlen、D=64、non-causal接口，再评估muDNN；没有验证不能声称命中MATE |
| merger | Python`VisionPatchMerger` | native层待做，注意10240中间维、GELU和LayerNorm bias |
| embedding合并 | `encode_multimodal` / `prepare_multimodal_inputs` | 按placeholder mask scatter；媒体token数需与编码输出严格一致 |

本地processor配置：image token预算16–8000；video为16–240000、fps=2；mean=[0.48145466,0.4578275,0.40821073]，std=[0.26862954,0.26130258,0.27577711]。输出token数与`grid_thw.prod()/4`匹配，不能用固定448×448代替所有输入的动态grid。图片和视频token分别154854/154855，起止标记也不同。

特别核实：Python类的patch embed注释写“no bias”，**实际`nn.Conv3d`默认带bias且loader加载bias**；真实checkpoint确有[1024] bias，vLLM参考也明确`bias=True`。实现时应遵循权重和执行代码。

当前安装Transformers是4.50.2，checkpoint声明5.16.0且processor类型为`Glm5NextImageProcessor/Glm5NextVideoProcessor`。如果沿用HF预处理，需要验证安装版是否确有相应注册/补丁；不能只传`trust_remote_code`就认为可用，模型目录没有配套remote modeling/processor Python源码。纯C++预处理可避开这项依赖，但要逐项对齐像素和grid结果。

## 10. 从现状到可服务的最小接入顺序

1. **补模型与args注册。** 新建/接入MUSA GLM native model、decoder，在`models/models.h`的MUSA分支引入；复用`glm5_next_args::load_model_args`，核对nested text/vision配置、dtype规范化、FP8 quant config和构建target。不要只改`model_type`成现有模型。
2. **补KDA整层。** 复用MLU层边界、Python/vLLM语义；把投影、conv、flash_kda、sigmoid norm、o_proj串起来。实现请求状态gather/scatter和reset，先不做MTP/graph/prefix复用。
3. **组合DSA和mHC。** decoder选择现有`Glm5NextSparseAttention`；适配engine cache页256和index/tail布局；验证MUSA mHC在4096 hidden上的pre/post和最后mean。
4. **接FP8 dense/MoE。** 读取weight_scale_inv，保持routing FP32、Top8、clamp10、scaling2.5、shared分支；核对多卡分片，控制加载峰值。
5. **先通文本eager。** CP=DP=EP=1、禁graph/prefix/MTP/mix/overlap；用足够普通TP承载权重。对照参考逐层hidden/logits和greedy输出，完成后再长上下文及并发。
6. **再接VLM。** 注册processor和native VLM，补vision tower与embedding scatter；先图片，后视频/混合多图；对齐预处理、视觉features及最终生成。
7. **最后扩展性能特性。** 专用decode、kernel融合、graph、prefix cache、MTP、CP/EP/PD分别验证。当前sparse层显式拒绝capture，所以直接开`enable_graph`不成立。

若选择Python模型路线，需要新增MUSA Python kernel包、attention backend、execute_linear/execute_mla及绑定，修factory gates和FP8 loader；其工作范围不限于注册一个模型名。当前已有native KDA/sparse层更适合优先组合native路径，此处属于基于现状的工程建议。

## 11. 启动与调用模板：接入完成后的验收合同

**下面启动命令是未来接入后的最小参数模板，不是本次已经跑通的命令。** 设备数量应依据实际显存、TP加载策略和拓扑确定；不在此虚构已可运行的卡数/吞吐。

```bash
XLLM_BIN=build/lib.linux-x86_64-cpython-310/xllm/xllm
MODEL_PATH=/data/models/GLM-5.3-Flash

# 前置：已生成所需KDA、sparse MLA、Indexer模块到同一部署cache。
export FLASHINFER_OPS_PATH=/absolute/path/to/validated/cached_ops

# 完成native VLM注册后；先设较小请求并发，普通TP的设备/进程拓扑另行配置。
"$XLLM_BIN" \
  --model="$MODEL_PATH" --model_id=glm5_3_flash \
  --backend=vlm --model_impl=native --port=18994 \
  --communication_backend=mccl \
  --dp_size=1 --ep_size=1 --cp_size=1 --kv_split_size=1 \
  --block_size=256 \
  --enable_graph=false --enable_prefix_cache=false \
  --enable_schedule_overlap=false --enable_mix_batch=false \
  --max_seqs_per_batch=1
```

先完成文本native时可验证`backend=llm`；VLM配置的backend解析与args loader应作为接入的一部分测试。LLM/VLM并行拓扑沿现有worker/进程配置，**不要套用只供DiT使用的`--tp_size`来配置LLM TP**。chunked-prefill开关及chunk长度应在状态衔接测试通过后启用；关闭chunking不代表能容纳任意长prefill。

```bash
curl http://127.0.0.1:18994/v1/models
curl http://127.0.0.1:18994/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm5_3_flash","messages":[{"role":"user","content":"请用一句话介绍你自己。"}],"temperature":0,"max_tokens":64,"stream":false}'
```

多模态请求body（替换可读取的image URL或data URL）：

```json
{
  "model": "glm5_3_flash",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "描述这张图片。"},
      {"type": "image_url", "image_url": {"url": "<IMAGE_URL_OR_DATA_URL>"}}
    ]
  }],
  "temperature": 0,
  "max_tokens": 64,
  "stream": false
}
```

验收至少覆盖：真实权重短文本prefill+多步decode；第二条请求复用状态槽；packed多请求；chunked与完整prefill对照；跨256页；超过2048token触发真实Indexer评分/Top-K；图片feature/token数；多图与视频；不同TP结果。HTTP 200和可读字符串不足以证明FP8 scale、mHC或state正确，应比较reference logits/greedy序列并检查设备活动。MTP可单独启用，现有层45只证明checkpoint提供draft权重，不表示调度/accept/reject/state rollback已经接好。

## 12. 本次证据、既有测试与未验证项

本次完成：读取配置/模板与全部62个safetensors文件头；核对注册、layer调用方、kernel、CMake、runtime和其他backend；确认当前CMake cache为`USE_MUSA=ON, USE_CUDA=OFF, USE_NPU=OFF`。已有`xllm`、`kda_test`、`sparse_mla_test`、`glm5_next_sparse_attention_test`产物，但二进制时间不同，本次未重新构建，不能视为当前提交全量重编译证据。

环境读取结果：Torch/TorchMUSA 2.9.1；MATE 0.2.8.post1；flash_kda 0.2.8.post1+musa；Transformers 4.50.2；存在`/dev/mtgpu.0`–`.7`。`ldd xllm`解析到muDNN 3.4.0、MCCL 2.3.0、TVM-FFI、TorchMUSA及MUSA runtime，输出未见缺失依赖。这只证明动态依赖可解析，不证明GLM符号/产物已经链接到服务。

本地vLLM参考commit为`ebac032b92cc55c2232deced9217cd5122648ce8`，MATE checkout为`9d3260d633597f08fc9d5d4dc73344872241c34d`；安装MATE与checkout不是同一概念，FFI ABI和模板以本机安装/导出版本为准。本次未重新导出kernel，具体ABI对照和历史验证见下列已有报告。

| 已有记录（非本次重跑） | 记录结果 | 边界 |
| --- | --- | --- |
| [musa_kda.md](musa_kda.md) | 2026-11-05 KDA 3/3；数值、graph replay、默认stream | 算子级，不含模型层/服务 |
| [musa_sparse_attention_integration.md](musa_sparse_attention_integration.md) | sparse MLA 3/3，整层8/8，两卡MCCL各1/1 | 合成权重/有限尺寸，不是下载checkpoint的输出精度 |
| [musa_sparse_attention.md](musa_sparse_attention.md) | MATE能力和ABI调研 | 早期方案需以当前整层代码及integration报告修正为准 |

复查已有组件的命令（运行前须配置匹配的模块目录与构建环境）：

```bash
python tools/musa_glm5_3_vl_bench/compile_musa_kda.py --ops-path /path/to/cached_ops --jobs 2
python tools/musa_glm5_3_vl_bench/compile_musa_sparse_mla.py --ops-path /path/to/cached_ops
cmake --build build/cmake.linux-x86_64-cpython-310 \
  --target kda_test sparse_mla_test glm5_next_sparse_attention_test -j 4
MATE_DISABLE_JIT=1 build/lib.linux-x86_64-cpython-310/xllm/kda_test
python tools/musa_glm5_3_vl_bench/validate_musa_sparse_attention.py
```

本次没有执行这些构建/数值测试，因为交付是静态调用链调研，未修改模型或kernel。尚未验证的核心范围：真实FP8整模型加载、GLM mHC/KDA decoder组合、多卡全模型推理、视觉encoder、HF processor可用性、长context、graph/prefix/MTP/PD、吞吐及长期压力。报告没有给出已跑通服务的结论。

## 13. 源码阅读导航

仓库内链接相对本报告目录；类/函数名可直接`rg -n`定位。

| 调用层级 | 主要源码与符号 |
| --- | --- |
| main与模型注册 | [xllm.cpp](../../xllm/xllm.cpp)、[models.h](../../xllm/models/models.h)、[model_registry.cpp](../../xllm/models/model_registry.cpp) |
| HF加载 | [hf_model_loader.cpp](../../xllm/core/framework/hf_model_loader.cpp)：`load_model_args`；[glm5_next.h](../../xllm/models/llm/glm5_next.h)：nested config loader |
| HTTP/请求 | [chat_service_impl.cpp](../../xllm/api_service/chat_service_impl.cpp)：`MMChatServiceImpl`；[vlm_request_factory.cpp](../../xllm/core/framework/request/vlm_request_factory.cpp) |
| master/scheduler | [vlm_master.cpp](../../xllm/core/distributed_runtime/vlm_master.cpp)、[continuous_scheduler.cpp](../../xllm/core/scheduler/continuous_scheduler.cpp) |
| engine/worker | [vlm_engine.cpp](../../xllm/core/distributed_runtime/vlm_engine.cpp)、[vlm_worker_impl.cpp](../../xllm/core/runtime/vlm_worker_impl.cpp) |
| native executor | [vlm_executor_impl.cpp](../../xllm/core/runtime/vlm_executor_impl.cpp)、[base_executor_impl.cpp](../../xllm/core/runtime/base_executor_impl.cpp) |
| Python bridge/registry | [py_causal_lm.cpp](../../xllm/models/llm/py_causal_lm.cpp)、[registry.py](../../xllm/python/registry.py)、[runtime初始化](../../xllm/python/__init__.py) |
| Python GLM text | [glm5_next.py](../../xllm/python/models/glm5_next.py)：KdaAttention/MlaAttention/Indexer/MLP/MoE/HyperConnection/DecoderLayer/Model |
| Python GLM vision | [glm5_next_vl.py](../../xllm/python/models/glm5_next_vl.py)；[processor注册](../../xllm/models/vlm/glm5_next_vlm.h) |
| MLU native参考 | [model](../../xllm/models/llm/mlu/glm5_next.h)、[decoder](../../xllm/core/layers/mlu/glm5_next/glm5_next_decoder_layer.cpp)、[KDA](../../xllm/core/layers/mlu/glm5_next/glm5_next_kda.cpp)、[indexer](../../xllm/core/layers/mlu/glm5_next/glm5_next_kpool_indexer.cpp) |
| MUSA GLM整层 | [sparse attention](../../xllm/core/layers/musa/glm5_next_sparse_attention.cpp)及同名`.h` |
| MUSA GLM kernel | [KDA](../../xllm/core/kernels/musa/kda.cpp)、[IndexPool](../../xllm/core/kernels/musa/glm5_next_kpool.cu)、[sparse MLA](../../xllm/core/kernels/musa/sparse_mla.cpp)、[cache](../../xllm/core/kernels/musa/sparse_mla_cache.cu) |
| 公共layer/kernel分发 | [linear.cpp](../../xllm/core/layers/common/linear.cpp)、[rms_norm.cpp](../../xllm/core/layers/common/rms_norm.cpp)、[ops_api.cpp](../../xllm/core/kernels/ops_api.cpp)、[activation.cu](../../xllm/core/kernels/musa/activation.cu) |
| MUSA可复用模块 | [DSV4 decoder](../../xllm/core/layers/musa/deepseek_v4/deepseek_v4_decoder_layer.cpp)、[MoE](../../xllm/core/layers/musa/deepseek_v4/deepseek_v4_moe.cpp)、[FP8 linear](../../xllm/core/layers/musa/musa_fp8_linear.cpp) |
| build/state契约 | [MUSA layers CMake](../../xllm/core/layers/musa/CMakeLists.txt)、[kernel CMake](../../xllm/core/kernels/musa/CMakeLists.txt)、[kv_cache_shape.cpp](../../xllm/core/framework/kv_cache/kv_cache_shape.cpp) |

外部checkout参考（本机绝对位置）：

- `/data/albert/Projects/vllm/vllm/models/glm5next/common/model.py`：`Glm5NextDecoderLayer/Glm5NextMLP/Glm5NextMoE/Glm5NextModel`，核对层顺序、mHC、clamp和router。
- 同目录`kda.py::Glm5NextLinearAttention`：融合投影次序、raw beta/forget gate、state与输出gate。
- 同目录`attention.py::Glm5NextMLAAttention/Indexer`及`nvidia/ops/kpool_compress.py`：NoPE、IndexPool、tail与MLA边界；当前MUSA层使用独立C++/MATE实现，没有调用vLLM Python执行路径。
- 同目录`multimodal.py`：`Glm5NextVisionTransformer/Glm5NextVisionPatchEmbed/Glm5NextPatchMerger`，核对vision bias、clamp和merger。
- `/data/albert/Projects/mate/`：提供attention/KDA实现与测试参考；实际离线生成器导入本机安装MATE，升级后需重新核对导出模块名、workspace、stream与架构。
