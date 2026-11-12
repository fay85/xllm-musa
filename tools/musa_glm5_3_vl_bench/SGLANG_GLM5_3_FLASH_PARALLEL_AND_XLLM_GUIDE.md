# SGLang GLM-5.3-Flash (PD 分离) 并行架构深度解析与 xLLM-MUSA 接入指南

- **分析日期**：2026-11-12
- **参考容器**：`albert_sglang`（镜像：`registry.mthreads.com/presale/inference/sglang:v0.5.17-glm5.3flash-amd64-ph1-musarc4.3.8-beta`，容器内源码路径 `/workspace/sglang`）
- **参考启动脚本**：
  - Prefill：`/data/albert/Projects/scripts/LLM_inference/GLM-5.3-Flash/sglang/start_prefill_service.sh`
  - Decode：`/data/albert/Projects/scripts/LLM_inference/GLM-5.3-Flash/sglang/start_decode_service.sh`
  - Router：`/data/albert/Projects/scripts/LLM_inference/GLM-5.3-Flash/sglang/start_router_service.sh`
- **模型配置**：`/data/nfs_shared/models/GLM-5.3-Flash/config.json`（架构：`Glm5NextForConditionalGeneration`，45 层主干 + 第 45 层 EAGLE/NextN MTP Draft + 24 层 ViT 视觉编码器，总参数 320B / 激活参数 18B）

---

## 0. 模型结构背景：为什么 GLM-5.3-Flash 的并行设计极其特殊？

在分析 `prefill` 和 `decode` 的并行策略前，必须先明确 `GLM-5.3-Flash`（代码中对应 `glm5_next`）的四个核心异构特征：

1. **混合注意力机制（34 层 KDA + 11 层 DSA）**：
   - 45 层主干采用 `LLL D` × 11 + `L` 的交错排布（DSA 位于层 `3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43`，其余 34 层为 KDA）。
   - **KDA（Kimi Delta Attention，`Glm5NextLinearAttention`）**：`num_heads=64, head_dim=128`，包含通道级 `short_conv_kernel_size=4` 的 `causal_conv1d`、低秩门控 `f_a/f_b` 与 `g_a/g_b`、以及状态形状为 `[64, 128, 128]` 的线性递归状态（Mamba-like Recurrent State）。
   - **DSA（DeepSeek Sparse Attention MLA，`DeepseekV2AttentionMLA`）**：`num_attention_heads=64, q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=256, v_head_dim=256, qk_rope_head_dim=0, mla_use_nope=true`（**纯 NoPE MLA**）；挂载 **KPool Indexer**（`index_n_heads=32, index_head_dim=128, index_topk=2048, index_kpool=4, index_kpool_compress=true, index_kpool_always_select_tail=true`），且每相邻两层 DSA 共享一次 Indexer TopK 选择结果（`index_share_for_mtp_iteration=true`，仅层 `3, 11, 19, 27, 35, 43` 含实体 Indexer）。
2. **混合 FFN 结构（3 层 Dense MLP + 42 层高稀疏度 MoE）**：
   - 层 `0, 1, 2` 为 **Dense MLP**（`intermediate_size=12288, swiglu_limit=10.0`）。
   - 层 `3..44`（共 42 层）为 **Sparse MoE**（`n_routed_experts=288, num_experts_per_tok=8, n_shared_experts=1, moe_intermediate_size=2048, scoring_func="sigmoid", routed_scaling_factor=2.5`）。注意单个专家的中间维只有 `2048`！
3. **4 路残差超连接（mHC, Multi-stream Hyper-Connections）**：
   - `mhc=true, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6`。跨层残差流不是 `[T, 4096]`，而是 **4 路宽残差 `[T, 4, 4096]`（展平后为 `16384` 维，是普通模型的 4 倍）**；仅在进入 Attention / FFN 前通过 `MHCPre`（含 20 次 Sinkhorn-Knopp 双随机矩阵迭代与加权收缩）坍缩回 `[T, 4096]`，出子层后再由 `MHCPost` 写回 4 路残差。
4. **混合精度（FP8 Block-wise + BF16/FP32）**：
   - 仅 DSA 主投影与 MoE 专家权重使用 `128×128` 块级 `FP8 E4M3` 量化；KDA 全层、mHC、前 3 层 Dense MLP、MoE 路由门控、ViT 以及第 45 层 MTP Draft 保持 `BF16/FP32`。

---

## 1. Prefill 是如何并行的？为什么这么选择？`tp8 ep8 cp8` 是怎么分的？

### 1.1 Prefill 启动脚本的核心并行参数

在 `start_prefill_service.sh` 中，单机 8 卡的并行与通信配置如下：

```bash
# 环境变量
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_EMBED_REPLICATION=1
export SGLANG_SBO_COMBINE_SHARED=1
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_MOONCAKE_SEND_AUX_TCP=1

# 启动参数
--tp-size 8
--ep-size 8
--dp-size 1
--pp-size 1
--attn-cp-size 8
--enable-prefill-cp
--cp-strategy interleave
--enable-dp-lm-head
--moe-dense-tp-size 1
--moe-a2a-backend deepep
--deepep-mode normal
--attention-backend dsa
--dsa-prefill-backend tilelang
--linear-attn-backend triton
--linear-attn-prefill-backend flashkda
```

### 1.2 SGLang 中 `tp8 ep8 cp8` 的真实含义与通信组推导

在 SGLang 的并行体系（`/workspace/sglang/python/sglang/srt/distributed/parallel_state.py:2418-2780` 与 `layers/dp_attention.py:140-260`）中，**`--tp-size` 代表的是每个 PP stage 的总 GPU 数（World Size）**，然后在这 8 张卡上独立正交地切分 **Attention 网格**与 **MoE/FFN 网格**：

1. **全局 TP 组（`tp_group`）**：`tp_size = 8`（覆盖 GPU `0..7`）。
2. **Attention 通信网格**：
   $$\text{tp\_size} = \text{attn\_dp\_size} \times \text{attn\_cp\_size} \times \text{attn\_tp\_size}$$
   - 因为 `--dp-size 1` 且未开 `--enable-dp-attention`，所以 `attn_dp_size = 1`。
   - 因为 `--attn-cp-size 8`，所以 **`attn_cp_size = 8`**（`attn_cp_group` 覆盖 GPU `0..7`）。
   - 由此推导出：**`attn_tp_size = 8 // 1 // 8 = 1`**（`attn_tp_group` 大小为 1，即单卡自身！）。
3. **MoE 通信网格**：
   $$\text{tp\_size} = \text{moe\_dp\_size} \times \text{moe\_ep\_size} \times \text{moe\_tp\_size}$$
   - 因为 `--ep-size 8`，所以 **`moe_ep_size = 8`**（`moe_ep_group` 覆盖 GPU `0..7`）。
   - `moe_dp_size = 1`。
   - 由此推导出：**`moe_tp_size = 8 // 8 // 1 = 1`**（每个专家内部不再做张量切分，保持完整的 `moe_intermediate_size = 2048`）。
4. **Dense MLP（层 `0..2`）通信网格**：
   - 因为显式传入了 `--moe-dense-tp-size 1`（触发 `enable_moe_dense_fully_dp() == True`），前 3 层 Dense MLP 的 TP 大小被强制设为 **`mlp_tp_size = 1`**（权重在 8 卡上复制，激活值按 Token 维度切分并行计算）。

### 1.3 `tp8 ep8 cp8` 在模型各模块的具体切分方式（Hybrid CP 架构）

SGLang 为 `glm5_next` 专门设计了 `MHCHybridDSACPLayerCommunicator`（见 `layers/communicator_mhc_hybrid_cp.py` 与 `models/glm5_next.py:1060-1315`）。整个模型以 **Plain Token 切分（每张卡持有连续的 `S/8` 个 Token）** 作为跨层基准契约，但在不同算子内部采用截然不同的切分：

| 模块 / 层 | 权重切分方式 | 输入/输出 Token 切分 | 算子内通信原语 | 源码位置（`/workspace/sglang/python/sglang/srt/`） |
| :--- | :--- | :--- | :--- | :--- |
| **Embedding** | 完整复制（`SGLANG_ENABLE_EMBED_REPLICATION=1`，`tp_size=1`） | 输入 `[S]` 经 `cp_plain_split` 切为 `[S/8]`，直接查表输出 `[S/8, 4096]`，扩展为 4 路残差 `[S/8, 4, 4096]` | **0 通信**（无词表 AllReduce） | `models/glm5_next.py:1244-1262`, `layers/vocab_parallel_embedding.py:390` |
| **mHC (`MHCPre` / `MHCPost`) + RMSNorm** | 完整复制（仅 `[4096, 36]` 极小参数） | 每张卡仅处理本地 `[S/8, 4, 4096]` 残差与 `[S/8, 4096]` 归一化 | **0 通信**（吞吐随 8 卡线性扩展） | `models/glm5_next.py:1098-1170` |
| **KDA 线性注意力（34 层）** | **按 Head 8 路切分（Head-TP over CP）**：`q/k/v/b`、1D 卷积、`A_log/dt_bias` 按 64 heads 切为 **8 heads/卡**；低秩瓶颈 `f_a(128)`、`g_a(128)` 复制，`f_b`、`g_b` 按输出切分；`o_proj` 按行切分 | 1. 入口：`[S/8, 4096]` $\xrightarrow{\text{AllGather}}$ `[S, 4096]`（严格自然顺序）<br>2. 算子：每卡对完整序列 `S` 计算本地 8 个 Head 的 `causal_conv1d` + `flashkda` 状态扫描<br>3. 出口：`o_proj` 输出 `[S, 4096]` 部分和 $\xrightarrow{\text{ReduceScatter}}$ `[S/8, 4096]` | **1 次 AllGather + 1 次 ReduceScatter**（基于 `attn_cp_group`） | `models/glm5_next.py:362-575, 811-854` (`cp_plain_all_gather` / `cp_plain_reduce_scatter`) |
| **DSA 稀疏 MLA 注意力（11 层）** | **权重不切分（`attn_tp_size=1`）**：每张卡持有全部 64 个 Head 的 `q_a/q_b/kv_a/kv_b/o_proj` 及全部 32 个 Head 的 Indexer 权重 | 1. 入口：`[S/8, 4096]`（Plain 连续块）$\xrightarrow{\text{AllToAll}}$ `[S/8, 4096]`（**Interleave 轮询交错块**）<br>2. KV 与 Indexer：各卡为本地 `S/8` Token 生成 `kv_a(512)` 与 KPool，通过 `AllGather` 重建完整序列的 KV Cache 与 Indexer KPool Cache<br>3. 核心计算：每卡用本地 `S/8` 个 Query（全 64 头）检索并 Attend 全序列 KV<br>4. 出口：`[S/8, 4096]`（Interleave）$\xrightarrow{\text{AllToAll}}$ `[S/8, 4096]`（Plain 连续块） | **2 次 AllToAll**（Plain $\leftrightarrow$ Interleave）+ **KV/KPool AllGather**（仅传输压缩后的 512 维 latent 与 KPool） | `models/glm5_next.py:1138-1152`, `models/deepseek_v2.py:2174-2437`, `layers/attention/dsa/utils.py` |
| **Dense MLP（前 3 层 `0..2`）** | **权重完整复制**（`--moe-dense-tp-size 1`，`mlp_tp_size=1`） | 直接在本地 Plain `[S/8, 4096]` Token 分片上独立完成 Gate/Up/Down GEMM | **0 通信**（无 TP AllReduce） | `models/glm5_next.py:1007-1017`, `layers/communicator.py:553` |
| **Sparse MoE（后 42 层 `3..44`）** | **Routed Experts `EP=8, moe_tp=1`**：288 个专家均分为 **36 个专家/卡**，每个专家保持完整 `2048` 维；**Shared Expert 复制**（`_shared_expert_use_tp1=True`） | 本地 `[S/8, 4096]` Token 经 Router 选出 Top-8 专家，通过 **DeepEP Normal 模式** 做跨 8 卡 All-to-All Dispatch $\rightarrow$ Grouped GEMM $\rightarrow$ All-to-All Combine（与 Shared Expert 融合累加），输出仍为 `[S/8, 4096]` | **DeepEP Dispatch + Combine AllToAll**（无冗余 AllGather/ReduceScatter） | `models/glm5_next.py:1019-1035, 1173-1226`, `models/deepseek_v2.py:678-735` |
| **Final Norm + LM Head** | `--enable-dp-lm-head` 使 `ParallelLMHead` 绑定 `attn_tp_group`（大小 1） | 先在 `[S/8, 4, 4096]` 上做 `mhc_contract_mean` 和 `RMSNorm`；随后 ` MHCHybridDSACPLayerCommunicator` 收集各 CP 分片到 `[S, 4096]` 提取每个请求最后一个 Token 计算 Logits | **仅在网络末尾 Gather 一次** | `models/glm5_next.py:1292-1315`, `layers/communicator_mhc_hybrid_cp.py:118-138` |

### 1.4 为什么 Prefill 要这么设计？（五大核心原因）

1. **KDA 线性递归与因果卷积的物理约束 vs DSA 因果三角负载均衡的矛盾**：
   - KDA 包含沿序列维度的 `causal_conv1d(kernel=4)` 和块间状态递推 `flashkda`。如果对 KDA 按 Token 切分（尤其是 Interleave 交错切分），相邻 Token 不在同一张卡上，卷积和状态扫描根本无法执行；而 KDA 的 64 个 Head 之间完全独立。因此 **KDA 必须在 CP 组内转为 Head 切分（8 heads/卡）+ 全序列 `S` 计算**。
   - 同时，用 `AllGather + ReduceScatter` 代替传统 TP 的 `AllReduce`，不仅完美衔接了层间 `[S/8, H]` 的 Token 分片契约，而且在 Ring 算法下 **比 AllReduce 节省约 33% 的通信量**。
2. **DSA 层的因果掩码（Causal Mask）负载均衡与 KV 压缩优势**：
   - 在长序列 Prefill 中，如果按连续块切分 Query（卡 0 拿前 `S/8`，卡 7 拿最后 `S/8`），由于 Causal Mask 下三角特性，卡 7 的有效 KV 长度和 Indexer 搜索范围是卡 0 的近 8 倍，会造成严重的木桶效应。通过 `--cp-strategy interleave`（`cp_plain_to_scattered` 的 `all_to_all_single`），将 Token 按模 8 轮询打散，**8 张卡的 Causal Attention 与 Indexer 计算负载完全均衡**。
   - 同时，DSA 令 `attn_tp_size=1`（不切 Head），每张卡只对 `S/8` 个 Token 算 `q_b_proj (1536 -> 64*256)` 和 `kv_b_proj`，计算量天然除以 8；跨卡只需 AllGather 极小的压缩 `kv_lora_rank=512`（FP8）和 KPool 压缩槽位，通信开销远小于传统 TP8 的 `o_proj` 全序列 AllReduce。
3. **彻底消除 mHC、Dense MLP 和 MoE 入口的冗余通信**：
   - 因为层间残差始终保持 `[S/8, 4, 4096]` 的 Token 分片状态，**全部 90 个 `MHCPre/MHCPost` 算子、前 3 层 Dense MLP、以及全部 42 层 MoE 的 Gate 路由与 Shared Expert 均只处理 `1/8` 的 Token，且全程 0 通信**！
   - 进入 MoE 时，每张卡的 `[S/8, 4096]` 正好作为 DeepEP All-to-All Dispatch 的天然局部输入，无需像传统 TP8 那样先做 ReduceScatter 或复制计算。
4. **避免 MoE 专家窄维度劣化（`moe_tp_size=1, moe_ep_size=8`）**：
   - GLM-5.3-Flash 的 `moe_intermediate_size` 只有 `2048`。如果像纯 TP8 那样把每个专家切 8 份，单卡维度只剩 `256`（FP8 Grouped GEMM 的 Tile 利用率极低，且每张卡要遍历全部 288 个专家的碎片）。采用 `EP=8, moe_tp=1` 后，每张卡只负责 `288 / 8 = 36` 个完整专家（维度 `2048`），单个专家 Token 密度提升 8 倍，算力利用率大幅跃升。
5. **赋能 PD 分离的 8 卡并行 RDMA 状态传输（`SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1`）**：
   - 因为 Prefill 结束时，KDA 的 64 个 Head 状态天然按 `8 heads/卡` 分布在 8 个 CP Rank 上，开启该宏后，8 张 Prefill GPU 可以同时向 Decode 节点发起 Mooncake RDMA 写入（见 `disaggregation/prefill.py:391-429` 的 `_send_mamba_state_slice`），将状态传输带宽打满 8 张网卡。

---

## 2. Decode 是如何并行的？为什么？`tp`、`ep`、`dp` 同时设为 16 是怎么分的？

### 2.1 Decode 启动脚本的核心并行参数

在 `start_decode_service.sh` 中，采用 **双机 16 卡（`NNODES=2`，每机 8 卡，400Gb RoCEv2 互联）** 部署，核心参数如下：

```bash
# 拓扑与并行参数
NNODES=2
TP_SIZE=16
EP_SIZE=16
DP_SIZE=16
--tp-size 16
--ep-size 16
--dp-size 16
--pp-size 1
--enable-dp-attention
--enable-dp-lm-head
--moe-dense-tp-size 1
--moe-a2a-backend deepep
--deepep-mode low_latency
--max-running-requests 1024        # 16 个 DP Worker 总并发 = 1024 -> 每个 DP Rank 最大并发 64
--cuda-graph-max-bs-decode 8       # 每个 DP Rank 的 CUDA Graph 最大批大小 = 8
--speculative-algorithm EAGLE      # 5 步 TopK-1 Draft，每次 Verify 前向每个请求携带 6 个 Token
```

### 2.2 `tp16 ep16 dp16` 同时开启时的数学切分关系

刚看到 `--tp-size 16 --ep-size 16 --dp-size 16` 时容易产生误解：只有 16 张卡，为什么 TP、EP、DP 都是 16？
结合 SGLang 的 `compute_dp_attention_world_info`（`layers/dp_attention.py:140-185`）与 `initialize_moe_ parallel`（`distributed/parallel_state.py:2495-2610`），16 张卡在两套网格中的精确切分如下：

1. **总通信域**：`tp_size = 16` 代表整个 Decode 实例横跨双机共 16 张 GPU。
2. **Attention / KDA / Dense MLP / mHC / LM Head 网格（由 `--enable-dp-attention` 控制）**：
   - `attn_dp_size = dp_size = 16`（共 16 个完全独立的 DP Rank，编号 `0..15`）。
   - `attn_cp_size = 1`（Decode 不使用 CP）。
   - **`attn_tp_size = tp_size // attn_dp_size // attn_cp_size = 16 // 16 // 1 = 1`！**
   - **`mlp_tp_size = 1`**（由 `--moe-dense-tp-size 1` 决定）。
   - **含义**：在除 Sparse MoE 之外的所有层（Embedding、34 层 KDA、11 层 DSA、90 个 mHC、前 3 层 Dense MLP、Final Norm、LM Head、以及第 45 层 EAGLE Draft 的注意力部分），**每张 GPU 都是一个 `TP=1` 的独立数据并行（DP）实例**！每张卡独立持有一份完整的非 MoE 权重，独立维护落在自己卡上的请求（最多 `1024 / 16 = 64` 个请求）的 KDA 状态池和 DSA FP8 KV Cache 池。
3. **Sparse MoE 专家网格（由 `--ep-size 16` 控制）**：
   - `moe_ep_size = ep_size = 16`（覆盖双机全部 16 张卡）。
   - `moe_dp_size = 1`。
   - **`moe_tp_size = tp_size // moe_ep_size // moe_dp_size = 16 // 16 // 1 = 1`！**
   - **含义**：后 42 层的 288 个 Routed Experts 被均匀切分到双机 16 张卡上，**每张 GPU 仅驻留 `288 / 16 = 18` 个完整专家**（每个专家 `moe_intermediate_size = 2048` 不做 TP 切分）；1 个 Shared Expert 在每张卡上本地复制（`tp1`）。

### 2.3 Decode 单步前向的数据流（Attention DP16 $\leftrightarrow$ MoE EP16）

在 Decode 阶段（配合 EAGLE 投机采样，每个请求每次前向输入 $1 + 5 = 6$ 个验证 Token，单卡 CUDA Graph `bs=8` 即最多 $8 \times 6 = 48$ 个 Token）：

1. **Embedding $\rightarrow$ mHC $\rightarrow$ KDA (34 层) / DSA (11 层) $\rightarrow$ mHC**：
   - 16 张 GPU 各自处理自己 DP 分片内的请求 Token（例如每卡 `[B_local * 6, 4096]`）。
   - KDA 层（`head_shard_size = attn_tp_size = 1`）在单卡内跑完 64 个 Head 的 `causal_conv1d_update` 与 `fused_kda_gate` 状态更新，**0 跨卡通信**。
   - DSA 层（`tp_size = attn_tp_size = 1`）在单卡内跑完 64 个 Head 的 MLA 与 TileLang Indexer TopK 稀疏注意力，**0 跨卡通信**。
2. **前 3 层 Dense MLP (`0..2`)**：
   - 因为 `moe_dense_tp_size = 1`，每张卡直接在本地 `[B_local * 6, 4096]` 上算完 `Gate/Up/Down`，**0 跨卡通信**。
3. **后 42 层 Sparse MoE (`3..44`)**：
   - 每张卡本地完成 `RoutedMoeGate`（选出每个 Token 的 Top-8 专家 ID，范围 `0..287`）。
   - 并发启动/融合本地 `SharedExpert`（`SGLANG_SBO_COMBINE_SHARED=1`）。
   - 调用 **DeepEP `low_latency` 模式**（基于 IBGDA / RoCE RDMA GPUDirect Async 的纯 GPU 内核级 All-to-All）：将 16 张 DP 卡上的 Token 按目标专家 ID 路由到对应的 EP Rank（每卡负责 18 个专家），执行本地 FP8 Grouped GEMM 后，再通过 DeepEP `low_latency` Combine 将结果直接散射回原 DP 卡。
4. **LM Head & EAGLE Draft（第 45 层）**：
   - 回到各 DP 卡本地独立完成 Logits 计算、投机验证（Rejection Sampling）与下一轮 5 步 EAGLE Draft 生成。

### 2.4 为什么 Decode 要选择 `tp16 ep16 dp16`？（三大根本原因）

1. **彻底消灭 Attention / KDA 的 TP AllReduce 延迟墙**：
   - Decode 是典型的 **Memory-Bandwidth Bound + Collective Latency Bound** 阶段。如果采用传统 TP8/TP16，45 层主干 + 1 层 EAGLE 意味着每步 Decode 要做 **92 次跨卡（甚至跨机）AllReduce**！（在 `xllm-musa` 的 TP8 基线剖析中，仅单机 TP8 的 `all_reduce` 就占了 Decode 总耗时 TPOT 的 **32.2%**）。
   - 采用 `--enable-dp-attention`（`attn_tp=1, attn_dp=16`）后，34 层 KDA、11 层 DSA、前 3 层 Dense MLP 的 **跨卡通信次数降为 0**！唯一需要跨卡通信的只有 42 层 MoE 的 DeepEP Low-Latency All-to-All。
2. **消除 MLA KV Cache 与 KDA 状态的跨卡冗余复制，最大化总并发容量**：
   - GLM-5.3-Flash 的 DSA 层采用 `kv_lora_rank=512` 的压缩潛变量 KV Cache 和 128 维 KPool Index Cache。在传统 TP 下，`kv_lora_rank` 无法按 Head 切分，只能在所有 TP 卡上**完整复制**，导致 8 卡或 16 卡的 KV Cache 容量等于单卡容量。
   - 在 `DP=16`（`attn_tp=1`）下，16 张卡各自存储不同请求的 KV Cache 和 KDA 状态，**集群总 KV / 状态容量直接放大 16 倍**，轻松支撑 `max_running_requests = 1024`。
3. **将 320B MoE 权重的 HBM 带宽压力分摊到 16 张卡（`EP=16`）**：
   - Decode 阶段最大的访存瓶颈是读取 42 层 × 288 个专家的 FP8 权重（总计约 300GB）。
   - 非 MoE 权重（34 层 KDA + 11 层 DSA + 3 层 Dense MLP + Embedding/LM Head）总计仅约 **16~18 GB**，完全可以在每张卡上复制一份；而将 300GB 的 MoE 专家通过 `EP=16` 切分后，**每张卡只需存放 18 个专家（约 18.5 GB 专家权重）**，单步 Decode 只需从 HBM 读取落在本卡 18 个专家上的被激活专家权重，使 HBM 带宽瓶颈降低一个数量级，从而把单卡 CUDA Graph 的 TPOT 压到极致。

---

## 3. Router 是干什么用的？（PD 分离全链路协调机制）

在 `start_router_service.sh` 中，启动了 `sglang_router.launch_router`（监听 `32000` 端口）：

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --host 0.0.0.0 \
    --port 32000 \
    --prefill "http://${PREFILL_IP}:30000" 8998 \
    --decode "http://${DECODE_IP}:31000" \
    --prefill-policy round_robin \
    --decode-policy round_robin
```

结合 `/usr/local/lib/python3.10/dist-packages/sglang_router/` 与 `/workspace/sglang/python/sglang/srt/disaggregation/` 源码，Router 在 PD 分离架构中承担 **四大核心职责**：

### 3.1 统一对外入口与并发双发（Concurrent Dual-Dispatch）
- 客户端只需向 Router 的 `http://Router_IP:32000/v1/chat/completions`（或 `/generate`）发送一次标准 OpenAI 请求。
- Router 内部的 PD 引擎根据 `--prefill-policy round_robin` 和 `--decode-policy round_robin` 选出一组 `(Prefill 实例, Decode 实例)`，然后**同时（并发）向 Prefill 节点（`:30000`）和 Decode 节点（`:31000`）转发该请求**，最后将 Decode 节点的 SSE 流式响应透传回客户端。

### 3.2 Bootstrap 信令注入（`bootstrap_host` / `bootstrap_port` / `bootstrap_room`）
- Prefill 和 Decode 是两个独立集群，它们怎么知道“哪个 Prefill 请求对应哪个 Decode 请求”？
- Router 在转发请求前，会自动生成全局唯一的 63-bit 随机整数 **`bootstrap_room`**，并将 Prefill 节点的 **`bootstrap_host`**（`PREFILL_IP`）和 **`bootstrap_port`**（`8998`，由 Prefill 节点的 `MooncakeKVBootstrapServer` 监听）连同 `bootstrap_room` 一起注入到发往 Prefill 和 Decode 的 HTTP JSON Payload 中。

### 3.3 确定性 DP Worker 路由协同（`follow_bootstrap_room`）
- 注意 Prefill 是 `dp_size=1`，而 Decode 是 `dp_size=16`（内部有 16 个 DP Worker 进程）。
- 在 `start_prefill_service.sh` 和 `start_decode_service.sh` 中都配置了 `--load-balance-method follow_bootstrap_room`。
- 对应 `managers/data_parallel_controller.py:213-224` 的 `follow_bootstrap_room_scheduler`：
  $$\text{target\_dp\_rank} = \text{req.bootstrap\_room} \pmod{\text{dp\_size}}$$
- 这样一来，Decode 节点的 `DataParallelController` 会根据 `bootstrap_room % 16` 确定性地将该请求分发给对应的 Decode DP Worker（同时兼顾随机均匀负载均衡），无需中心化状态同步。

### 3.4 触发 Mooncake 零拷贝 RDMA 异构状态传输（KV + KDA State + KPool + EAGLE Aux）
当 Prefill（8 卡 CP8）和被选中的 Decode DP Worker（1 卡 DP Rank）都收到带有相同 `bootstrap_room` 的请求后，会通过 `8998` Bootstrap 端口完成握手（见 `disaggregation/mooncake/conn.py`）：
1. **Decode 侧预分配并注册地址**：Decode Worker 从预分配池（`SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=128`）中分配：
   - 11 层 DSA 的 FP8 KV Cache 页与 Indexer KPool 页 + `DSA_TAIL` 槽位；
   - 34 层 KDA 的线性状态槽位（Conv state `[64*128*3]` + Recurrent state `[64, 128, 128]`）；
   - EAGLE 投机采样的 `aux_data` 缓冲区；
   并通过 Bootstrap Server 将这些显存的 RDMA 虚拟地址告知 Prefill 的 8 个 CP Rank。
2. **Prefill 侧 8 卡并行推送**：Prefill 完成前向计算及第 1 步 EAGLE Draft 后（因开启 `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1`）：
   - **KDA 状态拼接写入**：8 个 Prefill CP Rank 各自持有 8 个 Head 的 KDA 状态，通过 `_send_mamba_state_slice` 并发发起 RDMA Write，按 Head 偏移量直接拼入 Decode 单卡的 64-Head 状态槽位中；
   - **DSA KV & KPool 写入**：将 11 层 DSA 的压缩 KV Cache、KPool Index Cache 和尾状态写入 Decode 显存；
   - **首 Token 与 EAGLE Draft 辅助数据**：通过 TCP（`SGLANG_MOONCAKE_SEND_AUX_TCP=1`）将首个生成的 Token、最后一层 `hidden_states` 以及 EAGLE Draft 的 `output_topk_p / output_topk_index` 发送给 Decode，使 Decode 拿到请求后第一拍就能直接进入 6-Token EAGLE Verify！

---

## 4. （重点）对 `xllm-musa` 接入与优化 GLM-5.3-Flash 有什么帮助？

结合 `xllm-musa` 当前源码（`xllm/models/llm/musa/glm5_next.h`、`xllm/core/layers/musa/glm5_next/*`、`xllm/core/layers/musa/routed_moe.cpp`）以及 `tools/musa_glm5_3_vl_bench/` 下的实测剖析数据（`baseline_tp8_1111_analysis/REPORT.md`、`p0_optimization_results/REPORT.md`、`p1_optimization_results/REPORT.md`），SGLang 的这套方案精准指明了 `xllm-musa` 下一阶段从“功能跑通”迈向“生产级高吞吐低延迟”的 **六大演进方向**。

### 4.1 现状对比：`xllm-musa` 当前瓶颈在哪里？

当前 `xllm-musa` 在 `tools/musa_glm5_3_vl_bench/start_service.sh` 中采用的是 **单实例混部 + 纯 TP8（`TP_SIZE=8, EP_SIZE=1, DP_SIZE=1, CP_SIZE=1`）**：
- 根据 `baseline_tp8_1111_analysis/REPORT.md` 实测（C1, ISL=2048, OSL=16）：
  - **Prefill 阶段（414.4 ms）**：`all_reduce` 耗时 **74.5 ms（占比 18.0%，91 次调用）**，`grouped_gemm_fp8` 耗时 **60.2 ms（14.5%）**，`mhc_pre/post` 耗时 **52.9 ms（12.8%）**。
  - **Decode 阶段（1307.9 ms，15 步）**：**`all_reduce` 高达 420.6 ms（占比 32.2%，720 次调用！）**，`routed_moe_fp8_gemv_stage1 + stage2` 耗时 **388.4 ms（占比 29.7%）**。
  - 经过 P0（mHC MATE big-fuse、IndexPool 并行、KDA 批描述符）和 P1（KDA 输入投影合并、MTP Indexer TopK 共享）优化后，C1 的 TTFT 降至 ~660 ms、TPOT 降至 ~163 ms，但 **TP8 每层 2 次 AllReduce 的通信墙（32%+）和 `EP=1` 下 MoE 窄矩阵（`2048 / 8 = 256`）的访存/算力墙** 依然是架构级瓶颈。

---

### 4.2 借鉴一：补齐 `RoutedMoe` 的真 EP（DeepEP / All-to-All）与 `moe_tp=1` 支持（收益最大）

- **`xllm-musa` 现状诊断**：
  查看 `xllm/core/layers/musa/routed_moe.cpp:395-430, 1025-1045` 可以发现：
  1. 当前 `start_service.sh` 使用 `EP_SIZE=1, TP_SIZE=8`，导致每张卡都加载全部 288 个专家，且每个专家的 `moe_intermediate_size=2048` 被 `tp_pg_` 切成了极窄的 **`local_intermediate_size_ = 256`**。在 FP8 Grouped GEMM 和 Decode GEMV 中，`N/K=256` 不仅 Tile 效率低，而且每张卡每次都要跨 288 个专家做离散访存。
  2. 即使在 `xllm-musa` 中开启 `ep_size > 1`，当前 `RoutedMoeImpl::forward_with_selected_experts`（第 1039 行）使用的是 `ep_pg_->allreduce(output)`，且第 1032 行 `compact_prefill` 要求 `(ep_pg_ == nullptr || ep_pg_->world_size() == 1)`——即尚未接入面向 Token 分片（DP Attention 或 CP Scatter）的 **DeepEP All-to-All Dispatch / Combine**。
  3. 此外，`RoutedMoeSharedExpertImpl`（第 250 行）绑定了 `tp_pg_`，当 `moe_tp_group_` 为 1 时才不会对 Shared Expert 做 TP 切分。
- **落地建议**：
  1. **支持 `EP=8, moe_tp=1`（单机）与 `EP=16, moe_tp=1`（双机）**：当 `ep_size = world_size` 时，`collective_communicator.cpp:642` 已计算出 `moe_tp_size = world_size / ep_size = 1`，此时每张卡持有 `288 / 8 = 36`（或 `288 / 16 = 18`）个完整专家（`local_intermediate_size_ = 2048`），Shared Expert 自动运行在 `moe_tp_size = 1` 上。
  2. **接入 MUSA DeepEP（`normal` 用于 Prefill，`low_latency` 用于 Decode）**：参考 SGLang 容器中已适配摩尔线程 MUSA 的 DeepEP 后端，在 `RoutedMoeImpl::forward_with_selected_experts` 中，当输入 Token 处于 DP/CP 分片状态时，用 `deepep_dispatch -> forward_grouped_fp8 -> deepep_combine` 替换全量 Token 计算 + `allreduce`。
  3. **Shared Expert 与 Combine 融合**：借鉴 `SGLANG_SBO_COMBINE_SHARED=1`，将本地 `tp=1` 的 Shared Expert 输出直接融合进 DeepEP Combine 归约阶段，省去额外的加法 Kernel 与流同步。

---

### 4.3 借鉴二：Decode 阶段支持 `--enable-dp-attention`（`DP=8/16, attn_tp=1, EP=8/16`）

- **`xllm-musa` 现状诊断**：
  - 在 `xllm/core/layers/musa/glm5_next/glm5_next_decoder_layer.cpp:106-116` 中，`DenseMLP`、`Glm5NextLinearAttention`、`Glm5NextMLAAttention` 全部硬编码绑定在 `parallel_args.tp_group_` 上。
  - 在 Decode 阶段，45 层 × 2 = 90 次 TP AllReduce 占用了近 1/3 的端到端时间，且阻碍了跨机扩展（跨机跑 TP16 AllReduce 会被网络延迟直接拖垮）。
- **落地建议**：
  1. **引入 `attn_tp_group_` 与 `mlp_tp_group_` 解耦**：在 `ParallelArgs`（`xllm/core/framework/parallel_state/parallel_args.h`）中支持 `dp_size = world_size`（Attention DP 模式，即 `attn_tp_size = 1`）与 `moe_ep_size = world_size`。
  2. **前 3 层 Dense MLP 支持 `moe_dense_tp_size = 1`**：在 `Glm5NextDecoderLayerImpl` 构造 `DenseMLP`（层 `0..2`）时，允许传入 `single_rank_group`（`enable_result_reduction=false`），使前 3 层在 DP/CP 模式下本地独立计算，消除 3 次 AllReduce。
  3. **收益预估**：开启 `DP=8, attn_tp=1, EP=8`（单机）或 `DP=16, attn_tp=1, EP=16`（双机）后，34 层 KDA、11 层 DSA、3 层 Dense MLP 的 **TP AllReduce 耗时（原占 Decode 32.2%）将直接归零**，且 MLA KV Cache（512 维）和 Indexer KPool Cache 的总容量提升 8~16 倍！

---

### 4.4 借鉴三：Prefill 阶段引入 Hybrid CP（KDA Head-TP + DSA Interleave CP + mHC/MLP Token-Shard）

- **`xllm-musa` 现状诊断**：
  - 在 `xllm/core/layers/musa/glm5_next/glm5_next_mla_attention.cpp:54` 中，目前写死了：
    ```cpp
    CHECK_EQ(parallel.cp_size(), 1) << "GLM sparse attention currently supports ordinary TP";
    ```
  - 同时，在纯 TP8 下，每张卡上的 `MHCPre`（含 Sinkhorn）、`MHCPost`、`RMSNorm`、以及 Embedding 都必须对**完整的 `[S, 4096]` 序列**重复计算一模一样的结果！在 `baseline_tp8_1111_analysis/REPORT.md` 中，仅 `mhc` 相关算子就占了 Prefill 的 **12.8%**。
- **落地建议**：
  借鉴 SGLang 的 `MHCHybridDSACPLayerCommunicator`（`models/glm5_next.py:1060-1226`），在 `xllm-musa` 中实现 GLM-5.3-Flash 的混合序列并行（Hybrid CP）：
  1. **跨层保持 `[S / cp_size, 4, 4096]` 残差切分**：Embedding 后直接切分 Token，使得全部 90 个 `MHCPre / MHCPost`、全部 `RMSNorm`、前 3 层 `DenseMLP` 以及 42 层 MoE 的 `RoutedMoeGate` 计算量直接降为 **`1 / cp_size`（降为 1/8）**！
  2. **KDA 层走 `AllGather + Head-TP + ReduceScatter`**：在 `Glm5NextLinearAttentionImpl::forward` 入口对 `[S/8, 4096]` 做 `all_gather` 得到自然顺序 `[S, 4096]`，内部复用现有的 8-Head 切分逻辑执行 `causal_conv1d` 和 `chunk_gated_delta_rule`，出口将 `o_proj` 的 `allreduce` 替换为 `reduce_scatter` 回到 `[S/8, 4096]`。
  3. **DSA 层走 `Interleave AllToAll + KV/KPool AllGather`**：在 `Glm5NextMLAAttentionImpl` 中解除 `cp_size == 1` 限制；当开启 Prefill CP 时，保持本地 64 个 Head 不切分，入口通过 `all_to_all_single` 转为轮询交错序列（均衡 Causal Mask 负载），仅对 `kv_lora_rank(512)` 和 `IndexerKPool` 做跨 CP 组的 `all_gather`，算完 Attention 后再 `all_to_all_single` 转回连续分片。

---

### 4.5 借鉴四：激活第 45 层原生 MTP（EAGLE 投机解码）并共享 Indexer TopK

- **`xllm-musa` 现状诊断**：
  - 模型权重中包含原生的第 45 层 MTP Draft 模块（`model.layers.45`，属于 Sparse MoE + DSA 层，带 `eh_proj`、`enorm`、`hnorm`、`shared_head`）。
  - `xllm-musa` 在 P1 优化中已经实现了 `DsaTopkSharePlan`（让奇数 DSA 层 `7, 15, 23, 31, 39` 复用前一个偶数 DSA 层 `3, 11, 19, 27, 35` 的 Indexer TopK 结果）。
  - SGLang 在 Prefill 和 Decode 中均开启了：
    `--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`
    并且结合 `config.json` 中的 `"index_share_for_mtp_iteration": true`，让第 45 层 MTP Draft 直接复用主干第 43 层 DSA 的 Indexer TopK！
- **落地建议**：
  1. 在 `xllm-musa` 的投机采样框架中完整接入 `glm5_next` 的 `layers.45` MTP 权重（配置 `num_steps=5, topk=1, num_draft_tokens=6`）。
  2. 将 `DsaTopkSharePlan` 扩展到 MTP Draft 层：在 Verify / Prefill 阶段，将主干第 43 层（最后一个 DSA 层）计算出的 `Glm5NextIndexerSelection` 直接传递给第 45 层 MTP，省去 MTP 层的 Indexer 投影与 TopK 检索开销。
  3. 配合 Decode CUDA Graph（捕获 `bs * 6` Token），在算力充裕的 Decode 阶段通过一次前向验证 6 个 Token，可将端到端等效 TPOT 再压缩 **2.5x ~ 3.5x**。

---

### 4.6 借鉴五：PD 分离（Disaggregated Prefill/Decode）需要传输的四类异构状态

`xllm-musa` 已经具备 `disagg_pd_scheduler` 与 Mooncake KV 传输基础框架（见 `xllm/core/distributed_runtime/disagg_pd_*`），但要让 **GLM-5.3-Flash** 跑通 PD 分离，不能只传常规 KV Cache，必须参考 SGLang 的 `disaggregation/mooncake/conn.py` 扩展以下 **四类异构状态传输**：

1. **11 层 DSA 的压缩 KV Cache（`kv_lora_rank=512`, `fp8_e4m3`）**。
2. **6 个实体 Indexer 层（`3, 11, 19, 27, 35, 43`）的 KPool 压缩索引缓存与 `DSA_TAIL` 状态**：
   - 包含 `index_kpool=4` 压缩后的 FP8 池槽位（含 scale）以及未满 4 个 Token 的尾部状态（`tail_k, tail_weights`）。如果漏传 `DSA_TAIL`，Decode 第一步跨 4-Token 边界生成的 KPool 槽位就会失真。
3. **34 层 KDA 的线性递归状态（Conv State + Recurrent State）与跨 CP 切片拼接**：
   - 每个请求在每层 KDA 有 `conv_state`（`[64, 128, 3]`）和 `ssm/recurrent_state`（`[64, 128, 128]`）。
   - 当 Prefill 采用 8 卡 Head 切分（每卡 8 个 Head），而 Decode 采用 `DP=16, attn_tp=1`（单卡全部 64 个 Head）时，Prefill 的 8 个 Rank 需支持**按 Head 切片并发 RDMA Write 到 Decode 单卡状态槽位的对应偏移处**（如 SGLang 的 `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1` + `_send_mamba_state_slice`）。
4. **EAGLE/MTP 投机采样的 `aux_data`**：
   - 当启用 MTP 时，Prefill 节点在完成主干前向后会紧接着跑一次 MTP Draft，并通过 TCP/RDMA 将 `(首 Token, hidden_states, draft_tokens)` 一并传给 Decode 节点，使 Decode 第一步就能直接开始 Verify。

---

### 4.7 借鉴六：MUSA 运行时、通信环境变量与算子选型最佳实践

从 `start_prefill_service.sh` 与 `start_decode_service.sh` 可以直接提取已在 MUSA 4.3.8 + S5000 集群上验证稳定的运行时参数组合，供 `xllm-musa` 的 `tools/musa_glm5_3_vl_bench/env.sh` 与 `start_service.sh` 对标：

| 类别 | SGLang 生产脚本配置 | 对 `xllm-musa` 的参考意义 |
| :--- | :--- | :--- |
| **MUSA Runtime 队列与调度** | `MUSA_EXECUTION_TIMEOUT=30000000`<br>`MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`<br>`MUSA_DEVICE_MAX_CONNECTIONS=1`（Decode 脚本中配合 DeepEP 与多流并发） | 防止长序列 Prefill 或 Graph Capture 超时；控制硬件队列连接数避免多流 Context 竞争 |
| **MCCL / RoCE 通信调优** | `MCCL_PROTOS=2`<br>`MCCL_ALGOS=1`（Ring 算法）<br>`MCCL_BUFFSIZE=20971520`（20MB）<br>`MCCL_IB_GID_INDEX=3`（RoCEv2）<br>`MCCL_NET_SHARED_BUFFERS=0` | 在 `xllm-musa` 多机或大块 AllGather/ReduceScatter 场景下，显式配置 `MCCL_BUFFSIZE=20MB` 与 `MCCL_IB_GID_INDEX=3` 可显著提升 RoCE 吞吐与稳定性 |
| **KV Cache 精度** | `--kv-cache-dtype fp8_e4m3`（Prefill 与 Decode 均开启） | 将 11 层 DSA 的 `512` 维 KV Cache 从 BF16 压缩为 `fp8_e4m3`，KV 显存占用与 Decode 访存带宽需求直接减半，且与 MATE Sparse MLA FP8 接口天然匹配 |
| **算子后端组合** | - DSA Prefill & Decode：`tilelang`（底层即 MATE / TileLang JIT 内核）<br>- KDA Prefill：`flashkda`（块并行扫描）<br>- KDA Decode：`triton`（Fused Gate + Recurrent Step）<br>- Embedding 复制：`SGLANG_ENABLE_EMBED_REPLICATION=1` | `xllm-musa` 已在 P0 中接入了 MATE 的 `_tilelang_mhc_pre_big_fuse_kernel` 与 Sparse MLA；建议进一步在 `VocabParallelEmbedding` 中支持小显存代价的词表复制（省去每步 Embedding 的 TP AllReduce），并在 KDA Prefill 中对标 `flashkda` 吞吐 |

---

## 5. 总结：建议 `xllm-musa` 推进路线图

| 优先级 | 改造项 | 涉及 `xllm-musa` 核心模块 | 预期核心收益 |
| :---: | :--- | :--- | :--- |
| **P0** | **接入 MoE 真 EP（`EP=8, moe_tp=1`）+ DeepEP All-to-All** | `xllm/core/layers/musa/routed_moe.cpp`<br>`xllm/core/framework/parallel_state/collective_communicator.cpp` | 消除专家 `2048/8=256` 窄维度劣化；每卡仅存 36（或 18）个完整专家，大幅降低 Decode MoE 访存耗时（当前占 29.7%） |
| **P0** | **Decode 支持 DP Attention（`DP=8/16, attn_tp=1`）+ Dense MLP `tp=1`** | `xllm/core/layers/musa/glm5_next/glm5_next_decoder_layer.cpp`<br>`xllm/models/llm/musa/glm5_next.h` | **彻底消除 Decode 阶段 34 层 KDA + 11 层 DSA + 3 层 Dense MLP 的 TP AllReduce（当前占 TPOT 32.2%）**；KV Cache 容量扩大 8~16 倍 |
| **P1** | **DSA KV Cache 支持 `fp8_e4m3` 存储** | `xllm/core/layers/musa/glm5_next/glm5_next_mla_attention.cpp` | DSA KV Cache 容量翻倍、Decode 注意力访存减半、PD 分离 KV 传输量减半 |
| **P1** | **接入第 45 层 EAGLE/MTP 投机解码（共享第 43 层 Indexer TopK）** | `xllm/models/llm/musa/glm5_next_mtp.h`<br>`xllm/core/layers/common/dsa_topk_share_plan.h` | 单次 Decode Verify 生成 4~6 个 Token，端到端生成速度提升 2.5x+ |
| **P2** | **Prefill 支持 Hybrid CP（KDA Head-TP + DSA Interleave CP + mHC/MLP 切 Token）** | `xllm/core/layers/musa/glm5_next/*` | 消除 Prefill 阶段 mHC / RMSNorm / DenseMLP 的 8 卡重复计算；解决长序列 Causal Attention 负载不均 |
| **P2** | **PD 分离支持 GLM-5.3-Flash 四合一异构状态传输（Mooncake）** | `xllm/core/distributed_runtime/disagg_pd_*` | 实现 Prefill（CP8/EP8）与 Decode（DP16/EP16）解耦部署，兼顾低 TTFT 与高吞吐 TPOT |
