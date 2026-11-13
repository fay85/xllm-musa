# GLM-5.3-Flash Prefill Hybrid CP8 + EP8 / `moe_tp=1` 开发计划

本文只规划 GLM-5.3-Flash 的 **Prefill** 优化。阶段性 benchmark 固定使用 `--osl 1`，先测首 token 延迟和 Prefill 吞吐，不把 Decode TPOT 优化混入本计划。

当前 P0/CP0 实施进度见 [PREFILL_EP_MOE_DEVELOPMENT_STATUS.md](PREFILL_EP_MOE_DEVELOPMENT_STATUS.md)。

## 目标和边界

目标拓扑是单机 8 卡：

```text
attention / dense MLP: TP8
routed MoE:            EP8, moe_tp=1
shared expert:         TP1，在每个 EP rank 复制
PP:                    1
DP:                    1
```

在当前 `world_size=8, ep_size=8` 的并行组构造中，xLLM 已经能够创建：

```text
moe_tp_group: world_size / ep_size = 1
moe_ep_group: ep_size = 8
```

因此本任务的关键不是再增加一个 `ep_size` 参数，而是把 `RoutedMoeImpl::forward_with_selected_experts` 从“每卡本地算全部输入、最后 EP `allreduce`”改成真正的：

```text
本地 token + top-k route
    -> EP dispatch（按目标专家 rank 发送）
    -> 本 rank 的完整专家 grouped GEMM
    -> EP combine（按源 token 归还并累加）
    -> 本地 token 输出
```

Prefill 阶段必须支持不均匀 token 数、空 rank、专家路由倾斜和 `T` 不是 128/256 倍数的情况。`--osl 1` 只用于缩短 benchmark 的 Decode 尾部；它不能替代对 Prefill 输出、KV/state 写入和首 token 语义的校验。

## 对 SGLang Prefill 特性的补充盘点

上一版只把 EP/MoE 作为主线，漏掉了 SGLang Prefill 的另一个核心：**Hybrid CP 是层间数据布局，EP dispatch 必须建立在这个布局之上**。这确实是上一版计划的遗漏；而且原报告 4.4 已明确把 Hybrid CP 列为“借鉴三”。SGLang 的 `MHCHybridDSACPLayerCommunicator` 不是“先做 TP8，再额外给 MoE 加 EP8”，而是：

```text
层间契约：每卡 Plain 连续 token = S / 8
KDA：    CP 内 AllGather 完整序列 + 8-way Head shard + ReduceScatter
DSA：    Plain <-> Interleave AllToAll + 压缩 KV/KPool AllGather
MLP：    本地 token shard，权重复制，0 次 TP reduction
MoE：    本地 token shard + EP8 Dispatch/Combine
```

需要补回计划的 SGLang Prefill 特性如下：

| SGLang 特性 | Prefill 作用 | xLLM 当前状态 | 计划落点 |
|---|---|---|---|
| `--enable-prefill-cp --attn-cp-size 8` | 建立 CP8 通信域，层间只保留 `S/8` token | GLM5 KDA/MLA 都有 `cp_size == 1` 检查 | CP0/CP1 |
| Plain token shard | Embedding、mHC、RMSNorm、Dense MLP、MoE 只处理本地 token | GLM5 model 没有 CP shard/restore 主流程 | CP0 |
| `--cp-strategy interleave` | DSA 将连续 token 轮询打散，均衡 causal triangular load | 没有 GLM5 DSA interleave 入口 | CP2 |
| KDA Head-TP over CP | 完整序列扫描满足 causal conv/递归状态；每卡只算 8 heads | `glm5_next_linear_attention.cpp:153` 直接要求 CP1 | CP1 |
| DSA Interleave CP | 本地 query 只算 `S/8`，KV/KPool 以压缩形式 AllGather | `glm5_next_mla_attention.cpp:63` 直接要求 CP1 | CP2 |
| Embedding replication | 避免词表 TP 通信，之后直接按 token 切分 | GLM5 embedding 路径没有 SGLang 的 replication contract | CP0 |
| mHC/RMSNorm token-shard | 90 个 mHC/Norm 不重复计算完整序列 | 当前 TP8 会重复处理完整 token | CP0/CP3 |
| `moe-dense-tp-size=1` | 前 3 层 Dense MLP 权重复制，本地 token 独立计算 | Dense MLP 当前绑定 `parallel_args.tp_group_` | CP3 |
| Shared Expert combine fusion | 本地 Shared Expert 与 EP Combine 融合，减少加法/同步 | 当前有 stream overlap，但不是 EP combine 内融合 | EP2/EP3 |
| `enable-dp-lm-head` / 末端 gather | 仅在取最后 token/logits 时收集 CP 分片 | 当前没有 Hybrid CP communicator | CP4 |
| `dsa-prefill-backend=tilelang`、`linear-attn-prefill-backend=flashkda` | 选择适配 Prefill shape 的 kernel backend | xLLM 已有 MUSA KDA/DSA kernel，但没有统一 CP dispatch contract | CP1/CP2 的 kernel 验证 |

以下 SGLang 设置属于 PD/Decode 或 MTP 侧，本轮不作为 Prefill 主线实现，但不能误认为已经接入：`SGLANG_ENABLE_SPEC_V2`、第 45 层 EAGLE/MTP、`SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER`、Mooncake auxiliary transfer。若 Prefill 完成后要把 KDA/DSA 状态交给 Decode，必须另做状态拼接和传输验证；本计划只要求本地 Prefill state/cache 写入正确。

## 执行顺序修订：先 Hybrid CP，再 EP

EP 计划必须后移到 CP token-shard 契约之后。推荐顺序为：

```text
CP0  并行域、Plain shard、Embedding、残差/metadata 契约
CP1  KDA：AllGather + Head-TP + ReduceScatter
CP2  DSA：Plain/Interleave AllToAll + KV/KPool AllGather
CP3  mHC、RMSNorm、Dense MLP、Shared Expert 的本地 token 路径
CP4  Final Norm/LM Head 的末端 gather 与输出 restore
EP0  EP8 权重 owner 和 moe_tp=1
EP1  Prefill dispatch/combine correctness
EP2  MUSA DeepEP/MATE normal path
EP3  graph、overlap 和 --osl 1 benchmark
```

如果先做 EP8 而仍让每个 rank 接收完整 `[S, 4096]`，测到的只是“专家权重分布”，不是 SGLang Prefill 的 Hybrid CP；后续 CP 接入时还会重新设计 source token index、DSA interleave 顺序和 MoE combine。

## 调研后的当前状态

### 已有的并行拓扑

- `xllm/core/framework/parallel_state/collective_communicator.cpp:642-685` 已按 `moe_tp_size = world_size / ep_size` 创建 `moe_tp_group_` 和 `moe_ep_group_`。`world=8, ep=8` 时 `moe_tp_group_` 是单 rank，`moe_ep_group_` 覆盖 8 卡。
- `xllm/core/layers/musa/glm5_next/glm5_next_decoder_layer.cpp:118-121` 已将 GLM 稀疏层接到 `RoutedMoe`；`glm5_next_decoder_layer.cpp:240-256` 已区分 Prefill/Chunked Prefill/Decode。
- `xllm/core/framework/parallel_state/process_group.h:110-117` 和 `process_group.cpp:309-333` 提供 `all_to_all_single`，支持变长 split；这是正确性原型可以使用的通信入口。

### 已有的专家权重布局

- `xllm/core/layers/musa/routed_moe.cpp:418-430` 已按 EP rank 计算 `num_experts_per_rank` 和 `start_expert_id`。
- `routed_moe.cpp:493-532` 的 FP8 storage shape 已能表示每 rank 的完整专家；当 `moe_tp=1` 时 `local_intermediate_size` 应为完整的 2048，而不是 TP8 的 256。
- `routed_moe.cpp:1192-1240` 的加载逻辑仍以 `tp_pg_` 的 rank/world 为主要分片依据。EP8 + `moe_tp=1` 时必须明确使用 `moe_ep_group_` 的 expert owner，并防止某个 checkpoint shard 缺少本 rank 专家时错误地分配整套 dense/BF16 专家。

### 已有但不能直接等同于通用 EP 的 MegaMoE

- 当前默认 `XLLM_GLM5_MEGA_MOE=1`，入口在 `routed_moe.cpp:436-448`，但它强制 `ep_size == 1`，然后用 TP8 rank 手工承担 36 个专家。这是 GLM 专用的 MATE MegaMoE 路径，不是并行参数意义上的 `EP8 + moe_tp=1`。
- `xllm/core/layers/musa/musa_mega_moe.cpp:168-271` 已有 Prefill stage1/2/3、IPC buffer、Token ownership 和不均匀 `count` 处理，可作为 MUSA/MATE 的性能参考。它当前还假设 attention 输出在各 TP rank 上复制，并在调用方保留 TP/shared-expert 的组合语义。
- `tools/musa_glm5_3_vl_bench/compile_musa_mega_moe.py` 导出的是 MATE MegaMoE 专用 ABI；当前环境没有单独安装 `deep_ep` Python 包。因此不能把 SGLang 的 Python DeepEP API 当成 xLLM C++ ABI，必须先确认 MUSA 版本的 DeepEP/MATE runtime 是否可链接。

### 当前真正的瓶颈

通用 EP 路径在 `routed_moe.cpp:1099-1101` 对 EP group 做 `allreduce(output)`；当前默认的 GLM MegaMoE 则因 `ep_size==1` 走后面的 TP all-reduce。两者都不是目标的“按 source token dispatch、combine 后直接得到本地输出”语义，会导致：

1. 每个 rank 必须处理重复的全量 Prefill token；
2. EP8 仍然承担全量路由计算和大量无效专家检查；
3. `allreduce` 的结果语义无法支持未来 CP token-shard；
4. 专家 GEMM 的 `M` 维度没有随 token owner 分布而增大。

## 推荐开发路线

### P0：锁定基线和配置契约

先不改专家 kernel，增加可审计的启动配置和日志：

- 新增 Prefill 专用环境变量，建议命名为 `XLLM_GLM5_PREFILL_EP`，默认 `0`，显式 `1` 才启用新路径。
- `tools/musa_glm5_3_vl_bench/start_service.sh` 当前没有显式传 `--ep_size`；P0 必须新增 `GLM5_EP_SIZE`（baseline=1，EP 实验=8）并传给每个 worker。仅设置运行时开关而保持 `ep_size=1` 不会创建 `moe_ep_group_`，不能算 EP A/B。
- 新路径只接受 `model_type=glm5_next`、`world_size=8`、`ep_size=8`、`moe_tp_size=1`、`n_routed_experts=288`、`topk=8`、`hidden=4096`、`moe_intermediate=2048`、FP8 expert 权重；其他组合在构造阶段 `CHECK` 失败，不在运行中静默 fallback。
- 先保留当前 `XLLM_GLM5_MEGA_MOE=0` 的 TP8 expert 路径作为基线，避免把现有 MegaMoE 的自定义 IPC 通信和通用 EP A/B 混在一起。
- 启动脚本增加并行配置打印：global rank、TP/EP/MoE-TP group size、expert range、expert weight dtype、Prefill EP mode。
- `bench.sh` 的 Prefill A/B 固定 `--osl 1`，同时保存 `isl`、`shared`、`num_images`、concurrency、请求数、模型/二进制 hash 和完整环境。

验收：未启用新开关时现有 TP8 baseline 结果不变；启用新开关但拓扑不满足时在加载阶段给出明确错误。

### CP0：建立 Hybrid CP 的模型级 token-shard 契约

先把 Prefill 的输入和层间残差从完整 `[S, 4096]` 改成每个 CP rank 的 Plain 连续分片 `[S/8, 4096]`。建议新增独立开关 `XLLM_GLM5_PREFILL_CP=1`，初期只允许 `dp=1, cp=8, world=8`。

- `tools/musa_glm5_3_vl_bench/start_service.sh` 需要显式传 `--cp_size=8`；当前 xLLM 的 `collective_communicator.cpp:441-543` 已能创建 narrowed `tp_group`（CP8 时 attention TP=1）和 `cp_group`（8 ranks），但必须先用日志验证 rank layout。
- 在 `Glm5NextModelImpl::forward` 中，Embedding 后建立 Plain row metadata：每个 rank 保存 global row offset、local token count、global positions、sequence boundaries 和 restore index。对应 SGLang 的 `cp_plain_split`/`MHCHybridDSACPLayerCommunicator` 语义，不能只对 tensor 做 `narrow`，否则多请求 packed prefill 会丢失 sequence 边界。
- `SGLANG_ENABLE_EMBED_REPLICATION=1` 的语义是 embedding 权重复制、输入 token 按 CP 切分；不能把 vocab parallel 的 logits/词表通信误当成 CP shard。
- mHC 的四路残差、RMSNorm、position 和 `AttentionMetadata` 都必须沿同一个 local-row 顺序传递；所有 rank 对空/padded rows 仍须进入相同 collective schedule。
- 当前 `xllm/core/layers/musa/pcp_context.*` 是连续行 relay/状态传递，不能直接称为 SGLang Hybrid CP：它针对特定 PCP/前缀状态场景，且不是 DSA 的 Plain↔Interleave 变换。可以复用 row metadata 和 gather 工具，但要为 GLM5 建立独立 context contract。

验收：CP1 关闭时 TP8 baseline 不变；CP8 开启后每层日志都能证明 local rows、global positions 和 restore order 正确，`S=1/7/128/129/2048/4092/4096` 及多请求 packed 输入均无错位。

### CP1：KDA 的 AllGather + Head-TP + ReduceScatter

SGLang 的 KDA 不是本地计算 `S/8` 个 token，而是：输入 local rows → CP AllGather 成自然顺序完整 `S` → 每 rank 计算 64 heads 中的 8 heads → `o_proj` 部分和按 token ReduceScatter 回 local rows。

- 解除 `xllm/core/layers/musa/glm5_next/glm5_next_linear_attention.cpp:153` 的 CP1 检查。
- KDA 的 head shard 必须绑定 `cp_group_`，不能绑定 CP8 下已经缩窄为单 rank 的普通 `tp_group_`；输入投影、卷积参数、`A_log/dt_bias`、`f_b/g_b` 和 `o_proj` 的 shard 规则要逐项核对。
- AllGather 必须恢复自然 token 顺序，不能使用 DSA 的 interleave 顺序；否则 causal conv kernel=4 和 recurrent scan 的边界会改变。
- 输出使用 CP ReduceScatter，恢复 `[S/8, 4096]` 的层间契约；不要用全量 AllReduce 再丢弃 7/8 行。
- 需要验证每个 CP rank 的 KDA final conv/SSM state 写入：Prefill 的状态 owner、global sequence position 和后续首 token 读取必须一致，即使 Decode 优化暂不实现。

### CP2：DSA 的 Interleave CP 和压缩 KV/KPool AllGather

解除 `xllm/core/layers/musa/glm5_next/glm5_next_mla_attention.cpp:63` 的 CP1 限制，但 DSA 与 KDA 使用不同的数据布局：

- Plain local rows 通过 CP `all_to_all_single`（对应 SGLang `cp_plain_to_scattered`）转换为按 `global_position % cp_size` 轮询的 Interleave rows；保存可逆 permutation 和 sequence offsets。
- 每卡只用 local `S/8` Query 计算完整 64 heads；DSA 权重不做 head TP。
- 每卡生成本地 `kv_lora_rank=512` latent 和 Indexer KPool 后，在 CP group 上 AllGather 压缩 latent/KPool，不 AllGather 展开的 64-head KV。
- Indexer causal lengths、top-k 候选、`DSA_TAIL` 和 page/slot mapping 都必须使用 global position；interleave 只改变计算排列，不能改变 cache 的逻辑顺序。
- Attention 输出从 Interleave rows 通过逆 AllToAll 恢复 Plain local rows，继续进入 mHC/MLP/MoE。
- 先用已有 MUSA DSA kernel 做 eager correctness；SGLang 的 TileLang backend 是实现参考和性能目标，不能直接假设 xLLM C++ ABI 可调用。

验收：比较 CP1 与 CP8 的 per-token DSA output、Indexer selection、KV/KPool cache 和首 token logits；覆盖 causal 长度极不均匀的长序列，确认 interleave 确实均衡 query/indexer 工作量。

### CP3：mHC、RMSNorm、Dense MLP 和 Shared Expert 的 local-token 路径

- `mhc_expand`、所有 `MHCPre/MHCPost`、RMSNorm 和 `mhc_contract_mean` 只处理 local rows；mHC 的 Sinkhorn 参数是复制权重，不需要 CP 通信。
- 前 3 层 Dense MLP 改用 `single_rank_group_`/等价 TP1 线性层，权重复制，输入输出保持 local rows；不能继续绑定全局 TP group 做 AllReduce。
- Routed MoE Gate 在 CP local rows 上选 top-k；EP dispatch 的 source token index 必须携带 global row/sequence 信息，但 payload 只发送 local rows。
- Shared Expert 只计算 local rows；之后在 EP Combine 阶段融合 routed output，避免额外的全量 token 加法和 stream sync。

### CP4：Final Norm、LM Head 和输出 restore

- Final mHC contract、RMSNorm 在 local rows 上执行。
- 只有提取每个请求最后一个 token、计算 logits 时做一次 CP gather；不要在每层恢复完整序列。
- 建立 `last_token_local_index`/`global_to_local` metadata，保证请求顺序和图片 token 插入位置不被 Plain/Interleave 变换破坏。
- `--enable-dp-lm-head` 在本轮 `dp=1` 下不等于“每卡独立完整 LM head”；要明确 LM head group、CP gather 和 vocab shard 的边界。

验收：CP8 的最后 token hidden/logits 与 CP1 baseline 对齐；多请求不同长度、末 rank 空 padding、图文混合输入都必须恢复正确请求顺序。

### EP0：实现无通信的单 rank EP 语义和权重契约

先在单进程/单卡验证 `moe_tp=1` 的本地完整专家计算，排除通信因素：

1. `RoutedMoeImpl` 保存三个明确的 group：`moe_tp_group_`、`moe_ep_group_` 和用于模型主体的 `tp_group_`；不要用 `tp_pg_` 推断 expert owner。
2. EP8 时每 rank 分配 `[36, 2*2048, 4096]` 的 FP8 `w13` 和 `[36, 4096, 2048]` 的 FP8 `w2`，block scale 使用完整专家的 128x128 shape。
3. 重写 `load_experts` 的 owner 逻辑：全局 expert id 由 `ep_rank * 36 + local_id` 得到；读取 checkpoint 时按 expert id 选择本 rank 的 36 个专家，禁止因为当前 shard 没有本 rank 专家而创建整套 dense storage。
4. Shared Expert 使用 `moe_tp_group_`。在 `moe_tp=1` 时，它必须是完整权重的本地副本，不能沿用 TP8 的 Column/Row split 和 TP all-reduce。
5. 本地 grouped FP8 Prefill GEMM 的输入、输出和 route weight 与原路径逐元素对照；先不引入异步 stream 或 graph。

验收：单 rank 的 36 个完整专家与独立 BF16/FP8 reference 对比；覆盖空 route、重复 expert、top-k=8、SwiGLU clamp=10、FP8 scale 边界。该阶段不宣称多卡 EP 已完成。

### EP1：正确性优先的 EP All-to-All dispatch/combine

先使用已有 `ProcessGroup::all_to_all_single` 做一个可读、可验证的 Prefill 路径。它的目标是确定通信语义，不是最终性能版本。

#### Dispatch

对本 rank 的 `T` 个 token 和 `[T, 8]` route 做以下 device-side 处理：

1. 将 route flatten 成 `[T*8]`；计算 `dst_rank = global_expert_id / 36` 和 `local_expert_id = global_expert_id % 36`。
2. 按 `(dst_rank, local_expert_id, source_token, topk_slot)` 稳定排序，生成 packed hidden、local expert id、route weight、source token id 和 top-k slot。
3. 为每个目标 rank 生成发送行数；交换 counts 后调用 `all_to_all_single` 发送 packed rows。
4. 所有 rank 都必须参与 collective；发送/接收 0 行时使用合法的空 contiguous tensor 和 0 split，不能提前 return。

#### Local expert compute

- 对接收的 route rows 按 local expert id 生成 grouped GEMM metadata。
- Prefill 使用已有 MUSA FP8 grouped/ragged GEMM；禁止调用 `forward_reference_experts`，因为它会把 route id 搬到 CPU 并逐专家循环。
- Shared Expert 与 routed expert 可以在独立 stream 上重叠，但 EP1 先在同一 stream 完成正确性，再加入 event/stream overlap。

#### Combine

1. 将每个 routed output 携带的 `source_token_id` 和 `topk_slot` 按原 source rank 重新 pack。
2. 用反向 `all_to_all_single` 归还到源 rank。
3. 在 device 上按 `source_token_id` 做 weighted sum，得到本地 `[T, 4096]`；每个 token 只出现一次输出，不能再做 EP allreduce。
4. Shared Expert 输出按模型语义相加一次。`moe_tp=1` 时不需要 shared expert TP reduction；attention TP 的其他层归约保持原有路径。

建议新增的职责（文件名可在实现时调整）：

```text
xllm/core/kernels/musa/moe_ep_dispatch.cu
xllm/core/kernels/musa/moe_ep_combine.cu
xllm/core/layers/musa/routed_moe_ep.cpp/.h
```

kernel 至少需要覆盖：route pack、stable metadata、source-index restore、weighted scatter-add、空 rank 和非均匀 token 数。不要在 host 上读取每个 token 的 expert id 来构造通信 payload。

验收：2/4/8 rank 的随机 route reference；T={1, 7, 128, 129, 2048, 4092, 4096}；每种形状覆盖全空 rank、单专家热点、均匀路由和跨 rank top-k。比较专家 id/顺序、dispatch counts、combine output、shared output 和最终 logits。

### EP2：替换为 MUSA DeepEP/MATE 高性能通信

EP1 通过后再决定最终 backend：

#### 路径 A：MUSA DeepEP

- 以 SGLang 的 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` 为语义参考：normal dispatch 返回 received hidden、received expert ids/weights、每专家 token counts 和 handle；combine 使用 dispatch handle 还原源 token。
- 在 xLLM 中不复制 Python dispatcher；新增一个 C++/MUSA wrapper，绑定当前安装版本的 DeepEP runtime/FFI，确认 `Buffer.get_dispatch_layout`、`dispatch`、`combine` 的 MUSA ABI、stream 和 workspace 生命周期。
- `normal` 模式用于 Prefill；不要直接使用 decode-oriented `low_latency` 模式。DeepEP buffer 按 `max_tokens_per_rank`、288 experts、topk=8、hidden=4096 预分配，容量不足时在执行前 CHECK。
- 通信 group 必须是 `moe_ep_group_`，不能传 `tp_group_`；每个 expert GEMM 的 TP group 必须是 single-rank `moe_tp_group_`。

#### 路径 B：复用/扩展现有 MATE MegaMoE

- 如果当前 MUSA runtime 没有可用 DeepEP ABI，优先把 `MusaMegaMoe` 改成显式接受 `moe_ep_group_`，解除 `routed_moe.cpp:438-440` 的 `ep_size==1` 限制。
- 保留 MATE Prefill stage1/2/3 的完整专家布局，但把 token ownership、dispatch/combine 的 group、shared expert 合并和 output ownership 与通用 EP 契约对齐。
- 不得仅把现有 `tp_pg_` 改成 `moe_ep_group_`：当前 MegaMoE 的 IPC buffer、rank ownership 和输出 buffer 都按“attention TP8 replicated input”设计，必须先验证 CP/token-shard 输入。
- EP2 的 MATE 路径仍要通过 EP1 的 dispatch/combine reference；MATE graph/eager 一致不等于与 TP8 BF16 reference 一致。

最终选择依据是：真实 MUSA ABI 可调用、Prefill 数值一致、通信 bytes、workspace 显存和 8 卡 trace。不能因为 Python 包存在就把路径标为已接入。

### EP3：接入 GLM Prefill 主路径

在 EP1/EP2 的 EP kernel 稳定、CP0-CP4 已建立 local-token 契约后修改 `RoutedMoeImpl::forward_with_selected_experts`：

```text
if prefill && prefill_ep_enabled:
    routed = ep_dispatch_compute_combine(...)
else:
    existing TP/MegaMoE path
```

要求：

- EP 路径在进入专家前不复制整个 `T x 4096` 到每张卡；每个 source rank 只 dispatch 自己拥有的 token。
- 删除 EP 路径的 `ep_pg_->allreduce(output)`；combine 已返回 source rank 的完整 routed output。
- Shared Expert 计算输入只使用本 rank 的 token，输出只加一次。
- `skip_tp_allreduce` 要由调用方语义明确传递；不能用它掩盖 EP combine 或 Shared Expert 的重复归约。
- 第一版关闭 Prefill CUDA graph，先完成 eager correctness；graph 只在 EP4 加入。
- Chunked Prefill 使用同一 dispatch/combine 协议；chunk 边界不能改变 source token index 的局部编号契约。

## EP4：Graph、workspace 和 overlap

只在 eager 多形状正确后处理：

1. 为每个支持的 token shape 固定 dispatch metadata、counts buffer、packed rows、combine output 和 MATE/DeepEP workspace；capture 内不能动态创建 ProcessGroup 或 Python/host 对象。
2. 处理 graph replay 的空 rank：所有 rank 必须执行相同数量的 collective，空 rank 通过预分配 sentinel/0-row schedule 参与通信。
3. Shared Expert 与 dispatch/compute/combine 做 event overlap；逐层检查 stream wait，不能把异步错误变成随机 hang。
4. 记录 workspace bytes、peak HBM、dispatch/combine bytes、MATE kernel 时间和 collective 等待时间。

## 验证和 benchmark 协议

### 数值验证顺序

1. 单层、随机权重、随机 route：TP8 baseline vs EP8 `moe_tp=1`。
2. GLM 真实 checkpoint、单层 Prefill：比较 routed output、shared output、FFN output。
3. 真实 45 层 Prefill：比较最后 hidden、首 token logits、top-1 token；允许 FP8 GEMM 的已知误差，但不能有 route id 集合变化或 NaN。
4. 覆盖文本和图文请求，ISL={1, 128, 2048, 4092, 4096}，并覆盖 `--enable_chunked_prefill` 开关。
5. 每个 shape 至少重复 eager 两次和 graph replay 两次；检查第二次 replay、空 rank 和非均匀行数。

### `--osl 1` 端到端 Prefill A/B

固定以下条件，避免 Decode 结果污染：

```bash
export GLM5_CP_SIZE=1
export GLM5_EP_SIZE=1
export XLLM_GLM5_PREFILL_EP=0   # baseline
export XLLM_GLM5_PREFILL_CP=0
export XLLM_GLM5_MEGA_MOE=0
bash tools/musa_glm5_3_vl_bench/bench.sh 1 \
  --isl 2048 --shared 1024 --osl 1 --num-images 1 \
  --requests 5

export GLM5_CP_SIZE=8
export GLM5_EP_SIZE=8
export XLLM_GLM5_PREFILL_EP=1   # EP8 + moe_tp=1
export XLLM_GLM5_PREFILL_CP=1   # Hybrid CP8
export XLLM_GLM5_MEGA_MOE=0
bash tools/musa_glm5_3_vl_bench/bench.sh 1 \
  --isl 2048 --shared 1024 --osl 1 --num-images 1 \
  --requests 5
```

每轮都要保存：

- TTFT mean/p50/p90/p99 和首 token 成功率；
- prompt tokens、cached tokens、completion tokens；
- Prefill MoE dispatch/compute/combine 时间及通信 bytes；
- 各 rank 接收 token 数、各专家 token histogram、最大/平均 skew；
- HBM 峰值、workspace 大小、graph capture/replay 状态；
- 二进制、MATE/DeepEP `.so`、模型 config 和 prompt hash。

性能拆分至少保留三种对照：现有 TP8/CP1/EP1；CP-only 的单层或合成测试；目标 CP8/EP8/`moe_tp=1`。不能直接做完整模型 CP8/EP1 对照，因为 EP1 会让每个 CP rank 保留全部 288 个专家，显存和权重加载成本不代表目标部署。

`--osl 1` 只能作为单请求端到端入口。性能结论以 Prefill trace 中 `batch_forward_type=PREFILL/CHUNKED_PREFILL` 的区间和 MoE kernel/collective 时间为准；不能把 HTTP TTFT 的全部差异都归因于 EP。

### 通过标准

- correctness：EP8 与 TP8 baseline 的 route 集合和 top-k 顺序一致；首 token logits 在预设 FP8 误差范围内；无死锁、NaN、空 rank 越界或 graph replay 差异。
- memory：每 rank 只保留 36 个 routed experts，Shared Expert 为完整 TP1 副本；不能出现整套 288 专家或 TP8 `intermediate=256` 的错误布局。
- communication：EP 路径不含 `ep_pg_->allreduce(output)`；trace 能看到 dispatch/combine，且每个 source token 只回收一次。
- performance：在固定 ISL/图片/请求数下，与 `XLLM_GLM5_MEGA_MOE=0` 的 TP8 baseline 比较 TTFT、MoE 总耗时和显存；只有 Prefill 区间稳定改善才进入默认开关评估。

## 主要风险和禁止事项

| 风险 | 处理方式 |
|---|---|
| 把现有 MegaMoE 当成通用 EP | 先通过 `ep_size=8` 的 group/weight/route 日志证明真正使用 `moe_ep_group_` |
| EP dispatch 后再 allreduce | combine 已按 source token 归还；EP 路径禁止重复 allreduce |
| Shared Expert 重复计算或重复相加 | `moe_tp=1` 本地完整计算，每个 source rank 只加一次 |
| host 读取 token route 后逐 token 打包 | 最终路径使用 MUSA metadata/pack kernel；不得使用 `forward_reference_experts` 作为 Prefill fallback |
| 只支持均匀路由 | 测试空 rank、热点专家、4092 token 和不均匀 split |
| graph 中 collective 次数不一致 | 固定 schedule，空 rank 也参与所有 dispatch/combine collective |
| 直接依赖未确认的 DeepEP ABI | 先检查实际安装库、导出符号、MUSA 架构和 stream/workspace 契约；失败时走已验证的 MATE 专用路径或停在 EP1 |
| 只看 `--osl 1` 的 TTFT | 同时记录 Prefill trace、通信 bytes、MoE histogram 和 logits correctness |

## 参考代码和实现映射

| 语义 | 当前参考 | 计划落点 |
|---|---|---|
| EP/TP group 创建 | `xllm/core/framework/parallel_state/collective_communicator.cpp:642-685` | 保持 group 拓扑，修正 RoutedMoe 使用方式 |
| 变长 All-to-All API | `xllm/core/framework/parallel_state/process_group.{h,cpp}:110,309` | EP1 correctness transport |
| GLM MoE 调用 | `xllm/core/layers/musa/glm5_next/glm5_next_decoder_layer.cpp:240-256` | Prefill gate 和 batch type 契约 |
| 当前 routed path | `xllm/core/layers/musa/routed_moe.cpp:815-920,997-1189` | 拆出 EP dispatch/compute/combine，移除 EP allreduce |
| GLM 专用 MATE EP 参考 | `xllm/core/layers/musa/musa_mega_moe.{h,cpp}` | EP2 MATE fallback/性能路径 |
| MUSA FP8 MoE kernel | `xllm/core/kernels/musa/musa_ops_api.h`、`xllm/core/kernels/musa/CMakeLists.txt` | 新增 dispatch/combine kernel 注册 |
| 其他 backend 语义参考 | `xllm/core/layers/mlu/fused_moe_all2all.cpp`、`xllm/core/layers/common/deep_ep.{h,cpp}` | 只复用 route/dispatch/combine 语义，不照搬 MLU ABI |
| SGLang DeepEP 参考 | `/data/albert/Projects/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py` | normal dispatch/combine 生命周期和 metadata 语义 |
| MATE 版本 | 安装 `mate 0.2.8.post1`；checkout `/data/albert/Projects/mate` commit `9d3260d633597f08fc9d5d4dc73344872241c34d` | 以实际安装 ABI 为准，不能假定 checkout API 等于安装包 |

本计划不包含 Decode TP/DP、KV cache 压缩、EAGLE/MTP 或 PD 状态迁移；CP8 Hybrid CP 已包含在 CP0-CP4。EP dispatch 从第一天就使用 source-rank/local-token 契约，不能把“每卡完整复制 Prefill 输入”写死在接口中。
