# GLM5 Prefill EP / Hybrid CP 开发状态

更新时间：2026-11-14

本文档对照 [开发计划](PREFILL_EP_MOE_DEVELOPMENT_PLAN.md)，记录 P0、CP0、CP1、CP2、CP3、CP4、EP0、EP1、EP2、EP3 和 EP4 的实际落地情况。阶段编号以开发计划为准。

## 当前阶段

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| P0：锁定基线和配置契约 | 已完成 | 启动脚本与 GLM5 模型构造阶段都校验 8 卡 EP/CP 试验契约。 |
| CP0：模型级 token-shard 契约 | 已接通 | CP3 已把均衡连续 shard 落到 token/embedding/positions 和层间 tensor。 |
| CP1：KDA AllGather + Head-TP + ReduceScatter | 已完成（单层） | 真实 8 卡 MCCL 验证通过；单层接收 Plain local rows、按 CP group 分 head、返回 local rows。 |
| CP2：DSA Interleave CP | 已完成（单层） | Plain↔Interleave、压缩 latent/KPool gather、tail 边界与共享 selection 已接通；3 组真实 CP8 测试通过。 |
| CP3：mHC / RMSNorm / Dense / Shared local rows | 已完成（local-token 路径） | 模型入口、四层混合模型、空 shard 和 FP8 local FFN 已接通；EP dispatch/combine 由 EP1 接入。 |
| CP4：Final Norm / LM Head / 输出 restore | 已完成（Prefill） | Final Norm 保持 local rows；padded device AllGather 按 CP0 offsets 恢复全局 Plain rows，两个 LM Head 重载支持 global selected indices。Decode、mixed/speculative 仍按支持范围拒绝；CP8 graph 的真实 checkpoint EP 链路已接通。 |
| EP0：单 rank EP 语义和权重契约 | 已完成 | `moe_tp=1` 的单 rank 无通信语义、EP owner 与 MoE TP 权重切片已分离；共享专家使用 single-rank group。 |
| EP1：正确性优先的 EP All-to-All | 已完成（eager reference） | 设备侧 route pack、counts exchange、双向 All-to-All、local expert compute 和 weighted combine 已接入；性能 kernel/DeepEP/MATE 仍未接入。 |
| EP2：MUSA DeepEP/MATE normal path | 已接入（MATE eager） | `XLLM_GLM5_PREFILL_EP_BACKEND=mate` 复用 MATE Prefill Stage1/2/3；每个 EP rank 直接提交 local token，完成设备侧 dispatch、grouped FP8 GEMM 和 combine。真实 288 expert checkpoint 已完成 8 rank 服务启动和一次文本请求；本次只验证链路，不给出 TTFT 收益结论。 |
| EP3：GLM Prefill 主路径 | 已接入（CP8+EP8 eager） | GLM Decoder 的 local rows、Routed Gate、MATE dispatch/combine、Shared Expert 和 chunked prefill 使用同一 token-shard 契约；EP 路径不做 output AllReduce。 |
| EP4：Graph、workspace、overlap | 已完成（CP8+EP8 Prefill） | MATE workspace、local output buffer 和 CP graph metadata/layout 已固定；CP8+EP8 graph capture/replay 覆盖 4096 分块和 3616 尾块，Shared Expert overlap 保持启用。 |

## P0 已完成内容

`tools/musa_glm5_3_vl_bench/start_service.sh` 新增以下配置契约：

- `GLM5_EP_SIZE` 和 `GLM5_CP_SIZE`，默认值均为 `8`；需要单卡 baseline 时显式设为 `1`。
- `XLLM_GLM5_PREFILL_EP` 和 `XLLM_GLM5_PREFILL_CP`，默认值均为开启；需要单卡 baseline 时显式设为 `0`。
- 启用 Prefill EP/CP 时，当前基线分别要求 `ep_size=8` / `cp_size=8`、`dp_size=1`，并使用 8 个 worker；脚本也会拒绝非 8 worker 的非默认 EP/CP size。
- 启动命令显式传递 `--ep_size` 和 `--cp_size`，启动日志打印完整并行契约。

模型构造阶段会再次校验启用的契约，并确认 native `cp_group` 或 `moe_ep_group` 已建立。脚本默认启动已验证的 CP8/EP8 graph Prefill；需要普通运行路径时显式覆盖这些环境变量。

单卡 baseline 配置是：

```bash
GLM5_EP_SIZE=1 GLM5_CP_SIZE=1 \
XLLM_GLM5_PREFILL_EP=0 XLLM_GLM5_PREFILL_CP=0 \
XLLM_GLM5_MEGA_MOE=0
```

脚本默认目标配置是：

```bash
GLM5_EP_SIZE=8 GLM5_CP_SIZE=8 \
XLLM_GLM5_PREFILL_EP=1 XLLM_GLM5_PREFILL_CP=1 \
XLLM_GLM5_PREFILL_EP_BACKEND=mate \
XLLM_GLM5_MEGA_MOE=0 \
GLM5_ENABLE_GRAPH=true GLM5_ENABLE_CHUNKED_PREFILL=true \
GLM5_MAX_TOKENS_PER_CHUNK=4096
```

目标配置已用于真实 checkpoint 的 CP8/EP8 服务验证；EP1 合成 correctness、MATE eager 和 graph capture/replay 均保留为分层验证入口。当前只对单序列纯文本 chunked Prefill 给出端到端结果，不把它外推为完整性能基线。

## CP0 已完成内容

`AttentionMetadata` 新增了 GLM5 Prefill CP0 字段：

- 当前 CP rank 和 CP size；
- 全局 token 行数；
- 当前 rank 的 `[local_row_begin, local_row_end)`；
- 当前 rank 的连续 `local_row_indices`；
- 所有 rank 的均衡连续 `row_offsets`。

`Glm5NextModelImpl` 在 eager Prefill 和 Chunked Prefill 中按以下规则生成元数据：

```text
rows_per_rank = global_rows // cp_size
remainder     = global_rows % cp_size
begin(rank)   = rank * rows_per_rank + min(rank, remainder)
count(rank)   = rows_per_rank + (rank < remainder ? 1 : 0)
```

CP0 最初只建立上述描述；CP3 已在准备完整序列 metadata 后实际切分 token IDs（先切再查 embedding）或输入 embedding，以及 positions。层间 residual/hidden states 保持 Plain local rows，KV/state 仍遵循 CP1/CP2 的全局请求语义。CP graph 仅支持 GLM5 单序列 chunked Prefill；Decode、speculative 和 dummy 仍会显式拒绝。

## CP1 已完成内容

- `Glm5NextLinearAttentionImpl` 在 `XLLM_GLM5_PREFILL_CP=1` 且 `cp_size>1` 时使用 `cp_group_` 作为 head 分片组，要求普通 attention TP=1、DP=1、world=CP。服务入口仍锁定 CP8。
- `q/k/v/b`、卷积通道、`A_log/dt_bias`、`f_b/g_b` 和 `o_proj` 均按 CP rank 分片；`f_a/g_a` 的 128 个低秩通道、`o_norm` 保持复制。64 heads 在 CP8 下为每卡 8 heads。
- 输入严格要求 `[local_rows, hidden]`，沿 CP0 的均衡连续 offsets 做一次变长 AllGather。补齐的通信行在进入 causal convolution 和 MATE KDA 前移除，全局 `q_cu_seq_lens` 和 state slot id 保持不变，packed sequence 边界不被 shard 边界替代。
- `o_proj` 在 CP 路径关闭 AllReduce，输出部分和做一次真正的 MCCL ReduceScatter。不能直接复用仅在全局尾部 padding 的通用 ReduceScatter：例如 S=4092 时，各 rank 的 512/511 行必须分别 padding，再按本 rank 的真实行数去掉 padding。S=1/7 的空 rank 也参与完整通信和本地 head 的状态更新。
- `KVCacheShape` 增加可选 `linear_state_world_size`；LLM engine 将 GLM5 Hybrid CP 的 KDA 状态估算与分配宽度设置为 CP size，普通 attention TP 保持 1。VLM engine 当前使用 world/DP 的状态宽度，CP8/DP1 已为 8。
- KDA state 每个 rank 保存所有请求的本地 heads：conv `[slots, 3*8*128, 3]` BF16，SSM `[slots, 8, 128, 128]` FP32；全局 sequence/state slot 顺序不变。
- CP 路径显式拒绝 Decode、mixed/speculative/dummy、错误的 CP metadata 和完整复制输入；CP=1 原有 Prefill/Decode/graph 路径保留，GLM5 CP8 graph 走固定 chunk contract。

算子复用已有 MUSA `causal_conv1d_prefill` 和 MATE `flash_kda` C++ wrapper，不增加 attention kernel、CPU fallback 或新外部 ABI。仓库内参考了 MLU `glm5_next_kda.cpp` 的 head/权重契约及 `glm5_next_kda_pcp.cpp` 的 state 语义；MLU 的 token/chunk relay 方式没有照搬。Hybrid CP 通信语义遵循原 [SGLang 对照报告](SGLANG_GLM5_3_FLASH_PARALLEL_AND_XLLM_GUIDE.md) §1.3。

## CP2 已完成内容

- DSA 在 Prefill CP 下使用普通 TP=1 的复制权重，每卡完整 64 heads，只计算本地 Interleave Query；禁止 KV split，启动脚本显式传 `--kv_split_size=1`。
- Plain local hidden/positions 经 AllToAll 转成按 packed 全局行号轮询的 Interleave 布局。原 request row、causal position、page/slot mapping 保持全局语义。输出和 selection 以无损 byte packing 共用逆 AllToAll，均恢复 Plain local rows；共享 Indexer 的后续 DSA 层可直接复用该 selection。
- 仅 AllGather 512 维 BF16 latent，然后按全局 slot 写入完整 BF16/FP8 主缓存；不传输展开后的 64-head KV。
- KPool 在 Plain 分片上计算，增加一次最多 3 行/rank 的 key/gate halo AllToAll，补齐跨 rank 的 rate-4 窗口。每卡只生成属于自己的完成池，以 **132 字节/池** AllGather，恢复各卡完整 KPool；没有广播整段原始 key/gate。
- 每请求最后最多 4 行 key/gate 通过独立小型 AllGather 更新 `DSA_TAIL`，支持 Chunked Prefill、读取旧 state slot 并写入新 slot。还有一轮 positions 元数据 gather。因而有 Indexer 的路径是 **2 次 activation AllToAll + 1 次 halo AllToAll + latent/pool/tail/positions gather**；复用 selection 的层省去 halo/pool/tail。
- 从现有 Indexer 拆出“缓存更新”和“按任意 query rows 选择”接口，复用已有 MUSA 压缩/量化 kernel、MATE sparse MLA 与 FP8 paged scoring ABI。主数据计算全部在 MUSA，host 只校验和准备调度 metadata。
- CP2 的 eager 路径继续验证请求内 positions、state/slot 映射；GLM5 CP8 graph 路径使用单序列、均匀 shard、设备侧固定 metadata。CP Decode、mixed/speculative/dummy、空请求及非法 state/slot 映射仍拒绝。空 rank、非整除长度、跨页、跨 shard、不同历史长度的 packed 请求已在 eager 路径验证。

通信细节、参考 backend、支持范围与复现入口见 [CP2 验证记录](cp2_validation/README.md)。后续可将本层 host metadata 校验与 permutation 计划移到模型级复用；本轮优先保证 eager 语义正确，没有给出性能收益结论。

## CP3 已落地内容

- 模型入口保留全局 `q_cu_seq_lens`、page/slot 和 KDA state descriptor，再按 CP0 offsets 切分 token/embedding/positions。Embedding 权重复制，`mhc_expand`、层间 mHC/PendingMHC、RMSNorm、`mhc_contract_mean` 和最终 Norm 都只处理 local rows。
- Decoder 校验当前 hidden/positions、pending residual/x 的行数。Dense MLP 绑定独立 `single_rank_group_`，完整加载 Gate/Up/Down 权重，关闭结果 AllReduce；前 3 层不再重复完整序列 FFN。
- Routed Gate 只为 local rows 生成 top-k。新增 `RoutedMoeTokenShard`，携带 source CP rank、global packed row、request row 和 causal position；索引构建在 MUSA 上执行。该接口是后续 EP dispatch 的输入契约，**当前没有发送 EP payload**。
- Shared Expert 绑定独立单卡组，使用完整权重处理 local rows；关闭 CP 下旧的 Shared overlap stream/event 路径，在当前 stream 上把本地 Shared 输出加到 routed 输出。后续 EP combine 将接替本地 routed 分支，目前没有实现 combine kernel 融合。
- 空 rank 跳过空行 Norm/Gate/FFN 的 kernel launch，mHC 保持正确的零行 shape；KDA/DSA 的通信和缓存更新仍由所有 CP rank 参与。
- 显式拒绝 CP+EP>1、CP+MoE TP>1 和 CP+MegaMoE。CP3 数值测试使用 **CP8 / EP1 / MoE TP1、每卡复制小型专家集**；这只是开发验证拓扑，不是 288 专家完整模型的部署方案，也不会自动回退到该拓扑。
- 修复串层发现的 CP2 边界：共享 Indexer 层没有 KPool/tail 分配，因此只在 `enable_indexer` 时检查 tail shape/slot 上限。测试使用真正缺省 pool/tail 的共享层缓存。
- 移除模型入口中未被 MUSA KDA/DSA 消费的通用 `build_linear_prefill` chunk schedule；已有 `musa_kda_batch` 继续描述全局边界，避免冗余设备标量回读。

实现复用了已有 MUSA mHC kernel、RMSNorm、DenseMLP、RoutedMoeGate、FP8 Grouped GEMM 和 Shared Expert；没有新增 attention kernel 或主数据 CPU fallback。仓库内参考 MLU `glm5_next_decoder_layer.cpp::forward_pcp` 的 local residual/Norm/Dense 契约，通信仍采用 CP1/CP2 的 Hybrid CP。

## 当前阻塞点和下一步

- CP1/CP2 已通过 CP3 接入模型层间 local rows。CP4 已将 Final Norm 后的 hidden 按 CP0 offsets 恢复为全局 Plain rows，并由两个 `logits` 重载按 global selected indices 进入 LM Head。
- EP dispatch/combine、真实权重首请求和首 token 生成已完成；后续仍需补首 token logits 对齐、图文请求、MTP、状态迁移和正式 `--osl 1` 性能基线。
- CP8+EP8 服务入口、MATE Prefill/Decode、graph capture/replay 和真实 20k 文本请求均已通过；请求关闭 prefix cache，使用 4096×4 + 3616 分块。
- 正式 benchmark 保持 `--osl 1`。当前验证不提供整模型 TTFT 或性能收益结论。
- EP2 通过 `XLLM_GLM5_PREFILL_EP_BACKEND=mate` 显式选择；默认仍为 `eager`，避免未生成 MATE ABI 时静默改变基线。MATE 路径支持 GLM5 的 EP8、`moe_tp=1`、FP8 expert、hidden=4096、intermediate=2048、top-k=8、每 rank 最多 512 个 local rows；Prefill 使用 Stage1/2/3，Decode 使用持久 workspace 的 MATE decode ABI。

## CP1 验证记录

验证日期：2026-11-12；设备：8 × MTT S5000；BF16 输入/权重、FP32 SSM state。参考为相同随机权重和输入的完整 64-head KDA 单层，CP8 各 rank 输出与其对应 token 行比较。

| 验证项 | 结果 |
| --- | --- |
| `xllm` 与 `glm5_next_linear_attention_layer_test` 构建 | 通过；使用 MUSA 构建环境和 ninja guard，额外扫描日志确认无编译/链接错误 |
| 普通 KDA 回归及状态池形状 | 5/5 通过，含 packed Prefill/Decode/reset、Prefill/Decode graph replay 和独立 state head 宽度 |
| CP8，hidden=128，S=1/7/128/129/2048/4092/4096 | 8/8 worker 通过；每种长度均含初始 Prefill、Chunked Prefill 续算和后续首 token 状态读取 |
| CP8，hidden=4096，S=129 | 8/8 worker 通过；64 heads、head_dim=128、非均匀 packed 序列跨 shard 边界 |
| 通信契约 | 每次 CP Prefill 恰好 1 次 AllGather + 1 次 ReduceScatter，0 次 AllReduce；使用真实 MCCL communicator |
| Prefill output 最大绝对误差 | hidden=128：`2.44141e-4`；hidden=4096：`1.46484e-3` |
| conv / SSM state 最大绝对误差 | 两组测试均为 `0`，未使用的 state slot 保持原值 |
| 读取 CP state 后的首 token KDA 输出最大绝对误差 | hidden=128：`1.22070e-4`；hidden=4096：`9.76562e-4` |
| 格式及脚本 | clang-format **20.1.6** 检查、diff whitespace 检查、`bash -n` 均通过 |

首 token 检查仅用现有 TP8 KDA 读取 CP 写入的状态，验证 head owner 连续性；没有新增 CP Decode，也不是首 token logits 验证。该单元验收不包含整模型推理或 `--osl 1` 性能 benchmark。

复现入口：[run_kda_cp_test.sh](run_kda_cp_test.sh)；完整命令和二进制校验值见 [CP1 验证记录](cp1_validation/README.md)。日志和 GTest XML 保存在 `cp1_validation/` 及 `cp1_validation/production/`，由该目录的 `.gitignore` 排除，不纳入源码提交。默认 TCP 端口 29953，每个 worker 超时 600 秒；实际维度测试使用独立端口 29954。测试未停止已有服务，测试耗时不作为性能依据。


## CP2 验证记录

验证日期：2026-11-12；真实 8 卡 MUSA / MCCL。新增 [run_dsa_cp_test.sh](run_dsa_cp_test.sh)，沿用现有 `glm5_next_mla_attention_test` 构建目标。

| 验证项 | 结果 |
| --- | --- |
| `xllm` 和 DSA 测试目标构建 | 通过，构建日志扫描无 compiler/linker error |
| 普通路径 | 15/15 回归通过，含独立数学参考、packed/续算、BF16/FP8 cache、Prefill/Decode graph；原有 2 卡 TP 测试 2/2 通过 |
| CP8 标准矩阵 | 8/8 worker 通过；64 heads、hidden=64、q_rank=32、topk=64；S=1/7/128/129/2048/4092/4096 |
| CP8 模型维度 | 8/8 worker 通过；hidden=4096、q_rank=1536、64 heads、S=129、topk=64 |
| CP8 实际 topk | 8/8 worker 通过；topk=2048，覆盖完整标准长度矩阵及 tail |
| cache / selection | 三组测试均与独立分片投影参考的完整 KV/KPool/tail cache 逐字节一致，候选集合、length、request row 完全一致 |
| output 最大绝对误差 | 相对独立分片投影参考：标准 `0.00390625`，模型维度 `0.015625`，topk2048 `0.00390625`；均通过 BF16 allclose |
| shared selection / 后续首 token | 共享 selection 输出最大误差 `0.00390625`；CP 缓存后续首 token DSA 输出误差为 `0` |
| 静态检查 | clang-format 20.1.6、diff whitespace、两个启动脚本的 `bash -n` 均通过 |

**精度边界：不能宣称与原 full-M 路径逐位相同。** 不同 GEMM 行数会带来极少量 BF16 舍入差异；量化 KPool/top-k 是离散操作，少量字节差异可能改变候选。因此测试增加独立单卡参考：按相同分片投影尺寸生成 Q/key/gate，再用原有完整序列的 IndexPool update 和 attention 计算，不调用 CP 通信或 CP pool update。它用于严格验证 CP cache 和选取语义，输出仍使用原有 `rtol=0.04, atol=0.004`。

同时保留与原 full-M 路径的比较：标准 topk64 的 S=4092 样例有 **4/4092 行**候选集合变化；分片投影参考相对 full-M 的整体输出 RMS 差异为 **0.775599%**、最大绝对差 `0.159912`。模型维度样例无候选变化，最大相对 RMS 为 `0.321977%`；topk2048 矩阵无候选变化，最大相对 RMS 为 `6.82449e-6`。测试对这项比较设置相对 RMS `<2%`、候选变化行比例 `≤1%` 的独立上限。这不是整模型 logits/质量验收，仍需后续真实权重与完整模型验证。

复现命令、日志位置及二进制 SHA256 见 [CP2 验证记录](cp2_validation/README.md)。CP2 单层验收本身不包含整模型服务或 TTFT benchmark；整模型验收见本文档末尾的 EP4 记录。

## CP3 验证记录

- `xllm` 与测试目标构建通过；8 卡混合模型测试 **8/8 通过**，覆盖 S=1/7/128/129/2048/4092、packed Prefill/Chunked Prefill、空 shard、token/embedding 入口和 shared Indexer 无 tail 分配。
- hidden=4096、Dense=12288、Shared=2048 的 FP8 组件测试通过；普通 FP8 MoE Prefill/Decode 和 shared selection 回归 2/2 通过；非法配置测试覆盖 EP8/MoE TP8/MegaMoE 三种拒绝。
- 混合模型默认固定 Indexer scoring，以隔离量化 top-k 离散变化；最大局部相对 RMS `1.38607%`。实际维度 Dense/Shared/最终 mHC 输出最大绝对差分别为 `0.000488281` / `0.000244141` / `0.00390625`。
- **精度限制保留：** 随机 Indexer 的 S=4092 诊断仍有失败，最大绝对差 `0.302734`，某个续算 shard 相对 RMS `5.11638%`。固定 scoring 后通过，差异依赖量化 Indexer/top-k；不能将 CP3 组件验收等同于随机路由、真实权重或完整模型 logits 的精度验收。未放宽测试阈值。

最终二进制复跑为 CP worker 8/8、组件/回归/拒绝检查合计 4/4；clang-format 20.1.6、diff whitespace 和脚本语法检查均通过。

完整范围、精度说明、诊断日志与复现命令见 [CP3 验证记录](cp3_validation/README.md)，八卡入口为 [run_local_cp_test.sh](run_local_cp_test.sh)。整模型服务、EP combine 和 `--osl 1` benchmark 尚未执行。

## CP4 验证记录

- CP1 KDA 与 CP2 DSA 回归脚本在 CP4 改动后仍各 **8/8 worker 通过**。
- CP8 混合模型测试 **8/8 worker 通过**：S=1/7/128/129/2048/4092，初始 Prefill 和 Chunked Prefill，packed requests、token/embedding 两种入口、空 rank；恢复后 LM Head 输出与单卡 global hidden 参考一致。
- 独立 Final Norm/LM Head 测试 **8/8 worker 通过**：真实 MCCL，hidden=32、vocab=17、S=1/7/129，每卡含空 shard 和 16/17 行，global selected rows `{0, floor(S/2), S-1}`；两个 logits 重载以及 selected hidden 均通过。
- restore 使用最大 shard padding 的一次 device AllGather 和 row-offset device copy；不做 host activation 计算，也不对 vocab logits 做 CP reduction。
- CP4 单元验收未覆盖真实 45 层 checkpoint 的 EP8 dispatch/combine、Decode、graph、MTP、KV state transfer 或 TTFT benchmark；这些真实链路在后续 EP3/EP4 验收中补齐。最终测试二进制 SHA256：`04abbb6d5430f0614622e36b33a7d50c319b3b6e6b974b3295fca7dfa7a69d6e6`。

完整实现契约和复现命令见 [CP4 验证记录](cp4_validation/README.md)。

## EP0 / EP1 验证记录

EP0 已把专家 owner 和 MoE TP 的职责分开：`start_expert_id = ep_rank *
(n_routed_experts / ep_size)`，checkpoint 的投影维度仍按 `moe_tp_group_`
的 rank/world 切片。`ep_size=1` 不创建通信路径，也不做 EP AllReduce；当
`moe_tp=1` 时每个本地专家是完整权重。Shared Expert 在 CP/EP Prefill 下
使用 single-rank group，避免 TP8 的错误切分和重复归约。模型构造阶段现在
允许 EP0 的 `ep_size=1`，EP>1 则要求 `moe_ep_group_` 的 world size 与
`ep_size` 相等，并要求 `moe_tp=1`。

EP1 当前是 eager correctness path，使用 `ProcessGroup::all_to_all_single`：

- local `[T, topk]` route 在 MUSA 上 flatten、按 owner 排序并打包 hidden、全局 expert id、source row 和 weight；不把 per-token route id 搬到 host；
- 先交换 device counts，再做 dispatch All-to-All；目标 rank 按本地 expert mask 计算，FP8 使用已有 grouped GEMM，dense/packed storage 使用同一 local expert fallback；
- 反向 All-to-All 后按 source row 在设备上 weighted `index_add`，EP 路径没有输出 AllReduce；
- 空 rank 仍参加所有 collective，覆盖非均匀 token 数和 0 行 split；当前 split count 的标量读取仅为适配 `all_to_all_single` 的 host vector API。

真实 8 卡 MCCL worker：

| 验证项 | 结果 |
| --- | --- |
| 构建 `glm5_next_mla_attention_test` / `xllm` | 通过，构建日志无 compiler/linker error |
| EP8 对 EP1 serial reference | 8/8 worker 通过 |
| local token 数 | `0`、非均匀 `1/2/3`、`7`、`33`，rank7 全空 |
| route | 8 experts、top-k=2，跨 owner、重复 route、weighted combine |
| CP token-shard entry | 8/8 worker 通过，`RoutedMoeTokenShard` 与 CP8 local rows 同时启用 |
| 通信 | 每个 shape 所有 rank 均参与 counts/dispatch/combine；无 EP output AllReduce |
| 复现 | [run_ep_test.sh](run_ep_test.sh)，记录见 [ep1_validation](ep1_validation/README.md) |

这项验收只证明 EP dispatch/combine 的语义和空 shard 行为。它使用小型
BF16 dense 权重隔离通信正确性，尚未宣称真实 288 expert FP8 checkpoint、
DeepEP/MATE 高性能 kernel、CUDA graph、45 层 logits 或 `--osl 1` TTFT
收益；这些属于 EP2/EP3 和最终 benchmark。

## EP2 实现记录

EP2 当前选择已安装并由 `compile_musa_mega_moe.py` 导出的 MATE Prefill ABI，
不假设 Python DeepEP 包提供可链接的 C++ 接口。`MusaMegaMoe` 增加了 local-token
入口：CP/EP rank 不再把完整复制输入按 rank 再切一遍，而是直接将自己的
`[local_rows, 4096]`、top-k expert ids 和 weights 交给 MATE Stage1；Stage2 在
owner rank 执行 grouped FP8 gate/up/down GEMM，Stage3 按 source row combine，
输出仍为本地 rows。空 rank 也进入同一组 MATE collective schedule。

MATE 权重使用 EP owner 的 36 个完整专家；gate/up 权重和 block scale 在加载时
转换为 MATE interleaved layout。旧的 `MusaMegaMoe` 复制输入路径保持不变，
EP1 `eager` 路径也保持可选。启动 MATE EP2 前必须生成
`glm53_mega_stage{1,2,3}_v1.so`，并设置：

```bash
GLM5_EP_SIZE=8 XLLM_GLM5_PREFILL_EP=1 \
XLLM_GLM5_PREFILL_EP_BACKEND=mate XLLM_GLM5_MEGA_MOE=0 \
tools/musa_glm5_3_vl_bench/start_service.sh start
```

本轮完成 `xllm` 编译、shell 语法检查，并用真实 GLM-5.3-Flash 8 rank 模型启动 CP8+EP8 MATE 服务；一次文本请求完成 Prefill 和 Decode，HTTP 返回成功。该请求只作为链路验收，不把短输出的内容质量或 TTFT 收益作为性能结论。

## EP3 / EP4 实现记录

EP3 已把 EP dispatch/combine 放入 `Glm5NextDecoderLayer` 的实际 Prefill/Chunked
Prefill 调用链。CP rank 先完成本地 mHC、attention、RMSNorm 和 Gate，随后 MATE
只接收本 rank 的 local rows；Shared Expert 在独立 stream 上提前计算，主 stream
等待 event 后与 routed combine 结果相加一次。空 rank 仍执行 MATE 的固定 workspace
和 collective schedule，不通过提前 return 跳过通信。

EP4 的 MATE workspace 在 `MusaMegaMoe` 构造阶段一次性分配，包括 IPC buffer、
metadata views、FP8 intermediate 和 `[512, 4096]` local output；forward 不再为
每层动态申请 local output。MATE graph capture 不再被 RoutedMoe 主路径拒绝。
CP8 的 KDA/DSA layout 在 capture 中使用设备侧 arithmetic metadata、固定 CP8
AllToAll counts 和持久 output contract；host validation 仅保留在 eager 路径。
启动 CP8+EP8 时保持 `GLM5_ENABLE_GRAPH=true`，并使用
`GLM5_ENABLE_CHUNKED_PREFILL=true`、`GLM5_MAX_TOKENS_PER_CHUNK=4096`。

## EP3/EP4 真实链路验收（2026-11-13）

| 验证项 | 结果 |
| --- | --- |
| `xllm` / `glm5_next_mla_attention_test` 构建 | 通过；clang-format 20.1.6、`git diff --check` 通过 |
| EP1 8 rank eager dispatch/combine worker | 8/8 通过 |
| GLM5 CP8+EP8 MATE 服务启动 | 8/8 rank 完成真实 62 分片权重加载，MATE EP2 日志确认 owner rank 0/8…7/8 |
| 真实文本请求 | HTTP completion 成功；Prefill 使用 CP8+MATE EP，Decode 使用 MATE decode workspace |
| CP8 graph | 通过；4096 分块和 3616 尾块均完成 capture/replay，MATE FFI scratch tensors 固定为 44 个 |

## Graph CP8+EP8 真实 20k 验收（2026-11-14）

| 验证项 | 结果 |
| --- | --- |
| 配置 | VLM engine、CP8、EP8、MATE、graph、chunked prefill、prefix cache=false |
| 请求 | 1 个纯文本请求，20,000 input tokens，1 output token |
| 分块 | 4096、4096、4096、4096、3616；每个 shape 均完成 graph capture/replay |
| HTTP | 首次请求成功，TTFT 9824 ms，server_errors=0；同形状第二次 replay 成功，TTFT 1516 ms，server_errors=0 |
| KV | prompt_tokens=20000，cached_tokens=0，kv_hit_rate=0 |
| 日志 | 首次请求 `lazy_captures=32`，第二次请求 `lazy_captures=0`；每个 CP rank 均记录 capture end，MATE Stage1/2/3 无异常 |

该验收证明 graph 模式 CP8+EP8 Prefill 的端到端链路，不代表最终吞吐或
质量基线。图文输入、并发请求、MTP、PD 状态迁移和性能调优仍需单独验收。
