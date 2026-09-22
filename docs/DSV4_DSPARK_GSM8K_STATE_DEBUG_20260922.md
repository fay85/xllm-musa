# DeepSeek-V4 DSpark GSM8K 状态/Graph 回归记录（2026-09-22）

## 目的与固定口径

- 模型：`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`
- TP8/EP1，MUSA，`max_memory_utilization=0.80`，`max_tokens_per_batch=2048`
- GSM8K parquet：`logs/gsm8k_eval_20260914_1925/gsm8k_test.parquet`
- 固定 20 题序列：`3,23,26,50,64,71,75,122,125,128,130,133,184,195,201,206,216,235,244,255`
- 固定 100 题序列：见 `logs/goal_graph_q2_eager_diag_full100b_20260922/eval100/run_config.json`
- 评测不读取 reference 参与生成，只在最后做数值 exact-match。

## 首次分叉定位

1. 单独启动 Graph+DSpark 请求 `index=128` 为 `880`；同一服务先跑前序请求后再跑 `index=128`，输出变为 `305`。因此不是固定 prompt 本身的不确定性。
2. Graph=true、draft=false 的控制组中，128/206/244 恢复正确，235 仍是语义题错误；说明主要分叉在 target q2 Graph 复用，而不是 q1 draft layer 或普通 prefill。
3. Graph+draft 默认候选 `graph_c4_final9_20260922/xllm` 固定 20 题为 **16/20 (80%)**，失败 `128,206,235,244`。
4. `XLLM_MUSA_GRAPH_SYNC_INPUTS=1` 和 `XLLM_MUSA_GRAPH_SYNC_OUTPUTS=1` 均不能消除 128，排除了单纯 caller/capture stream 同步竞态。
5. metadata 日志显示每个新请求都重新进入 `prefill=1, q_len=kv_len`；因此 sidecar/reset 已生效，剩余问题是 target q2 Graph 的 persistent DSA/accepted-token carrier 与 eager q2 不等价。

## 已做代码修复/控制项

- `dsa_metadata_builder_musa.cpp`：Graph/Eager 均生成 CPU `__musa_dsv4_host_sequence_lengths` sidecar，使 Attention 能识别 chunked prefill 的新请求起点。
- `deepseek_v4_attention.cpp`：新请求起点清理 compressor/indexer history；Graph compact q2 使用无分配 state workspace；保留 `XLLM_DSV4_DISABLE_GRAPH_COMPACT_INDEXER=1` 作为 indexer 对照开关。
- `dflash_worker_impl.cpp`：prefill 发布 target embedding 前清理当前 embedding id，避免无 request-id 时的旧尾状态残留。
- `musa_graph_executor_impl.cpp`：增加 `XLLM_DSV4_FORCE_EAGER_COMPACT_VERIFY=1` 控制项，只将 target q2 走 eager，draft q1 仍可走 Graph，用于隔离 Graph q2 问题；默认值仍未改变。
- `dflash_worker_impl.cpp`：增加 `XLLM_DSV4_FORCE_TARGET_TOKEN_ONLY=1` 对照项，运行 draft/target forward 但只提交 target 第一行，隔离 accepted-token 语义漂移；默认关闭。

## 对照结果

| 配置 | 结果 | 日志 |
|---|---:|---|
| Graph+DSpark 默认（q2 Graph）固定 20 | 16/20 = 80% | `logs/goal_graph_final9_20260922/eval20/summary.json` |
| Graph+target-only，q2 Graph | 3/4（128/206/244 对，235 语义错） | `logs/goal_graph_final8_targetonly_20260922/eval4/summary.json` |
| Graph+DSpark，target q2 eager 对照，固定 20 前 10 | 10/10 | `logs/goal_graph_q2_eager_diag_20260922/eval_prefix128/summary.json` |
| Graph+DSpark，target q2 eager 对照，固定 100 | **94/100 = 94%** | `logs/goal_graph_q2_eager_diag_full100b_20260922/eval100/summary.json` |

q2 eager 对照的剩余失败为 `130,201,333,570,952,1038`，内容检查显示主要是算术/语义解释分叉；其中 130/201/333 是 draft 路径相对 target-only 新增的语义漂移，570/952/1038 在 target-only 也可能出现，不能归因于 Graph q2 kernel。

## 当前结论与下一步

- **已定位**：默认错误集中在 target q2 Graph 的跨请求 persistent DSA/accepted-token 路径；q2 eager 对照可以恢复 128/206/244，证明不是 GSM8K evaluator 或普通 MATE 单算子错误。
- **尚未达成**：默认 Graph+draft 尚未完成固定 100 题 >=95%，不能把 q2 eager 的 94% 宣称为最终通过。
- 下一步应在不改变默认 Graph 性能的前提下，对同一请求的 target q2 Graph/eager 逐 layer 对比 `q/kv/slot/block/state`，重点是 q2 第一行的 accepted-context 与 C4/SWA state；然后再做无 reference 的 arithmetic/semantic audit，处理剩余 6 个语义 case。

## 2026-09-22 accepted-state 回写实验

尝试在 q2 前保存 compressor/indexer state，并在 draft 被拒绝后的下一次 q1
恢复；该候选在固定 20 题得到 **17/20**，但首题从基线 540 变为 60，且
128/255 也出现新错误。说明当前 snapshot 的 slot 生命周期尚未正确闭环，
代码未合入 `cde77311`，只保留为失败实验候选，不影响默认二进制。

候选构建目录：`build/gsm8k_candidates/dsv4_state_restore_diag5_20260922/`。
该实验同时确认：简单地在下一步 scatter 旧 state，不能替代 target q2 对
rejected proposal 的正确回滚。

## 2026-09-22 eager q2 causal A/B

将单序列 q_len>1 的 target verify 强制视为 causal fallback，禁止多行 MATE
sparse 路径。该修改修复了 q2 eager 对照中的部分语义漂移（130、201），但
固定 20 仍为 **16/20**，失败仍集中在 `128,206,235,244`；因此它不是完整的
Graph/state 修复，默认高性能路径仍需继续定位。

短答案 prompt 对照更差（6 题仅 1/6），说明问题不能通过缩短输出提示解决。

## 2026-09-22 SWA block-table 追踪

对固定前序请求开启临时 `XLLM_DSV4_LOG_COMPACT_META=1` 后，target q2 Graph
在跨过 SWA block 边界时观察到 SWA manager 从 `[1,2]` 变为 `[-1,2,3]`，而
C4/C128 manager 仍为 `[1]`。这确认 128/206/244 的首次分叉位于 rolling SWA
block-table / compressed attention 读取交界，而不是 q2 shape 本身；简单把
Graph q2 改成 eager 或关闭 indexer 不能稳定修复。

临时 full-dense SWA A/B 能让 index=128 正确，但会使 122/125 错误，因为跳过
压缩 cache 构建会破坏后续长 decode state；该 A/B 未合入默认代码。

## 2026-09-22 MATE `mate_sparse_attention` 独立单测

为区分 MATE kernel 数值问题和上层 Graph/state 输入问题，新增
`tests/core/kernels/musa/deepseek_v4_mate_sparse_attention_test.cpp`，并用独立
构建脚本 `tools/build_mate_sparse_unit.py` 复用当前 `libcuda_kernels.a`，不改动
主 xLLM 构建产物。测试使用真实 MODEL1 FP8 page layout（每页 stride 按 576
字节对齐），先调用 `deepseek_v4_pack_fp8_cache`，再将 packed cache 解码成
同一份 K/V 做普通 BF16 dense reference。

覆盖三类输入：

1. q1 scalar：8 heads、top-k 8、SWA extra 128；
2. q2 两行：两行使用不同 main/extra 物理索引，验证 batched row 独立性；
3. q2 rolling SWA 边界：`compressed block table=[1,2]`、`SWA table=[-1,2,3]`、
   `kv_len=258/window=128`，通过 `deepseek_v4_prepare_mate_dsa_indices` 验证
   第 439/440 类边界的物理映射。

运行命令：

```bash
env MUSA_VISIBLE_DEVICES=0 \
  FLASHINFER_OPS_PATH=/data/nfs_shared/jiaming.wang/test/workspace_fp8_28/cached_ops \
  LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/musa/lib:/usr/local/mudnn/lib \
  build/mate_sparse_unit_20260922_d/mate_sparse_test --gtest_color=no
```

结果：**2/2 passed**。

| case | max abs | mean abs |
|---|---:|---:|
| q1 scalar | 0.0078125 | 0.000601768 |
| q2 batched rows | 0.0078125 | 0.000497341 |
| q2 `[-1,2,3]` SWA boundary | 0.0078125 | 0.000286102 |

因此，在真实 page stride、packed FP8 cache、q1/q2、rolling SWA table 这几个
受控条件下，MATE sparse kernel 与同一 packed cache 的 dense reference 对齐；
目前没有证据表明 `mate_sparse_attention` 本身是 GSM8K 错误的首因。该单测不
覆盖 Graph persistent buffer、accepted-token 回写、跨请求 state 生命周期，所
以不能替代固定 100 题 Graph+draft 回归。

## 后续定位准备：accepted-state 日志（未运行验证）

新增默认关闭的 `XLLM_DSV4_LOG_ACCEPTED_STATE=1`，记录 decode 的输入 token、
位置、KV 长度、embedding state offset，以及 accepted token 对应的 context
写回位置和物理 slot。独立候选
`build/gsm8k_candidates/accepted_state_trace_20260922/xllm` 已编译链接成功，
SHA256 为 `ec9d9e59e58bfac7b15dc5146822f09c22bf1ed17bb51525ed72571dbbd6be93`。
本候选继承了未验证的 SWA 实验；尚未启动，不作为新准确率或修复证据。
候选中一处 layer0 indexer 日志在构建后从工作树移除（layer0 为 ratio=1，
不会经过该分支），精确源码以候选目录 `source.diff` 和快照为准。

下一步需固定 token 前缀，分别在相同 prefill/cache 状态下比较 Eager target
与 Graph target 的 q2 第0行，保存 Graph replay 后的设备诊断 buffer（不能在
capture 内做 D2H/item 日志）。在首个 logits 分叉 step 内，再逐层比较 hidden、
Q/KV、SWA/C4/indexer state 和 Attention 输出。accepted 数量本来可能因草稿
不同而变化，应按绝对 token 位置对齐，不能按外层迭代序号直接断言状态错误。
现有三个 MATE 输入 case 通过并未证明 Graph replay 数值、完整索引范围、真实
失败激活都正确；`[-1,2,3]` 本身也可以是正常逻辑页淘汰表示，并非根因证明。

## 2026-09-22 真正 Eager q2 结果与临时默认策略

为避免把旧候选误称为 Eager，`musa_graph_executor_impl.cpp` 增加了命中日志
`[DSV4_GRAPH_ROUTE] target compact verify forced eager`。使用包含该 runtime
对象的独立候选 `build/gsm8k_candidates/route_accurate_ab_20260922/xllm`，
固定 100 题真正走 Eager q2 的结果为 **95/100（95%）**，失败为
`235,255,333,952,1038`。这五题是当前 draft/语义路径问题，原先 Graph q2
跨请求污染的 `128,206,244` 不再出现。

随后确认清空 compressor/indexer state cache、清空 decode workspace、关闭
compact indexer，以及给 Graph metadata 增加 producer→capture event，均未让
默认 Graph q2 的 index=128 从 305 恢复到 880。因此没有把这些诊断项直接合入
默认性能路径。

当前工作树暂不把 Eager q2 默认切换提交为生产修复；Graph q2 仍是默认路径，
Eager q2 只能通过 `XLLM_DSV4_FORCE_EAGER_COMPACT_VERIFY=1` 做控制对照，
Graph q2 可通过 `XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY=1` 显式控制。显式
Eager 控制候选的固定 100 题结果是 95/100，但默认策略候选尚未完成同口径
固定 100 题复验，目标仍未闭环。
