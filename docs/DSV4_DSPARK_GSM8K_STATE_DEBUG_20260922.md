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

最终候选已将 Eager q2 设为默认准确率保护路径；Graph q2 只有通过
`XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY=1` 才显式 opt-in，仍保留用于性能实验。
使用当前工作树改动构建的独立候选
`build/gsm8k_candidates/dspark_accuracy_default_eager_verified_20260922/xllm`
实际命中 `[DSV4_GRAPH_ROUTE] target compact verify forced eager`，固定 100 题
结果为 **95/100（95%）**，失败 `235,255,333,952,1038`。该结果完成了当前
准确率门槛，但 Eager q2 会降低长序列 decode 性能；后续仍需在 Graph q2 修复
后重新恢复 Graph 并复测性能。

## 2026-09-22 Graph+DSpark 长短序列性能复测

按历史高性能口径启动当前候选，并显式设置
`XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY=1`，确保 target compact verify 不回退到
默认 Eager。历史高性能口径使用 DSpark `N=5`，因此对应 target q6，并不等于
真正的 q2。

- 模型：`DeepSeek-V4-Flash-0731-FP8-mt`；MUSA TP8/EP1/DP1/CP1；B1；
  DSpark `N=5`；Graph/VMM/decode-no-padding 开启；Prefix Cache 开启，
  `--prefix-cache-rate=0.65`；overlap 关闭。
- 输入/输出：2500/1500；`count-to=750`；先执行两次 64/64 warmup，
  正式窗口为 variant 0--7，热态窗口原计划为 variant 2--7。
- 二进制：`build/gsm8k_candidates/dspark_accuracy_default_eager_verified_20260922/xllm`。
- 日志：`logs/repro_graph_q2_current_20260922_retry1/`。

两次 warmup 均严格通过，但 N=5 长序列没有形成有效性能窗口：

| 请求 | TTFT | TPOT | 结果 |
|---|---:|---:|---|
| warmup 1，64/64 | 8007.4 ms | 56.724 ms | 32/32 PASS |
| warmup 2，64/64 | 586.8 ms | 39.554 ms | 32/32 PASS |
| variant 0，2500/1500 | 487 ms（服务端记录） | 690.1 ms（仅生成631 token） | 客户端 180 s 超时 |

variant 0 服务端在约 435 秒后仍只生成 `631/1500`，因此不能计算合格的
TTFT/TPOT/TGS；variant 2--7 未执行。该结果只能说明当前 Graph+DSpark 长序列
存在严重 decode 卡滞，不能单独归因于 q2。

### 1024/512 短序列对照

为区分 DSpark N=5 与真正 q2，使用相同 TP8/EP1、B1、Prefix Cache 0.65、
Graph 配置，分别测试 4 个 1024/512 请求，严格数到 256：

| Graph 路径 | TTFT_AVG | TPOT_AVG | Input TGS | Output TGS | Total TGS | 严格正确 |
|---|---:|---:|---:|---:|---:|---:|
| DSpark N=5（target q6） | 460.138 ms | 49.707 ms | 39.597 tok/s | 19.799 tok/s | 59.396 tok/s | 4/4 |
| DSpark N=1（target q2） | 449.374 ms | 98.971 ms | 20.069 tok/s | 10.035 tok/s | 30.104 tok/s | 4/4 |

N=1 的每个请求均生成 `512/512`，cached tokens 为 variant 0 的 0 和后续
variant 的 640；但 q2 Graph 的 TPOT 约 99 ms，仍明显慢于历史高性能路径，
所以短序列也不能证明 Graph q2 已恢复高性能。日志分别为
`logs/repro_graph_q2_short1024_20260922_retry1/` 和
`logs/repro_graph_q2_short1024_n1_20260922/`。

上表统计全部 variant 0--3，包含首条缓存未命中和首次缓存命中请求，不能称为
纯热态均值。单列最后两条 variant 2--3 作为短热态观察窗口（不是历史六请求
窗口的等量复现）：

| Graph 路径 | 热态 TTFT_AVG | 热态 TPOT_AVG | Output TGS | Total TGS | 热态样本 |
|---|---:|---:|---:|---:|---:|
| N=5，target q6 | 189.391 ms | 49.805 ms | 19.969 tok/s | 59.906 tok/s | 2/2 |
| N=1，target q2 | 189.402 ms | 99.092 ms | 10.074 tok/s | 30.221 tok/s | 2/2 |

TGS 均为整套 TP8 服务总 token 数除以所列请求 E2E 耗时之和，未除以卡数，
不包含独立客户端进程之间构建 prompt 的间隔；输入包含缓存命中的 token。
实际缓存命中 640/1024=62.5%，0.65 为配置的前缀比例。TPOT 使用客户端
首个至最后一个非空 content 的间隔除以 usage completion_tokens-1。
两个热态窗口的总 E2E 分别为 51.280097 s 和 101.650776 s。

N=1 rank0 日志实际记录 `bucket_num_tokens: 2, actual_num_tokens: 2` 的 Graph
capture 完成，没有命中 `target compact verify forced eager` 路由日志；N=5
记录的是 bucket 6，故不再将前一轮 N=5 长序列超时称为 q2 专属问题。
本轮复用原独立二进制，SHA256 为
`4c24d92f154843526fdde51ac709ff798f46ba39a5f422f90add511765aea5d4`，未修改
推理代码或重编共享库。短计数全部通过不证明 GSM8K Graph+draft 精度已经恢复。

N=5 目录的 `window_0_3.summary.json` 是旧汇总器因只支持六请求 2500/1500
窗口而产生的参数校验失败，不是模型结果失败；该文件保留，不覆盖。上述短序列
统计从四份 `formal_variant_*.json` 及逐条 accuracy gate 重新计算。

## 2026-09-22 GSM8K 93% 目标与选择性性能保护

本轮目标是固定 100 题准确率至少 93%，同时尽量恢复 q6 Graph 性能。新增的
Attention 改动只对 `compact_spec_verify && is_acl_graph && total_tokens > 2`
冻结 compressor/indexer 的 persistent state 更新，覆盖 DSpark `N=5 -> target q6`；
q2 不改变，仍由 `musa_graph_executor_impl.cpp` 的默认 Eager accuracy route
保护。这样没有把 q2 未闭环的 Graph state 直接重新开放到默认路径。

### 固定 100 题精度

最终精度候选：
`build/gsm8k_candidates/selective_q6_freeze_verified_runtime_20260922/xllm`，
SHA256=`58cb6d2e1dd6668a02c1914e3551a2fb97d6fca2981cf9c462397241e20004b9`。
该候选包含当前 Attention 和当前 Graph runtime，启动日志确认每个 target q2
命中：`[DSV4_GRAPH_ROUTE] target compact verify forced eager; tokens=2`。

固定同一组 100 题、DeepSeek-V4-Flash-0731-FP8-mt、TP8/EP1/DP1/CP1、Graph=true、
DSpark `N=1`、Prefix Cache=false、reasoning prompt、temperature=0 的结果：

| 正确 | 错误 | 准确率 | 请求错误 | 失败题号 |
|---:|---:|---:|---:|---|
| 95 | 5 | **95.0%** | 0 | 235, 255, 333, 952, 1038 |

原始结果：[eval100/summary.json](../logs/goal_selective_accuracy_verified_20260922/eval100/summary.json)。
这满足本轮 `>=93%` 目标；剩余错误属于 draft/算术/语义路径，未通过改变 q6
Graph state 直接修复。

### q6 Graph 热态性能

使用同一最终候选、TP8/EP1、B1、2500/1500、Prefix Cache 0.65、Graph+DSpark
`N=5`、热态 variant 2--7，6/6 严格 `1..750` 通过：

| TTFT_AVG | TPOT_AVG | 严格正确 | cached tokens |
|---:|---:|---:|---:|
| **215.813 ms** | **30.0145 ms** | 6/6 | 1536 |

结果目录：[repro_final_q6_retry_20260922](../logs/repro_final_q6_retry_20260922/)。
相同当前分支、未冻结 q6 state 的短热态对照约为 TPOT 49.7 ms，因此本改动保留了
约 40% 的 q6 decode 时间；历史旧二进制约 8 ms 的记录仍未恢复，不能把本轮
30 ms 宣称为历史最优。

### 重要边界

- 当前默认精度路径是 q2 Eager，q6 性能路径是 Graph state-freeze；两者不是
  同一个请求形状，不能合并成“Graph+draft 全路径同时 95% 且 8 ms”。
- q6 state-freeze 尚未完成固定 100 题的 Graph+draft GSM8K 精度验收，因此默认
  生产精度保护仍以 q2 Eager 为准。
- `tools/build_gsm8k_isolated_candidate.py` 现在只要重编任一 DSpark 核心源文件，
  就自动把 Attention、DSA metadata、Graph executor、DFlash worker 四个源一起
  纳入候选，避免 Attention 新代码与旧 runtime 归档混链；单元检查覆盖去重和
  非 DSpark 源文件不扩展。
- 测试结束后 xLLM 已停止，GPU 0--7 由占卡程序 PID `41030` 保留，每卡约 70 GiB。
