# DeepSeek-V4 运行开关与路由清理（2026-10-29）

## 核查结论

- `attention.cpp`、`moe.cpp`、`gate.cpp`、`decoder_layer.cpp` 中原来大量在 forward/逐层路径调用 `getenv()` 的问题存在。本轮将 DeepSeek-V4 MUSA 开关统一收口到 `deepseek_v4_runtime_options.h`，每个枚举项使用函数内 `static const` 缓存，进程生命周期内只读取一次。
- 保留未设置、空串、`0`、`1` 的区别；环境变量仍是启动时配置，运行后修改环境不会改变已经缓存的开关。
- `moe.cpp` 的 `if (!output.defined() && use_fp8_gemv)` 首个条件确实恒真，已改为 `if (use_fp8_gemv)`。
- shared expert 串联路径存在局部变量遮蔽，已拆成 `forward_fp8_gemv()` 和 `forward_reference_experts()`，并修复 shared output 局部命名。
- `must be77 or112` 已修正为带空格的错误信息。
- `world==8`、`rows` 范围、4096 hidden size、77/112/128/7 等约束集中到 `deepseek_v4_execution_limits.h`，没有放宽已验证 kernel 的适用范围。
- Prefix snapshot 的 80% 已改为启动开关 `XLLM_DSV4_PREFIX_SNAPSHOT_PERCENT`，默认仍为 80；只接受 1～99 的整数，并按 base block 对齐。 `20000/128` 默认仍得到 16000，设置 65 时得到 12928。
- Attention 主函数已将输出投影/collective 尾段抽到 `project_attention_output()`；MoE 的 GEMV、reference routing 也已抽出。`forward_current_impl` 仍然较长，完整的路由状态机拆分需要独立变更和更大回归矩阵。
- Decoder 的两套 Attention 不能直接全部删除：普通 Attention 使用 TP 分片，Prefill-SP Attention 使用单 rank 完整头布局。当前只对 DSpark draft（明确禁止 Prefill-SP）不创建第二套 Prefill Attention，减少其权重句柄和 workspace；target/Prefill-SP 仍保留两条路径。

## 验证

候选源码基线：`c829f0c7047a25035d9923eb34489690999b7955`。A/B 表中的 baseline 服务是原 r14（二进制由前一轮新镜像源码构建），不是对 c829f0c70 整包重新部署的性能结果。运行拓扑 P94/P96 + D90，测试时 Control 保持原服务。

单元测试：

- `deepseek_v4_runtime_options_test`：5/5 通过；
- 覆盖 env 缓存、空值/0/1 语义、线程内同一缓存地址、Prefix 比例解析和溢出边界、ReduceScatter/Grouped FP8 形状约束；
- 上一轮 `deepseek_v4_compressor_state_test` 仍为 5/5。

独立候选：

```text
/data/nfs_shared/jd/dsv4_review2_20261029/bin/prefill/xllm
/data/nfs_shared/jd/dsv4_review2_20261029/bin/decode/xllm
```

P/D 均通过编译、链接、`ldd -r` 和 `--help`。

冻结同一组 20K/100、80% prefix、cached_tokens=16000 输入：

| 项目 | baseline | runtime-option 候选 |
|---|---:|---:|
| B1 严格请求 | 64/64 | 64/64 |
| B2 严格请求 | 64/64 | 64/64 |
| B1 TTFT 平均 | 539.569 ms | 539.949 ms |
| B1 TPOT 平均 | 9.806 ms | 9.621 ms |
| B2 TTFT 平均 | 538.370 ms | 541.281 ms |
| B2 TPOT 平均 | 11.721 ms | 11.631 ms |
| GSM8K 固定 20 题 | 20/20 stop，正文一致 | 20/20 stop，正文一致 |

这批有限样本未观察到精度回退；B1/B2 TPOT 分别减少约 1.90%/0.76%，TTFT 分别增加约 0.07%/0.54%。单组先后 A/B 不足以判断因果或统计显著性。该窗口不是全量 GSM8K 或长稳验收。

第一次候选请求在服务未就绪时失败，原始记录保留于 `smoke_candidate/`。确认各角色启动并重新注册后，在 `smoke_candidate2/` 完整重跑，才得到上表；没有覆盖第一次失败记录。

测试二进制 SHA256：

- P：`28664abbcb46bb6c2e75d34d734f80e8fc7a9159b4fab5af816d8d4e2b5be58e`
- D：`9eca3c797eb5f0335f7ab2180517181f81320a7f5adfbf84dbb94ba17c87acc3`

## 可复现证据

```text
/data/nfs_shared/jd/dsv4_review2_20261029/
  build_r1.log
  prefill_r1/
  decode_r1/
  unit.log
  smoke_baseline/
  smoke_candidate2/
  runtime/
  restored/
```

验收后曾按原 r14 计划启动恢复服务，未 reset GPU，也没有停止其他容器或 Control。但随后检查发现并行任务已在同一机器启动 `/data/nfs_shared/jd/feihu_89e8b415_20261029/bin/decode/xllm`；恢复日志存在 dispatcher schema 错误，24888 曾无法连接。因此不能将当前状态描述为“r14 已稳定恢复”；不继续操作并行任务的进程，等待协调。

## 并行提交与发布门禁

- 本轮候选修复提交：`2b66eece4`，在上述独立构建及有限样本中通过。
- 测试期间远端推进到 `89e8b4153`，涉及 100 个文件，包括共有通信、Graph、请求及 PD 代码。
- 本地已无冲突合并为 `06dae67c0`，保留双方历史；这不等于整合后的二进制验收通过。
- 当时未将合并结果推送到 feihu-1016-dev；获得接管授权后，继续完成下文最新合并版验收，再发布。
- 旧 refine 工作树仍有 21 个已跟踪文件未提交修改。它属于旧实验状态，本轮没有自动提交或覆盖；旧全量 GSM8K 成绩也没有转移为新候选成绩。

## 最新远端合并版验收

用户确认另一任务已结束并授权接管后，冻结源码提交 `5a54ef075fbe64ddf8effa2b3de730060f4f4022`（含 `89e8b4153` 与本轮修复），使用其最新 P/D 构建缓存重编本轮修改及头文件消费者，复制受影响静态库后独立链接。所有未修改依赖复用，没有覆盖其他任务 source/build。

独立证据目录：`/data/nfs_shared/jd/dsv4_review2_integrated_20261029`。

### 必须保留的配置边界

另一任务验证 `89e8b4153` 的 Decode VMM + 内置 Custom AR IPC 组合在 B16 动态捕获时会失败。本次使用其通过的绕行配置：

- Decode：`--enable_graph_vmm_pool=false`；Graph 与 overlap 仍开启。
- 使用新版内置 Custom AR，P/D 均不加载旧 `libxllm_custom_ar_native.so`。
- Prefill VMM 保持原值；AOT、模型、Control 沿用 r14 资产。
- 此处没有修复底层 VMM/IPC，也没有验证 Decode VMM=true 可运行。

### 构建与身份

- P SHA256：`934a83264d9a55a9a7a00b83392f8d3ff42b2d2793fc469327f29af500fe1189`。
- D SHA256：`c652e080224329f41c9ae5b49424a1f30f6caa6c746d1dfa545d9d32c9f9abb6`。
- 3159 个源码文件与 Git archive 逐字节比较，0 不一致；archive SHA256 为 `b01f0e9851991952c75005f3409f39548f9b392387984180d074ca939969e052`。
- P94/P96/D90 共 24 个 rank 的 exe、启动 ticks、argv、Graph/VMM 与 preload 实际环境检查通过，maps 和核查 JSON 在 `runtime/<role>/`。
- 10 项 CPU 测试再次通过；P/D 编译、链接、ldd/help 通过。该构建复用依赖，不是完全无缓存 clean build。

### 正确性与性能

新的同输入顺序 A/B（baseline 为恢复后的 r14；candidate 为最新合并版 + D VMM off + 内置 AR）：

| 并发 | 正式请求 | r14 TTFT ms | 合并版 TTFT ms | r14 TPOT ms | 合并版 TPOT ms | 合并版输出 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 64 | 530.964 | 532.863 | 9.547 | 9.125 | 69.551 |
| B2 | 64 | 534.174 | 533.725 | 11.493 | 10.680 | 124.102 |

两版均 132/132 严格通过（含 4 条预热）；正式请求全部 20000 输入、100 输出、cached_tokens=16000。输入文件 SHA256：`4baff1c187bd8758e1b69685473dfbbe4a27c8c5180c0b6a6409b02125161afa`。
这次同时合入了上游通信等改动并改变 VMM/AR 配置，因此不能把 TPOT 差异单独归因于 getenv 缓存或本轮清理。

额外 B16 短窗使用另一组冻结输入：primer 2/2、B1 warm 2/2、B1 正式 6/6、B16 warm 16/16、B16 正式 16/16，共 42/42 严格通过。
正式 B16：TTFT 2279.227 ms，TPOT 20.450 ms，输出 267.916 tok/s。预热 TPOT 119.972 ms 包含动态 Graph capture，未作为稳态性能。

GSM8K 前 20 题：20/20 正常 stop，boxed 答案与原 r14 全部一致；正文与原 r14 仅 4/20 相同，但与另一任务同配置 `89e8 + D VMM off` 的 20 条正文全部逐字一致。不要将此标为全量 GSM8K 准确率。

主要证据：`build.log`、`source_integrity.json`、`results.json`、`smoke_baseline/`、`smoke_candidate/`、`b16/`、`unit.log`。

### 非默认快照比例

`XLLM_DSV4_PREFIX_SNAPSHOT_PERCENT=65` 的解析与边界算术已覆盖单测，但本次端到端只验收默认 80%。修改比例须对参与 Prefill 的进程统一设置并重启，再检查 state snapshot/cache 命中和正文正确性；不能只改压测客户端的比例。

## 未在本轮完成的项目

- `forward_current_impl` 仍需按路由状态拆分为多个私有函数，并对 Graph/PCP/DSpark 组合做完整矩阵；
- prefill/decode Attention 的 workspace 共享需要先证明生命周期、Graph capture 地址和 checkpoint loader 兼容，暂不继续冒险删除；
- 67 个环境开关中仅 DeepSeek-V4 MUSA .cpp/MUSA decoder/gate/MoE 路径已统一；独立 header-only policy 的 getenv 将在下一轮统一接入相同 accessor，避免超大机械 diff。
