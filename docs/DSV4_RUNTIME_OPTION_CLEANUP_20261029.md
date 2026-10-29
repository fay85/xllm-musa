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
- 未将合并结果推送到 feihu-1016-dev，需与并行任务协调后验证最新基线再发布。
- 旧 refine 工作树仍有 21 个已跟踪文件未提交修改。它属于旧实验状态，本轮没有自动提交或覆盖；旧全量 GSM8K 成绩也没有转移为新候选成绩。

## 未在本轮完成的项目

- `forward_current_impl` 仍需按路由状态拆分为多个私有函数，并对 Graph/PCP/DSpark 组合做完整矩阵；
- prefill/decode Attention 的 workspace 共享需要先证明生命周期、Graph capture 地址和 checkpoint loader 兼容，暂不继续冒险删除；
- 67 个环境开关中仅 DeepSeek-V4 MUSA .cpp/MUSA decoder/gate/MoE 路径已统一；独立 header-only policy 的 getenv 将在下一轮统一接入相同 accessor，避免超大机械 diff。
