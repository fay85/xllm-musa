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

基线：`c829f0c7047a25035d9923eb34489690999b7955`，运行拓扑 P94/P96 + D90，Control 保持原服务。

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

候选没有精度回退；B1/B2 TPOT 小幅改善，TTFT 在测量波动范围内。该窗口不是全量 GSM8K 或长稳验收。

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

原 r14 服务已恢复，未 reset GPU；没有停止其他容器或 Control。

## 未在本轮完成的项目

- `forward_current_impl` 仍需按路由状态拆分为多个私有函数，并对 Graph/PCP/DSpark 组合做完整矩阵；
- prefill/decode Attention 的 workspace 共享需要先证明生命周期、Graph capture 地址和 checkpoint loader 兼容，暂不继续冒险删除；
- 67 个环境开关中仅 DeepSeek-V4 MUSA .cpp/MUSA decoder/gate/MoE 路径已统一；独立 header-only policy 的 getenv 将在下一轮统一接入相同 accessor，避免超大机械 diff。
