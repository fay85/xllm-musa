# xLLM eager vs SGLang eager TTFT 性能差距定位（2026-08-01）

## 结论

在同一张 MUSA GPU、同一模型、同一请求与客户端下，原始 xLLM eager 的 TTFT 确实慢于 SGLang eager：

- 原始 xLLM：296.889 ms
- SGLang：270.330 ms
- 初始差距：+26.559 ms（xLLM 慢）

本轮已定位并修复主要差距。最终默认配置的多轮结果为：

- xLLM：272.285 ms（3 轮均值，271.448–273.421 ms，sample std 1.020 ms）
- SGLang：270.976 ms（2 轮均值，270.330–271.622 ms，sample std 0.914 ms）
- 剩余差距：+1.309 ms（+0.48%）

即已回收 25.250 ms，约占原始差距的 95.1%。目前两者在本测试噪声范围内接近持平，但不能声称 xLLM 已稳定快于 SGLang；单轮最优 xLLM 268.703 ms 可以快于 SGLang，但尚未稳定复现为整体优势。

## 公平配置

两套服务均使用：

- 物理设备：MUSA GPU 3
- 模型：`/workspace/model_weights/Qwen3.5-27B-FP8`
- 并发：C1
- 输入长度：2000 tokens
- 输出长度：16 tokens
- warmup：4
- measure：20
- greedy / temperature 0
- 相同 benchmark client、prompt seed、采样参数
- client SHA256：`d4a334590246dae6be5146166a4fb4f741900058fddb2a3473cdee0dea4ce9e4`

xLLM 与 SGLang 均为 eager prefill；测试按顺序独占同一物理 GPU，未并行竞争。

## 最终统计

| 实现 / 轮次 | TTFT mean (ms) | model forward (ms) |
|---|---:|---:|
| xLLM 原始 baseline | 296.889 | 264.088 |
| SGLang r1 | 270.330 | 231.572 |
| SGLang r2 | 271.622 | 232.160 |
| xLLM final r1 | 271.987 | 238.795 |
| xLLM final r2 | 273.421 | 239.587 |
| xLLM final r3 | 271.448 | 237.734 |
| **SGLang 均值** | **270.976** | **231.866** |
| **xLLM final 均值** | **272.285** | **238.705** |

阶段拆分：

| 阶段 | xLLM final (ms) | SGLang (ms) | xLLM - SGLang |
|---|---:|---:|---:|
| model forward | 238.705 | 231.866 | +6.839 |
| forward 外部 | 33.580 | 39.110 | **-5.530** |
| endpoint TTFT | 272.285 | 270.976 | +1.309 |

这说明 xLLM 的 C++ 请求调度/前后处理已经比 SGLang 快约 5.5 ms；剩余差距不在 C++ 服务框架，而在模型 forward，主要是 GDN 路径。

## 已定位并修复的问题

### 1. FP8 block GEMM 每次调用存在两个 host stream sync

文件：`xllm/core/kernels/musa/fp8_block_gemm.cpp`

原实现每个 block-FP8 GEMM 后执行两次 host stream synchronize。一次 prefill forward 有约 256 次该 GEMM，导致流水被频繁打断。

修复后默认异步执行；可用以下环境变量回退：

`XLLM_FP8_FORCE_HOST_SYNC=1`

效果：

- 修复前：296.889 ms
- 修复后：277.518 ms
- 单项改善：19.371 ms

该问题解释了原始差距的大部分。

### 2. GDN prefill 的不必要 materialization 和 dtype 转换

文件：

- `xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp`
- `xllm/core/kernels/musa/gdn_decode.cu`

已完成并默认启用：

1. Q/K L2 norm 原地写回，减少临时 tensor。
2. C1 pure-prefill 使用 zero-state scratch，避免无意义状态准备。
3. C1 pure-prefill 直接使用 Q/K/V 视图，减少 QKVZ split/materialization；Z 仍保持必要 contiguous。
4. gating kernel 直接输出 FP32，避免 BF16 g/beta 再转换到 FP32，行为与 SGLang 路径一致。
5. 以上高风险 shape 优化仅限已经验证的 `batch_size == 1` pure-prefill，不扩展到未验证场景。

逐项 A/B：

| 版本 | TTFT (ms) | 相对上一版本 |
|---|---:|---:|
| no-host-sync | 277.518 | -19.371 vs baseline |
| + inplace Q/K norm | 274.675 | -2.843 |
| + zero-state scratch | 273.172 | -1.502 |
| + direct prefill views | 271.688 | -1.484 |
| + FP32 gating output | 270.958 / 268.703 | 继续改善但有轮间波动 |

回退开关：

- `XLLM_GDN_DISABLE_INPLACE_QK_L2NORM=1`
- `XLLM_GDN_DISABLE_ZERO_STATE_SCRATCH=1`
- `XLLM_GDN_DISABLE_VIEW_SPLIT=1`
- `XLLM_GDN_DISABLE_GATING_FP32=1`

## 剩余瓶颈

最终 profiler（约 2050 tokens）显示：

| xLLM bucket | 时间 (ms) |
|---|---:|
| full attention | 23.138 |
| GDN total | 84.705 |
| GDN projection | 30.977 |
| GDN conv | 11.726 |
| GDN gate | 1.296 |
| GDN MATE wrapper | 19.271 |
| GDN o_proj | 11.080 |
| GDN state prep | 0.241 |
| GDN state write | 0.913 |
| GDN norm | 3.468 |
| MLP | 125.313 |
| outer norm | 6.751 |

相近 token 数下，SGLang 的 linear-attention/GDN 约 74.331 ms，而 xLLM 约 84.705 ms；约 10 ms 的主要剩余 forward 差距集中在 GDN core，而 projection、o_proj、full attention 已基本接近。

进一步的直接 primitive 测试排除了以下原因：

- 相同形状的 groupwise FP8 GEMM：xLLM 0.195–0.197 ms，SGLang 0.195–0.198 ms，基本一致。
- 当前公开 MATE GDN op：xLLM container 0.312 ms/layer，SGLang container 0.318 ms/layer，xLLM 并不慢。
- 但 xLLM 实际 MATE wrapper scope 为约 19.271/48 = 0.402 ms/layer。

根因是 xLLM 当前预编译/缓存的 MATE FFI 模块要求 Q/K/V 必须 contiguous。xLLM 从 fused QKVZ 输出切视图后，必须额外 materialize 三个 tensor；SGLang 当前公开 MATE 接口只要求最后一维 stride 为 1，可接受非连续 leading stride。实验把 strided view 直接传给 xLLM MATE 时明确 fail-closed：

`Check failed: (q.IsContiguous()) is false: q must be contiguous`

因此下一步应当是：重编译/替换 xLLM 的 MATE FFI ABI，使其支持 stride-aware Q/K/V，或从 C++ 直接绑定当前公开 MATE 实现。该项预计可再回收数毫秒，是目前最明确的后续优化方向。

## 失败或无收益实验

这些实验均未保留为默认行为：

- `MUSA_ENABLE_LLC_OPT=1`：xLLM 退化到约 300.95 ms。
- 模仿 SGLang 的 row-wise split：274.83 ms，相比 273.17 ms 退化。
- `index_copy_` state write：因 index dtype 为 int32、算子要求 int64 而 fail-closed；该阶段本身仅约 1.15 ms，不是主瓶颈。
- 直接传 strided MATE input：被旧 FFI 的 contiguous ABI 拒绝，实验代码已撤回。
- padded/non-MATE GDN 路径：明显退化，未保留。
- 关闭 packed 路径仅从 296.889 降至 295.243 ms，不是主要根因。

所有失败实验的 artifact 均保留，未覆盖成功结果。

## 正确性与源码状态

最终默认版本与原始 async baseline 做了 5 个 greedy 请求逐文本比对：

- 5/5 exact match
- 两侧生成文本集合 SHA256：
  `455bc89c9aeb2718b1062369c64803f3345c74a77407fa67b6147cb9c531a223`
- 0 request errors

最终 xLLM binary SHA256：

`9a016628d8d3248fa466b3bea0fbb09702bcd914b9b5f5c2cb44d4f9d4c759a9`

本轮相关文件的 `git diff --check` 通过。全仓检查仍会命中用户已有的 `fp8_act_quant.cu` CRLF/trailing-whitespace 噪声，本轮未修改该文件。

SGLang 只做了临时 profiler 插桩，patch 已保存在 evidence 目录后完整 reverse；以下两个文件当前 focused diff 为空：

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/models/qwen3_5.py`

临时 patch SHA256：

`6dd4cc61f9868c6c98c9e705cbe4849a9c96dfe941871ba4d5232d6cb868dbfb`

## Evidence

xLLM 根目录：

`/workspace/bench_results/eager_gap_xllm_sglang_20260801_113814`

关键子目录：

- `xllm_baseline`
- `xllm_no_host_sync`
- `xllm_inplace_qk`
- `xllm_zero_state_scratch`
- `xllm_view_split_v4`
- `xllm_gating_fp32`
- `xllm_gating_fp32_r2`
- `xllm_final_default_r1`
- `xllm_final_default_r2`
- `xllm_final_default_r3`
- `xllm_breakdown_final_default`
- `correctness_final_default`
- `xllm_row_split`
- `xllm_strided_mate`
- `xllm_index_copy`
- `xllm_llc`

SGLang 根目录：

`/workspace/sglang_qwen35/benchmark_results/eager_gap_xllm_sglang_20260801_113814`

关键子目录：

- `sglang_llc`
- `sglang_llc_r2`
- `sglang_breakdown_r6`
- `temporary_timing_and_breakdown.patch`
