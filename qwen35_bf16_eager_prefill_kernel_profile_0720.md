# Qwen3.5-35B-A3B BF16 eager prefill：kernel/stage 定位（2026-07-20）

## 结论

本轮已经定位到造成 xLLM 长 ISL TTFT 大缺口的具体实现：不是 BF16 grouped GEMM，而是大 prefill 的 MoE 路由/搬运 kernel。旧实现让一个 GPU thread 串行扫描全部 top-k assignment，20k token、top-k=8 时约 16.4 万条 assignment 被串行处理。

将它改为并行 histogram + aligned prefix + token-major gather 后，测量 prefill 的 `moe_preprocess` 从约 **3550 ms** 降到 **44 ms**，约 80 倍；同一轮的 xLLM eager C=1 measured TTFT 从此前约 4.4–4.8 s 降到约 **0.83 s（服务端 request 日志）/1.355 s（benchmark 客户端统计）**。这已经进入 SGLang 的同一量级，不能再把剩余差距归因于 MoE GEMM 本身。

## 对比协议

- 模型：Qwen3.5-35B-A3B，BF16
- 设备：GPU 2；GPU 1 上的既有服务未触碰
- eager：`ENABLE_GRAPH=0`、`ENABLE_PREFILL_PIECEWISE_GRAPH=0`
- C=1：ISL=20,000、OSL=1，1 个长请求 warmup，随后 3 个串行 measured 请求
- xLLM：FA3、custom prefill conv、contiguous BF16 prefill GEMM、chunked prefill 8192
- prefix cache：off
- SGLang：同模型、同 GPU、eager/graph-cache disabled 的已有 Kineto trace；此前干净 benchmark 为约 947.6 ms TTFT，本次 trace run 为 1124.8 ms，存在编译/运行时波动

## xLLM 分阶段计时

`XLLM_PREFILL_BREAKDOWN=1` 的 measured step（路由修复后，3 个请求）：

| 阶段 | 典型耗时 | 说明 |
| --- | ---: | --- |
| full attention | 168–173 ms | 10 个 full-attn layer |
| GDN attention | 270–271 ms | 30 个 GDN layer |
| MoE route | 8.9–9.0 ms | router softmax/top-k，不是主要瓶颈 |
| MoE preprocess | 43.8–44.1 ms | histogram + prefix + gather，修复后 |
| MoE gate/up GEMM | 97–99 ms | 40 个 routed layer |
| MoE activation | 35 ms | SwiGLU |
| MoE down GEMM | 61 ms | 40 个 routed layer |
| MoE combine | 79–80 ms | indexed combine |
| MoE shared | 26 ms | shared expert |
| model executor | 821–826 ms | 不含 HTTP 客户端额外计时 |

旧实现的同一 measured step：

```text
moe_route_ms       ~= 9 ms
moe_preprocess_ms  ~= 3550 ms
moe_gate_up_ms     ~= 103 ms
moe_down_ms        ~= 62 ms
model_executor_ms  ~= 4354 ms
```

所以之前把 `mlp_gate_up` 的大数值当成 routed GEMM 慢是不准确的；外层计时包含了旧 preprocess 的串行路由/搬运。

## 已实现的修复

### 1. MoE BF16 大 prefill 路由改为并行路径

`xllm/core/kernels/musa/fp8_act_quant.cu` 现在使用：

1. `moe_preprocess_histogram_kernel`：每个 assignment 并行统计 expert histogram；
2. `moe_preprocess_prefix_aligned_kernel`：按 expert 计算 aligned cursor 和 grouped-GEMM 的 `group_m_counts`；
3. `moe_preprocess_assign_bf16_kernel`：每个 token 一个 block，读取 hidden row 后按 top-k fan-out，写入 expert-major padded layout 和 `original_to_padded`。

旧的 `moe_decode_route_bf16_kernel` 仍保留给 decode 专用路径；大 prefill 不再调用它。

### 2. TVM FFI stream barrier 修复

`MusaTvmffiStreamGuard` 不再把合法的 MUSA default stream（null handle）误判为“未初始化”，也不再创建专用 FFI stream 并在每个 wrapper 前后同步。当前直接把 Torch 当前 stream（包括 null default stream）绑定给 TVM FFI。旧 trace 中约 4.59 s 的 `musaStreamSynchronize` 与该误判直接相关；修复后的 A/B 请求 TTFT 从约 4.89 s 降到约 4.45 s，并为本次 MoE 路由优化提供了正确的异步基线。

## 性能验证

### C=1，修复后 clean eager

目录：`/data/feihu/profile_results/qwen35_bf16_moe_route_fix_clean_c1_20260720_0030`

- warmup：TTFT 10,771 ms（首轮包含模型/算子冷启动，不作为性能值）
- measured 3 请求：benchmark mean TTFT **1,354.68 ms**，median **1,353.06 ms**
- xLLM 服务端 request 日志的三个 measured TTFT：**837 / 834 / 831 ms**
- measured model executor：**825.6 / 823.5 / 821.1 ms**
- 三个请求均 HTTP 成功、输出 token 数正确

benchmark 客户端统计与服务端 request 日志不一致，说明当前 benchmark 的首 token 时间戳还包含约 0.5 s 的客户端/流式接收开销；后续比较 xLLM/SGLang 必须使用同一统计口径。服务端的 kernel/stage 计时更适合做算子归因。

### C=5 并发正确性回归

目录：`/data/feihu/profile_results/qwen35_bf16_moe_route_fix_c5_20260720_0015`

- 5 个 20k 请求全部成功；其中一轮实际以 batch=4、chunk=8192 进入 prefill，未出现 routing、索引或 grouped-GEMM 错误。
- `correctness_check.sh`：C=1 及并发 C=5 均 `CORRECTNESS: PASS`，输出非空且并发结果一致。

## libkineto / Torch Kineto 覆盖

`XllmKinetoProfiler` 新增纯 prefill 选择窗口：

```text
XLLM_KINETO_PROFILE_PREFILL=1
XLLM_KINETO_WARMUP_PREFILL_STEPS=1
XLLM_KINETO_TRACE_PREFILL_STEPS=1
```

Torch Kineto trace 已成功保存：

`/data/feihu/profile_results/qwen35_bf16_moe_route_fix_20260720_0005/xllm_torch_kineto_prefill.json`

该构建的 MUSA Torch Kineto 没有把设备 kernel 名称暴露为 `kernel` category，但记录了 CPU/user scopes 和 MUSA runtime API；修复后的 trace 中 `xllm/prefill_step` 约 834 ms，且可以看到 23 次 `musaStreamSynchronize`、1986 次 `musaLaunchKernel` 等调用。直接 libkineto 也已扩展到 `CONCURRENT_KERNEL`、`PRIVATEUSE1_RUNTIME`、`PRIVATEUSE1_DRIVER`、memcpy/memset、CPU/user annotation 等类别；不过本机 MUSA driver 在独立 `startTrace()` 时报告 `External init callback must run in same thread`，最终 activities=0，因此该 direct trace 不能作为 kernel 数值依据，已保留 Torch trace 和 breakdown 结果。

## SGLang 参考 trace

已有 SGLang eager trace 的主要 GPU kernel 累计时间：

| kernel family | calls | 累计 |
| --- | ---: | ---: |
| BF16 grouped GEMM | 80 | 209.7 ms |
| Mate full-attention | 10 | 124.5 ms |
| BF16 GEMM family | 110 | 115.9 ms |
| `tilelang_fused_chunk_gdn_prefill` | 30 | 36.6 ms |
| `sglang_musa_causal_conv1d_fwd_kernel` | 30 | 27.7 ms |
| deep-gemm preprocess | 40 | 25.9 ms |
| post reorder | 40 | 21.9 ms |
| qkvzba contiguous | 30 | 21.8 ms |

SGLang trace 中 `musaStreamSynchronize` 仅 10 次、累计约 0.34 ms；xLLM 修复前的大量同步主要来自 FFI stream guard，现已去除。两边 trace 的 kernel 命名/驱动可见性不同，不能直接逐项相减，但可以确认当前 xLLM 的 3.55 s 已不是 GEMM kernel 的正常执行时间。

## 下一步

当前 xLLM eager measured executor 约 0.82 s，已接近或低于已有 SGLang trace 的 0.95–1.12 s TTFT。剩余工作应聚焦在：

1. 用统一的服务端 TTFT 统计口径重跑 xLLM/SGLang，排除客户端流式计时偏差；
2. 对 GDN 270 ms、full attention 170 ms、MoE combine 80 ms 分别做 kernel 名称/launch 级对齐；
3. 继续检查 xLLM 的 MoE activation/combine 是否能合并成 SGLang 的 fused kernel，但不要再优先改 grouped GEMM；
4. 在 MUSA driver 修复 libkineto MUPTI callback 后，再补一份真正包含 device kernel duration 的 direct trace。

## 2026-07-20：response tokenizer 首 token 固定开销

在 MoE kernel 修复后，仍观察到 benchmark client TTFT 比服务端
`first_token_internal` 多约 0.43–0.44 s。通过在 response processor 中加入临时分段计时，确认：

- `response_dispatch` 和 response worker 启动几乎即时；
- `Sequence::generate_streaming_output` 内的 tokenizer decode 单次约 **415–441 ms**；
- logprob 与 output callback 均约 0 ms；
- 根因是 `TokenizerProxy` 为每个 response worker 懒加载 tokenizer clone，Qwen tokenizer
  文件在首个任务上被重复加载。response thread 轮转时，前几个请求会分别付费一次。

修复位于 `xllm/core/scheduler/async_response_processor.cpp`：构造
`AsyncResponseProcessor` 后，对每个 response worker 使用 `schedule_with_tid` 预先执行一次
tokenizer decode，并用 `BlockingCounter` 等待全部 worker 完成。请求路径不再触发 lazy clone。

### 修复后的验证

最终构建日志：
`/workspace/xllm-git-master/build_logs/build_cuda_graph_musa_20260720_034929.log`

最终二进制在容器内通过完整 `xllm` link；GPU2 运行，GPU1 的既有服务未触碰。

目录：`/data/feihu/bench_results/qwen35_bf16_tokenizer_warmup_final_20260720_0610`

- 模型：Qwen3.5-35B-A3B BF16；eager；prefix-cache off；FA3/custom prefill；ISL≈20k。
- C=1、OSL=1：1 个 warmup + 4 个 measured，4 个请求均成功。
- benchmark client measured mean TTFT **847.81 ms**，median **848.53 ms**，P90
  **849.73 ms**；mean latency **847.84 ms**。
- 同轮服务端 prefill breakdown：model executor **约 749–752 ms**；
  `moe_preprocess` **约 37.1–37.3 ms**；`moe_act` **约 15.0 ms**；
  `moe_combine` **约 22.4–22.5 ms**；response tail 仅约 0–1 ms。
- `correctness_check.sh`（17×23，MAXTOK=128）`CORRECTNESS: PASS`。
- 预热发生在 `Application startup complete` 之前；本次日志中 response pool 创建到服务
  ready 约 1.9 s，属于一次性启动成本，不进入请求 TTFT。
- C=1、OSL=300 回归（1 warmup + 2 measured）：mean TTFT **1678.69 ms**（请求按
  1 req/s 串行发出，包含请求间隔影响），mean TPOT **27.26 ms**，mean latency
  **9828.89 ms**。decode 正常，不能用该串行 client TTFT 与 OSL=1 直接比较。

修复前同类 measured client TTFT 为 **约 1.27–1.36 s**，服务端内部 TTFT 约
**0.75 s**；因此本修复消除了约 **0.42–0.44 s/request** 的 CPU detokenization
固定开销。期间尝试的 scheduler 500 ms flush 改动没有带来收益，已从最终源代码撤回；
最终只保留 tokenizer worker warmup。

在相同统计口径下，当前 xLLM C=1 BF16 长 ISL 已不再落后于已有 SGLang eager 参考值
（约 0.95 s clean benchmark；trace run 约 1.12 s）。下一步再针对真正的 GPU
GDN/full-attention kernel 差异做统一 benchmark，不应继续把 response tokenizer 开销
误认为 prefill kernel gap。

## 最终同协议 C=1 BF16 对比（4+10，20k/300）

修复完成后，使用与 SGLang clean baseline 相同的正式协议重跑 xLLM：BF16、GPU2、
prefix-cache off、FA3、graph decode、eager long prefill、ISL=20,000、OSL=300、
4 个 warmup、10 个 measured、C=1。结果目录：
`/data/feihu/bench_results/qwen35_bf16_current_matched_c1_4p10_20260720`。

| implementation | mean TTFT | median TTFT | mean TPOT | total TGS | success |
| --- | ---: | ---: | ---: | ---: | ---: |
| xLLM current | **837.21 ms** | **836.58 ms** | 15.93 ms | 3326.57 tok/s | 10/10 |
| SGLang mixed | 947.58 ms | 936.64 ms | **12.28 ms** | **4393 tok/s** | 10/10 |

xLLM mean TTFT 比 SGLang 低 110.37 ms（-11.65%），说明长 prefill/首 token gap 已闭合；
但 TPOT 仍高 3.65 ms（+29.72%），total TGS 仅为 SGLang 的 75.72%。下一阶段应转向
BF16 decode 小 batch：当前主要差距已经从 TTFT 转移到 TPOT，而不是继续优化 20k prefill。
