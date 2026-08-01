# MUSA Graph Stream Debug Handoff — 2026-07-29

## 目标与运行边界

- 目标：定位 USE_MUSA-only 重构后 C=5 性能回退，并消除 graph 路径中多余的 host `musaStreamSynchronize`，前提是 graph 正确性不能退化。
- 运行权威环境：`mccxadmin@10.121.38.92`，容器 `xllm-musa2.9.1-sdk5.1-dev`，工作树 `/workspace/xllm-musa-main-sync`。
- 模型/正确性协议：Qwen3.5-27B BF16，`--enable_graph=true`，`XLLM_USE_FA3=1`、`XLLM_USE_FA3_DECODE=1`，`correctness_check.sh`，`MAXTOK=256`。
- C=5 性能协议：ISL=2500、OSL=1500、C=5、4 warmup waves、4 measure waves（20 measured requests）。

## 已验证的性能/同步事实

| 版本/实验 | TTFT | TPOT | 结论 |
|---|---:|---:|---|
| 7/24 pre-split C5 基线 | 1522.65 ms | 52.50 ms | 对照 |
| post-refactor 严格基线 | 2398.70 ms | 58.19 ms | TTFT 是主要回退 |
| worker persistent stream + homogeneous scheduler | 2099.39 ms | 58.06 ms | TTFT 有改善，TPOT 未恢复 |
| event-handoff C5 完整 20 request | 2123.14 ms | 59.72 ms | 去掉 steady host sync，但该版本端到端 TPOT 未提升 |

稳定 replay trace 的有效结论：

- 旧路径每 decode step 有 8 个 `musaStreamSynchronize`；约 55 ms/step 中的大部分是等待图内 GPU 工作，不应把“8 次同步”直接等同于“可节省 55 ms TPOT”。
- 已恢复 MUSA `replace_token`，消除了其中两处 CPU↔MUSA fallback 同步。
- event-pool 版本的同形状重复 replay trace 没有记录到 steady-state `musaStreamSynchronize`。

## Graph stream 调试：有效结论

### 正确基线

以下组合在 92 上已现场复测通过 C1：

- graph executor：`/tmp/xllm_graph_stream_pre_20260729/musa_graph_executor_impl.{cpp,h}`。
- TVM-FFI stream：`/tmp/musa_tvmffi_stream.{cpp,h}.new`（event-pool 回退路径）。
- build log：`build_logs/build_cuda_graph_musa_event_pool_baseline_20260729.log`。
- correctness log：
  `/workspace/bench_results/qwen35_27b_bf16_post_refactor_20260729/event_pool_baseline_verify_20260729/correctness_c1.log`。

该 run 的答案包含 `391`，`CORRECTNESS: PASS`。

### 直接 default-stream 实验（有效）

把 null handle 直接传给 TVM-FFI/Mate 后：

- 非阻塞 graph 路径在 bucket=1 capture 后触发 `MUSA error: operation aborted`；日志在
  `.../default_stream_validate_20260729/server.log`。
- 加 `MUSA_LAUNCH_BLOCKING=1` 时 C1/C2 可以通过。

这说明该实验至少存在异步 graph 路径问题；不能作为最终修复。

### FFI binding 追踪（有效）

在正确 event-pool 基线上，临时日志显示：

- graph capture 前，很多 FFI 调用的 current handle 为 null，因此按设计绑定到一个稳定的非默认 FFI pool stream（该 run 为 `0x7f57604980b0`）。
- bucket=1 capture 前后，FFI 开始**逐调用**重绑到非默认 current/capture stream（该 run 为 `0x7f576033d6e0`）。
- torch_musa 的 `currentStreamCaptureStatusMayInitCtx()` 在该 capture 期间仍可能报告 `capturing=0`。
- 每次 FFI 调用都会重新执行两次 `TVMFFIEnvSetStream`（DL CUDA 与 ExtDev）；这很可能带有 Mate 所需的逐调用绑定语义，不能被“一次性全局 SetStream + 后续 early return”替代。

追踪日志：

`/workspace/bench_results/qwen35_27b_bf16_post_refactor_20260729/event_pool_stream_debug_20260729/server.log`

`LD_PRELOAD` 尝试拦截 `TVMFFIEnvSetStream` 未抓到调用（符号绑定方式导致），最终使用了源码内临时、marker-gated 日志。

## 重要：此前 override A/B 的混杂因素

**不要把此前“强制 capture stream 后输出错误”的结果直接归因于 stream override。**

原因：从本机同步到 92 的 `musa_graph_executor_impl.cpp` 除 stream 改动外，还包含无关的 decode metadata 改动（例如放宽 `int32` 到 `int64`、在 graph replay 中插入 `to(torch::kInt32)`）。该文件与正确基线的 diff 不止 stream 代码；这些改动本身可破坏 graph 的 pointer/shape/stream 行为。

因此以下实验是**无效归因数据**：

- 全局 `MusaTvmffiStreamOverrideGuard` + replay handoff；
- 只在 `capture_begin → capture_end` 包 override；
- null fallback-to-capture-stream 版本。

它们都使用了夹带 metadata 改动的 graph 文件，虽然都出现错误回答，但不能证明 stream patch 是根因。

## 当前远端状态（接手前必须恢复）

当前 92 容器不是干净基线：

- 运行中的测试 server：PID `677907`，端口 `18141`，输出错误；先停止它。
- 当前 graph executor 包含本地同步来的 metadata 改动和 fallback-override 实验代码。
- 当前 `musa_tvmffi_stream.*` 包含 marker-gated debug/fallback 实验代码。
- 最新实验 build log：
  `build_logs/build_cuda_graph_musa_fallback_override_20260729.log`。

不要用当前二进制做性能结论。

## 安全恢复步骤

1. 停掉 PID `677907` 及其子进程；确认没有 non-zombie `xllm`。
2. 恢复严格正确基线文件：

```bash
cd /workspace/xllm-musa-main-sync
cp /tmp/xllm_graph_stream_pre_20260729/musa_graph_executor_impl.cpp \
  xllm/core/runtime/musa/musa_graph_executor_impl.cpp
cp /tmp/xllm_graph_stream_pre_20260729/musa_graph_executor_impl.h \
  xllm/core/runtime/musa/musa_graph_executor_impl.h
cp /tmp/musa_tvmffi_stream.cpp.new \
  xllm/core/kernels/musa/musa_tvmffi_stream.cpp
cp /tmp/musa_tvmffi_stream.h.new \
  xllm/core/kernels/musa/musa_tvmffi_stream.h
```

3. 仅删除这四个相关 build object/archive 后，运行规范构建：

```bash
MAX_JOBS=16 NINJA_TARGET=xllm ./_build_cuda_graph_musa.sh
```

4. 先用 C1（随后 C2）重新确认 `correctness_check.sh` 通过；若不通过，停止 stream 调优并先恢复正确性。

## 正确的下一轮最小实验

在**严格基线文件**上做，不要再从本机整文件覆盖 graph executor：

1. 保留每个 `MusaTvmffiStreamGuard` 的 `TVMFFIEnvSetStream` 调用。
2. 不做全局 `SetStream`、不在 `bind_musa_tvmffi_stream()` early-return。
3. 仅在 actual capture scope 记录一个 thread-local fallback stream：当且仅当某个 FFI guard 读到 null current handle 时，使用 capture stream 作为本次调用的 fallback，随后仍调用 `TVMFFIEnvSetStream`。
4. fallback active 时禁止该 guard 对 FFI pool 建立错误的 event handoff；否则会把同步关系连到未实际执行工作的 pool stream。
5. 每一步都依次验证：bucket=1 capture、C1、C2、稳定 replay sync trace、C5 correctness，最后才跑完整 C5 20-request benchmark。

该补丁应以小型 unified patch 直接叠加到 `/tmp/xllm_graph_stream_pre_20260729` 的 graph 文件，避免再次带入本机 graph executor 的 metadata 改动。

## 清理项

- 删除本地临时文件 `.tmp_tvmffi_set_stream_trace.cpp`。
- 删除/恢复 marker-gated `XLLM_TVMFFI_STREAM_DEBUG` 代码后再做正式性能测试。
- 保留 `/tmp/musa_tvmffi_stream.{cpp,h}.new` 与
  `/tmp/xllm_graph_stream_pre_20260729/`，它们是已验证正确的恢复点。
