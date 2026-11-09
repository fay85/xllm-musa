# Qwen3.8-27B-FP8 Prefill (CP > 1) GDN `conv1d` 跨 Rank 通信依赖分析与优化方案

## 1. 结论概述

**是的，当前实现中存在明显且不必要的数据依赖，阻碍了计算与通信的 overlap。**

在 `cp > 1`（当前 MUSA PCP 为 `CP = 2`）的 Prefill 推理中，GDN（Gated Delta Net）模块每层有两次跨 CP rank 的前向状态传递（Rank 0 $\to$ Rank 1）：
1. **第一次通信（`conv1d` 边界状态传递）**：Rank 0 将其序列分片末尾的 $K - 1 = 3$ 个 token 的 `conv1d` 状态发送给 Rank 1。
2. **第二次通信（GDN 递归状态 `ssm_cache` 传递）**：Rank 0 将 chunked GDN recurrence 计算完成后的 `final_state` 发送给 Rank 1。

对于**第一次通信**，由于 `causal_conv1d` 是深度可分离的一维有限脉冲响应（FIR）卷积（无输出递归依赖）：
$$y_t = \text{SiLU}\left(b + \sum_{j=0}^{K-1} w_j \cdot x_{t - (K-1) + j}\right)$$
Rank 1 计算其前 $K - 1 = 3$ 个 token 所需的历史状态，**仅仅是 Rank 0 在 `conv1d` 之前的原始输入 `conv_input`（即 `qkvz_proj_` 输出切分出的 `mixed_qkv`）的最后 $K - 1 = 3$ 个 token**，完全不依赖 Rank 0 的 `causal_conv1d_prefill` 计算结果，也不依赖 `ba_proj_` 投影结果。

然而当前代码将第一次通信放在了 `ba_proj_` 和 `causal_conv1d_prefill` 之后执行，在每层 GDN 中引入了 **约 283 $\mu\text{s}$ 的非必要串行等待**（48 层 GDN 累计约 **13.6 ms**），并进一步在 Rank 1 引发了 MCCL 通信算子与 FP8 Persistent GEMM 之间的 SM 驻留争用。

---

## 2. Profile 实证分析（基于 `/data/albert/Pkgs/profile/qwen3.8-27b-fp8`）

以 Profile 中第二个 Prefill step 的第 0 个 GDN 层为例（基准时间 $T_0 = 2678239470000\ \mu\text{s}$），对比 `prefill0_rank0_pid34601` 与 `prefill0_rank1_pid34603`：

### 2.1 Rank 0 时间线与串行依赖链

| 阶段 | Stream | Correlation | Kernel / Runtime API | 起止时间 ($\mu\text{s}$, 相对 $T_0$) | 耗时 ($\mu\text{s}$) | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. `qkvz_proj_` | stream 20 | `4501122` | `musa_asm_e4m3e4m3bf16bf16ssgemm...` | `3315.1 .. 4087.2` | `772.0` | 生成 `qkvz_flat` (`[3072, 16384]`)，此时 `conv_input` 的最后 3 行已就绪 |
| 2. `ba_proj_` | stream 0 | `4501160` | `musa_asm_bf16bf16bf16bf16ssgemm...` | `4095.8 .. 4152.4` | `56.6` | 生成 `[b \| a]`，仅后续 `gdn_gating_kernel` 使用，`conv1d` 并不依赖它 |
| 3. `conv1d` | stream 0 | **`4501178`** | `causal_conv1d_fwd_token_major_kernel` | `4154.6 .. 4375.8` | **`221.1`** | Grid `[1, 384, 40]`，计算全部 3072 个 token 的卷积，并顺带将 `conv_input` 末尾 3 行写入 `conv_cache` |
| 4. 状态拷贝 | stream 0 | `4501195` | `IndexSelectVectorKernel` | `4378.7 .. 4384.4` | `5.7` | `send_prefix_cache_row` 中执行 `conv_cache.index_select(0, indices)` |
| 5. 跨流同步 | CPU | `4501205` / `4501209` | `musaEventRecord` $\to$ `musaStreamWaitEvent` | `3514.5 .. 3530.0` (Host) | - | 强制通信流 stream 23 等待计算流 stream 0 完成 `4501160` + `4501178` + `4501195` |
| 6. 第一次通信 | stream 23 | **`4501219`** | `imcclDevKernel_Broadcast_RING_LL` | `4388.4 .. 4429.8` | **`41.4`** | 发送 `[1, 10240, 3]` BF16 (60 KB) 的 `conv_cache` 状态给 Rank 1 |

### 2.2 代码根因定位

在 `xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp:1054-1076` 中：

```cpp
if (pcp_context != nullptr) {
  musa_pcp::receive_prefix_cache_row(
      conv_cache, logical_state_indices, *pcp_context);
}
mixed_qkv =
    xllm::kernel::musa::causal_conv1d_prefill(conv_input,
                                              conv_weight,
                                              conv_cache,
                                              std::nullopt,
                                              attn_metadata.q_cu_seq_lens,
                                              logical_state_indices,
                                              has_initial_state,
                                              /*silu_activation=*/true);
if (pcp_context != nullptr) {
  musa_pcp::send_prefix_cache_row(
      conv_cache, logical_state_indices, *pcp_context);
}
```

而在 `xllm/core/kernels/musa/causal_conv1d.cu:540-550` 中，`causal_conv1d_fwd_token_major_kernel` 更新 `conv_state` 的逻辑为：

```cpp
if (seq_len >= 3) {
  conv_state[0 * params.conv_state_l_stride] =
      x[(seq_start + seq_len - 3) * params.x_l_stride +
        feature * params.x_c_stride];
  conv_state[1 * params.conv_state_l_stride] =
      x[(seq_start + seq_len - 2) * params.x_l_stride +
        feature * params.x_c_stride];
  conv_state[2 * params.conv_state_l_stride] =
      x[(seq_start + seq_len - 1) * params.x_l_stride +
        feature * params.x_c_stride];
}
```

可以看到，写入 `conv_state`（即 `conv_cache`）的数据**完全来自未卷积的输入张量 `x`（`conv_input`）的最后 3 行**。但因为复用了全量卷积 kernel `4501178` 附带的 `conv_state` 写回逻辑，导致：
1. **不必要的 `causal_conv1d_prefill` 依赖（`221.1 us`）**：必须等 `4501178` 算完所有 3072 个 token 的卷积，stream 0 才能推进到 `send_prefix_cache_row`。
2. **不必要的 `ba_proj_` 依赖（`56.6 us`）**：在 `project_flat_inputs` 中，`qkvz_proj_`（`4501122`）算完后立即在 stream 0 上启动了 `ba_proj_`（`4501160`），而 `conv_input` 只依赖 `qkvz_proj_`。
3. **不必要的中转拷贝开销（Rank 0 `5.7 us` + Rank 1 `12.4 us`）**：
   - Rank 0 先在 `causal_conv1d` 中将末尾 3 行转置写入 `conv_cache`，再通过 `index_select`（`corr=4501195`, `5.7 us`）拷出连续内存发送；
   - Rank 1 收到后通过 `index_put_`（触发 `CastOp` `corr=4199218` `5.5 us` + `IndexVectorizedKernel` `corr=4199223` `6.9 us`）写入 `conv_cache`，随后 Rank 1 的 `causal_conv1d_fwd_token_major_kernel`（`corr=4199243`）读取这 3 列后又立刻用 Rank 1 自己的最后 3 个 token 覆盖写回 `conv_cache`。

### 2.3 连锁反应：Rank 1 上的 SM 争用与 Persistent GEMM 膨胀

观察同一层在 **Rank 1**（`prefill0_rank1_pid34603`）上的表现：
- Rank 1 在进入 `Qwen3GatedDeltaNetBaseImpl::forward` 最开头（`qkvz_proj_` 之前）就调用了 `expect_prefix_cache_row`，在通信流 stream 20 上启动了 `imcclDevKernel_Broadcast_RING_LL`（`corr=4199035`，`2985.0 .. 4436.9`，持续 **`1451.9 us`**）。
- 该 MCCL kernel 的配置为 `grid=[4, 1, 1]`, `shared memory = 92032 B (90 KB)`，提前占用了 4 个 SM。
- 随后 Rank 1 在 stream 23 上启动 `qkvz_proj_` 的 FP8 Persistent GEMM（`corr=4199144`，`musa_asm_e4m3e4m3bf16bf16ssgemm...persis_stage2`），其配置为 **`grid=[60, 1, 1]`, `shared memory = 131072 B (128 KB)`, `blocks per SM = 1.0`**（刚好填满 S5000 的 60 个 SM）。
- 由于 4 个 SM 被正在自旋等待 Rank 0 发送的 `Broadcast_RING_LL`（`4199035`）占用（90 KB + 128 KB 超出单 SM 共享内存上限），`qkvz_proj_` 的 60 个 persistent CTA 中有 **4 个 CTA 无法上核**，一直被阻塞到 `4436.9 us`（即 Rank 0 的 `4501219` 完成时刻）才得以调度执行，导致 Rank 1 的 `qkvz_proj_` 耗时从正常的 `~840 us` 暴涨至 **`1762.0 us`（`3353.3 .. 5115.4`）**！

---

## 3. 优化方案

### 方案一：消除 Rank 0 的计算依赖，提前发送 `conv_input` 尾部 token（核心改动，推荐立即落地）

#### 核心思路
在 Rank 0（`cp_rank == 0`）上，`musa_pcp` 的切分粒度为 `kRowAlignment = 256` 的整数倍（`first_rows >= 256 > K - 1 = 3`）。因此 Rank 0 传递给 Rank 1 的 `conv_cache` 状态恒等于 `conv_input`（即 `qkvz_proj_` 输出的前 `qkv_dim` 列）的最后 $K - 1 = 3$ 行转置后的 `[1, qkv_dim, 3]`。

将 Rank 0 的第一次通信前移到 **`qkvz_proj_` 完成之后、`ba_proj_` 与 `causal_conv1d_prefill` 之前**：
1. 先执行 `qkvz_proj_` 投影，切分出 `mixed_qkv`（即 `conv_input`）。
2. **在 Rank 0 上立即提取最后 $K - 1 = 3$ 行并启动异步广播 `send_prefix_rows`**：
   - 提取 `conv_input.narrow(0, seq_len - (conv_kernel_size_ - 1), conv_kernel_size_ - 1).t().unsqueeze(0).contiguous()`（仅 60 KB 数据，拷贝/转置耗时仅 `~2-3 us`）。
   - 调用 `musa_pcp::send_prefix_rows` 在通信流（stream 23）上发起 `broadcast_async`（`4501219`）。
3. 随后在计算流（stream 0）上依次执行 `ba_proj_`（`4501160`）和 `causal_conv1d_prefill`（`4501178`）。
4. 此时通信流 stream 23 上的 `imcclDevKernel_Broadcast_RING_LL`（`4501219`, `~41 us`）与计算流 stream 0 上的 `ba_proj_`（`56.6 us`）+ `causal_conv1d_prefill`（`221.1 us`）**完全并行 overlap**，且移除了原有的 `conv_cache.index_select`（`4501195`, `5.7 us`）。

#### 改动示意（`xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp`）

```cpp
// 1. 当启用 PCP 时，拆分 qkvz_proj_ 与 ba_proj_ 的调用顺序，使 qkvz 先完成以尽早发起 conv 状态广播
torch::Tensor qkvz_flat;
torch::Tensor ba_flat;
if (pcp_context != nullptr && uses_contiguous_qkvzba_layout() && !use_spec_verify) {
  qkvz_flat = project_qkvz(hidden_states);
  const int64_t qkv_dim = 2 * local_nk * head_k_dim_ + local_nv * head_v_dim_;
  torch::Tensor conv_input =
      qkvz_flat.narrow(/*dim=*/-1, /*start=*/0, /*length=*/qkv_dim);
  if (pcp_context->cp_rank == 0) {
    const int64_t state_len = conv_kernel_size_ - 1;
    CHECK_GE(conv_input.size(0), state_len);
    // 直接从 conv_input 末尾取 K-1 个 token，构造 [1, qkv_dim, K-1] 发送给 Rank 1，
    // 无需等待 ba_proj_ 和 causal_conv1d_prefill 完成。
    torch::Tensor conv_prefix =
        conv_input.narrow(/*dim=*/0,
                          /*start=*/conv_input.size(0) - state_len,
                          /*length=*/state_len)
            .t()
            .unsqueeze(0)
            .contiguous();
    musa_pcp::send_prefix_rows(conv_prefix, *pcp_context);
  }
  // 随后启动 ba_proj_，与通信流上的 conv_prefix 广播重叠执行
  ba_flat = project_ba(hidden_states);
}
```

同时在 `causal_conv1d_prefill` 之后去掉 `musa_pcp::send_prefix_cache_row(conv_cache, logical_state_indices, *pcp_context)`。

---

### 方案二：调整 Rank 1 `expect_prefix_cache_row` 的投递时机，避免阻塞 Persistent GEMM（推荐配合方案一落地）

#### 核心思路
在 Rank 1 上，当前在 `forward()` 最开头（`qkvz_proj_` 之前）调用了：
```cpp
musa_pcp::expect_prefix_cache_row(kv_cache.get_conv_cache(), *pcp_context);
musa_pcp::expect_prefix_cache_row(kv_cache.get_ssm_cache(), *pcp_context);
```
这会导致 MCCL 的 `imcclDevKernel_Broadcast` 在 `qkvz_proj_` 之前抢占 4 个 SM（每个占用 90 KB 共享内存），使得随后的 60-CTA FP8 Persistent GEMM（`qkvz_proj_`，每个 CTA 需 128 KB 共享内存）有 4 个 CTA 无法上核，造成 `qkvz_proj_` 耗时翻倍（`840 us -> 1762 us`）。

#### 优化方式
1. 在 Rank 1 上，**先启动 `qkvz_proj_`（stream 23）**，让 60 个 Persistent GEMM CTA 立即占满全部 60 个 SM。
2. 紧接着（在 `qkvz_proj_` launch 之后、`ba_proj_` / `causal_conv1d_prefill` 之前）再调用 `expect_prefix_cache_row` 投递 MCCL 接收任务（stream 20）。
   - 或者在 `expect_prefix_rows` 中让通信流等待 `qkvz_proj_` 或者是将其放在 `qkvz_proj_` 发起之后。
3. 结合方案一（Rank 0 提前 `~283 us` 发送），当 Rank 1 的 `qkvz_proj_` 释放出前几个 SM 时，Rank 0 的数据已经就绪，Rank 1 的 `Broadcast_RING_LL` 只需 `~40 us` 即可完成，不再发生长时自旋占核。

---

### 方案三：消除 `conv_cache` 的中转读写与转置（进阶优化）

#### 核心思路
目前 Rank 0 $\to$ Rank 1 传递的 $K - 1 = 3$ 个 token 需要经过：
`conv_input [3, D] (Rank 0)` $\xrightarrow{\text{transpose}}$ `[1, D, 3]` $\xrightarrow{\text{MCCL}}$ `[1, D, 3] (Rank 1)` $\xrightarrow{\text{Cast + index\_put\_}}$ `conv_cache[slot] (Rank 1)` $\xrightarrow{\text{read \& overwrite}}$ `causal_conv1d_fwd_token_major_kernel`。

可以进一步简化为：
1. **方式 A（零 kernel 修改，消除 Rank 1 `index_put_`）**：
   Rank 1 的 `take_prefix_rows(*pcp_context)` 直接拿到形状为 `[1, D, 3]` 的连续张量 `prefix_conv_state`。在 Rank 1 调用 `causal_conv1d_prefill` 时：
   - 若僅需读取初始状态，或者在 `causal_conv1d_fwd_token_major_kernel` 中支持传入独立的只读 `initial_conv_state_ptr`（当提供时从 `initial_conv_state_ptr` 读取初始 3 个 token，而最终状态仍写回 `conv_cache[cache_idx]`），即可完全省去 Rank 1 上的 `conv_cache.index_put_`（消除 `CastOp` `4199218` + `IndexVectorizedKernel` `4199223`，每层省 `~12.4 us`）。
2. **方式 B（Token-major 直接传递，消除 Rank 0 转置 + Rank 1 `index_put_`）**：
   - Rank 0 直接切片发送未转置的最后 $K - 1 = 3$ 行：`conv_input.narrow(0, seq_len - 3, 3).contiguous()`（形状 `[3, D]`，行主序连续拷贝，带宽利用率最高）；
   - `causal_conv1d_fwd_token_major_kernel` 增加可选参数 `prefix_x_ptr`（指向 `[3, D]` 的前序 token），当 `token_offset == 0 && prefix_x_ptr != nullptr` 时，直接按与 `load_x` 完全相同的 `feature` 连续访存方式读取 `prefix_x_ptr[0..2, feature]`，同时将 Rank 1 自身的最后 3 行写入 `conv_cache[cache_idx]`。

---

## 4. 预期收益汇总

| 优化项 | 单层 GDN 节省 / 提前时间 | 48 层 GDN 累计收益 | 说明 |
| :--- | :--- | :--- | :--- |
| 移除 `causal_conv1d_prefill` 对第一次通信的阻塞 | `~221 us` | `~10.6 ms` | `4501178` 与 `4501219` 完全并行 overlap |
| 移除 `ba_proj_` 对第一次通信的阻塞 | `~57 us` | `~2.7 ms` | `4501160` 与 `4501219` 完全并行 overlap |
| 移除 Rank 0 `conv_cache.index_select` + Rank 1 `index_put_` | `~15-18 us` | `~0.8 ms` | 省去 `conv_cache` 散写再聚读及 `int64` 索引转换 |
| 消除 Rank 1 `Broadcast_RING_LL` 抢占 `qkvz_proj_` Persistent GEMM SM | 首层及失步层节省 `~700-900 us` | 显著降低双卡流水气泡 | 避免 60-CTA Persistent GEMM 因 4 个 SM 被 MCCL 占住而尾延迟翻倍 |
