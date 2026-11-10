# Qwen3.8-27B-FP8 Prefill (CP > 1) GDN Prefix Cache 状态恢复 (`restore_linear_state_slots`) 跨 Rank 依赖分析与优化调研

## 1. 核心结论

1. **您的判断在算法与数据依赖层面完全正确**：
   在 `cp > 1`（当前 MUSA PCP 为 `CP = 2`）且请求进入 PCP 分片路径（`pcp_context.has_value()`）时，**Rank > 0（即 Rank 1）完全不需要在每次 Prefill 请求开始时执行“从 Prefix Cache 槽位（`read_id`）将 48 层 GDN 的 `conv_cache` 和 `ssm_cache` 拷贝到当前工作槽位（`write_id`）”这一设备端 D2D 拷贝操作**。该操作在 PCP 路径下只需要 **Rank 0** 读取。
2. **Trace 时间段精准定性**：
   在 `/data/albert/Pkgs/profile/qwen3.8-27b-fp8/prefill0_rank1_pid34603_20261101_201433237.pt.trace.json` 中，相对时间 `00:00:00.760184991` 到 `00:00:00.763182461`（耗时 **`2997.470 us` $\approx$ `3.00 ms`**）对应第 2 个 Prefill Step（Step 1）的 `WorkerImpl::prepare_linear_state_cache` $\to$ `restore_linear_state_slots` $\to$ `copy_linear_state_slot` 阶段。
   在这 `3.00 ms` 内，Rank 1 的 Worker 线程（`tid=36122`）遍历全部 48 个 GDN 层，逐层发射 `conv_cache.select(0, write_id).copy_(conv_cache.select(0, read_id))` 和 `ssm_cache.narrow(...).copy_(ssm_cache.narrow(...))` 共 **96 个 D2D 拷贝算子**。
3. **为什么 Rank 1 上的这 3.00 ms 是纯冗余（Dead Write）**：
   - 在 PCP 分片下，Prompt 序列 `[0, num_rows)` 被切分为 Rank 0 负责 `[0, first_rows)`、Rank 1 负责 `[first_rows, num_rows)`。
   - **Rank 0 (`cp_rank == 0`)** 紧接在 Prefix Cache 命中的历史前缀之后，因此其 `causal_conv1d_prefill` 和 `mate_gated_delta_rule_prefill` **必须从本地 `conv_cache[write_id]` 和 `ssm_cache[write_id]` 读取历史初始状态**。
   - **Rank 1 (`cp_rank == 1`)** 紧接在 Rank 0 的分片末尾之后（`receives_prefix_state(*pcp_context) == true`），其 `conv1d` 初始状态直接取自 Rank 0 发来的 `conv_prefix`（`params.prefix_x_ptr != nullptr`），其 GDN Recurrence 初始状态直接取自 Rank 0 发来的 `final_state`（`take_prefix_rows`），**在整整 48 层 GDN 的前向计算中从不读取本地 `conv_cache[write_id]` 和 `ssm_cache[write_id]`**；随后在本层算子结束时，Rank 1 又直接将自己分片算出的末尾状态覆写进 `conv_cache[write_id]` 和 `ssm_cache[write_id]`。因此 step 开头拷贝进 `write_id` 的数据在被任何算子读取之前就被 100% 覆盖了。
4. **更进一步的端到端优化结论（读写槽位分离 `reads_distinct_state = true`）**：
   - 由于 Rank 0 和 Rank 1 在每层 GDN 存在跨卡通信同步（Rank 1 在第 0 层 GDN 需要等待 Rank 0 算完 `qkvz_proj_` 并发送 `conv_prefix`），而同一时刻 Rank 0 也在花 **`2628.533 us`（约 `2.63 ms`）** 执行完全相同的 `restore_linear_state_slots`，**如果仅跳过 Rank 1 的 D2D 拷贝而保留 Rank 0 的拷贝，Rank 1 省下的 3 ms 会在第 0 层 GDN 的 `take_prefix_rows` 通信等待处被 Rank 0 阻塞**。
   - 在 xLLM 框架中（`worker_impl.cpp:866`）已设计有 `reads_distinct_state` 机制（NPU/MLU 已启用，MUSA 尚未开启）：让 GDN Prefill 算子在读取初始状态时直接索引 `linear_state_read_ids`（`read_id`），在写出最终状态时索引 `linear_state_ids`（`write_id`）。**启用该机制后，Rank 0 和 Rank 1 均可彻底省去这 96 次 D2D 拷贝，单次 Prefill Step 端到端可直接减少约 `2.6 ~ 3.0 ms`（11 个 Prefill Steps 累计减少约 `27 ~ 30 ms`）**。

---

## 2. Profile Trace 深度拆解（`00:00:00.760184991` ~ `00:00:00.763182461`）

### 2.1 时间戳对齐与全局上下文

- **Rank 1 Trace 文件**：`/data/albert/Pkgs/profile/qwen3.8-27b-fp8/prefill0_rank1_pid34603_20261101_201433237.pt.trace.json`
  - Trace 起始绝对时间（`min_ts`）：`2678238708436.021 us`
  - 您指出的窗口：`00:00:00.760184991` $\to$ `00:00:00.763182461`（相对 `min_ts` 为 `760184.991 us` $\to$ `763182.461 us`，对应绝对时间 `2678239468621.012 us` $\to$ `2678239471618.482 us`），持续时间 **`2997.470 us`**。
- **Rank 0 Trace 文件**：`/data/albert/Pkgs/profile/qwen3.8-27b-fp8/prefill0_rank0_pid34601_20261101_201433237.pt.trace.json`
  - Trace 起始绝对时间（`min_ts`）：`2678238698693.750 us`（比 Rank 1 早启动 `9742.271 us`）。
  - 同一物理时刻（Step 1 开头），Rank 0 在相对时间 `00:00:00.769904012` $\to$ `00:00:00.772532545`（绝对时间 `2678239468597.762 us` $\to$ `2678239471226.295 us`）执行完全相同的 48 层状态恢复循环，持续时间 **`2628.533 us`**。

### 2.2 窗口内事件精确统计（Rank 1，`tid=36122` & `stream=5`）

在 `760184.991 us` 到 `763182.461 us` 之间，Worker 线程 `tid=36122` 在 `prepare_stream_`（`stream 5`）上循环遍历 64 个 Decoder Layer（其中 48 层为 GDN 层）：

| 事件名称 | 类别 (`cat`) | 调用次数 | 累计耗时 (`us`) | 单次平均 (`us`) | 对应代码操作（每层 GDN 各执行 2 次：1 次 `conv_cache`，1 次 `ssm_cache`） |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `aten::select` | `cpu_op` | 96 | 192.741 | 2.008 | `conv_cache.select(0, write_id)` 与 `conv_cache.select(0, read_id)`（48 层 $\times$ 2） |
| `aten::narrow` | `cpu_op` | 96 | 385.601 | 4.017 | `ssm_cache.narrow(0, write_id * stride, stride)` 与 `read_id`（48 层 $\times$ 2） |
| `aten::slice` | `cpu_op` | 96 | 257.836 | 2.686 | `aten::narrow` 内部调用的视图切分 |
| `aten::copy_` | `cpu_op` | 96 | 2201.610 | 22.933 | 48 层 `conv_cache` 槽位拷贝 + 48 层 `ssm_cache` 槽位拷贝 |
| `aten::_copy_from` | `cpu_op` | 96 | 2064.315 | 21.503 | `aten::copy_` 内部派发 |
| `aten::expand_as` / `expand` | `cpu_op` | 96 | 297.287 | 3.097 | `copy_` 内部张量形状对齐 |
| `aten::as_strided` | `cpu_op` | 288 | 231.353 | 0.803 | `select` (96) + `narrow/slice` (96) + `expand` (96) 的底层视图构建 |
| `musaDeviceCanAccessPeer` | `privateuse1_runtime` | 96 | 52.537 | 0.547 | MUSA `copy_` 每次发射前的设备访问检查 |
| `musaLaunchKernel` | `privateuse1_runtime` | 96 | 383.786 | 3.998 | 向 `stream 5` 发射 96 个 `IdentityOp` 拷贝 Kernel |
| `musaEventRecord` | `privateuse1_runtime` | 1 | 2.203 | 2.203 | `763172.066 us`：在 `prepare_stream` 记录 `metadata_ready_event` |
| `musaStreamWaitEvent` | `privateuse1_runtime` | 1 | 8.672 | 8.672 | `763182.461 us`：`compute_stream` (`stream 0`) 等待 `prepare_stream` 完成 |

对应的 **`stream 5` GPU Kernel 执行情况**（`760269.902 us` $\to$ `763184.590 us`）：
- **48 个 BF16 拷贝 Kernel**（`conv_cache` 单槽位 `[10240, 3]` BF16 = `60 KB`）：
  - Kernel：`musa::dnn::unary::kernel_unary<__mt_bfloat16, IdentityOp, ...>`
  - 48 次累计 GPU 执行时间：**`177.001 us`**（单次平均 **`3.688 us`**）。
- **48 个 FP32 拷贝 Kernel**（`ssm_cache` 单槽位 `[32, 128, 128]` FP32 = `2 MB`）：
  - Kernel：`musa::dnn::unary::kernel_unary<float, IdentityOp, ...>`
  - 48 次累计 GPU 执行时间：**`340.041 us`**（单次平均 **`7.084 us`**）。
- **Host-Bound 特征明显**：GPU 实际有效搬运耗时仅 **`517.042 us`（0.52 ms）**，但由于在 CPU 上逐层发起 96 次独立的 PyTorch 视图切分与算子发射，Host 端串行发射耗时高达 **`2997.470 us`（3.00 ms）**，导致 `stream 5` 上每个 Kernel 之间存在大量气泡，且阻塞了后续主计算流 `stream 0` 的模型前向发射。

### 2.3 全 Trace 11 个 Prefill Steps 的状态恢复耗时汇总

在该 Profile Trace 包含的全部 11 个 Prefill Steps 中，每个 Step 开头均触发了完全相同的 48 层 `restore_linear_state_slots`：

| Step 序号 | Rank 1 窗口起点 (`us`) | Rank 1 窗口终点 (`us`) | Rank 1 耗时 (`us`) | Rank 0 窗口起点 (`us`) | Rank 0 窗口终点 (`us`) | Rank 0 耗时 (`us`) | Rank 0 / Rank 1 `conv1d` Grid (`[B, ceil(L/8), 40]`) |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Step 0 | 243188.504 | 246397.725 | **3209.221** | 253236.843 | 256250.202 | **3013.359** | `[1, 384, 40]` (3072) / `[1, 403, 40]` (3224) |
| **Step 1** | **760184.991** | **763182.461** | **2997.470** | **769904.012** | **772532.545** | **2628.533** | **`[1, 384, 40]` (3072) / `[1, 403, 40]` (3224)** |
| Step 2 | 1273400.090 | 1276385.242 | **2985.152** | 1283465.082 | 1286115.555 | **2650.473** | `[1, 352, 40]` (2816) / `[1, 314, 40]` (2512) |
| Step 3 | 1693671.234 | 1695688.283 | **2017.049** | 1703551.575 | 1706044.735 | **2493.160** | `[1, 320, 40]` (2560) / `[1, 291, 40]` (2328) |
| Step 4 | 2093603.874 | 2096549.591 | **2945.717** | 2103710.015 | 2106242.872 | **2532.857** | `[1, 384, 40]` (3072) / `[1, 370, 40]` (2960) |
| Step 5 | 4363078.308 | 4366138.928 | **3060.620** | 4373248.680 | 4377341.652 | **4092.972** | `[1, 416, 40]` (3328) / `[1, 379, 40]` (3032) |
| Step 6 | 4938139.440 | 4940146.684 | **2007.244** | 4948262.876 | 4951313.026 | **3050.150** | `[1, 384, 40]` (3072) / `[1, 375, 40]` (3000) |
| Step 7 | 5408908.598 | 5410914.310 | **2005.712** | 5418503.312 | 5421118.666 | **2615.354** | `[1, 320, 40]` (2560) / `[1, 331, 40]` (2648) |
| Step 8 | 5830054.484 | 5832051.414 | **1996.930** | 5840089.194 | 5842559.311 | **2470.117** | `[1, 320, 40]` (2560) / `[1, 285, 40]` (2280) |
| Step 9 | 6228637.388 | 6230666.113 | **2028.725** | 6238510.348 | 6240911.998 | **2401.650** | `[1, 352, 40]` (2816) / `[1, 317, 40]` (2536) |
| Step 10 | 6649081.809 | 6651122.981 | **2041.172** | 6658962.629 | 6661352.943 | **2390.314** | `[1, 320, 40]` (2560) / `[1, 311, 40]` (2488) |
| **合计** | - | - | **27295.012 us (27.30 ms)** | - | - | **30338.938 us (30.34 ms)** | **11 个 Steps 均启用 PCP (`CP = 2`)** |

---

## 3. 代码链路与设计原理解析：为什么会有这段操作？

### 3.1 Prefix Cache 下的 GDN 状态写时复制（Copy-on-Write）机制

与标准 Full Attention 每个 Token 独立占用 Paged KV Cache Slot 不同，GDN（线性注意力）每层对每条序列只维护固定大小的状态槽位（`BlockType::LINEAR`）：
- `conv_cache`：保存最近 $K - 1 = 3$ 个 Token 在 `conv1d` 前的输入历史；
- `ssm_cache`：保存 GDN 递归状态矩阵 $S_t \in \mathbb{R}^{H_v \times D_v \times D_k}$。

当开启 Prefix Cache（或 Chunked Prefill 跨块恢复）时：
1. **分配阶段（`LinearStateBlockManager::allocate_prefill`，`xllm/core/framework/block/linear_state_block_manager.cpp:324-348`）**：
   如果某条请求命中了 Prefix Cache 中的线性状态检查点槽位 `read_id`（即 `source_id`），为了防止当前请求的前向计算原地修改 `read_id` 从而破坏 Prefix Cache 中可被其他请求复用的公共前缀状态，管理器会保留 `read_id` 作为只读源，并新分配一个空闲槽位 `write_id` 作为当前请求的工作槽位，构成 `[read_id, write_id]` 二元窗口。
2. **Batch 构建阶段（`BatchInputBuilder::append_linear_state_row`，`xllm/core/framework/batch/batch_input_builder.cpp:907-927`）**：
   ```cpp
   const Slice<Block> linear_blocks =
       sequence->kv_state().blocks(BlockType::LINEAR);
   const size_t read_index =
       linear_blocks.size() > 1 ? linear_blocks.size() - 2 : 0;
   state.linear_state_ids.emplace_back(linear_state_id);       // write_id = linear_blocks.back().id()
   state.linear_state_read_ids.back() = linear_blocks[read_index].id(); // read_id
   ```
3. **Worker 准备阶段（`WorkerImpl::prepare_work_before_execute_on_stream`，`xllm/core/runtime/worker_impl.cpp:1400-1404`）**：
   在 Target Worker 的 `prepare_stream_` 上调用 `prepare_linear_state_cache(input_params)`（`xllm/core/runtime/worker_impl.cpp:860-898`）：
   ```cpp
   // TODO: need remove when all device support none D2D
   restore_linear_state_slots(kv_caches_,
                              params.embedding.linear_state_ids,
                              params.embedding.linear_state_read_ids,
                              params.linear_state_validity_mask,
                              reads_distinct_state);
   ```
4. **逐层 D2D 拷贝（`copy_linear_state_slot`，`xllm/core/framework/kv_cache/linear_state_restore.cpp:64-82`）**：
   因为在 MUSA 后端下 `reads_distinct_state` 为 `false`，当 `source_ids[row] != write_ids[row]` 时，执行：
   ```cpp
   for (const KVCache& kv_cache : kv_caches) {
     ...
     if (!conv_cache.defined()) {
       continue;
     }
     const int64_t stride = ssm_cache.size(0) / conv_cache.size(0);
     conv_cache.select(0, write_id).copy_(conv_cache.select(0, read_id));
     ssm_cache.narrow(0, static_cast<int64_t>(write_id) * stride, stride)
         .copy_(ssm_cache.narrow(
             0, static_cast<int64_t>(read_id) * stride, stride));
   }
   ```
   并将 `validity_mask` 填为 `1`，表示 `write_id` 槽位中已经拥有合法的前缀初始状态。

---

## 4. 数据依赖分析：为什么 `cp > 1` 时 Rank 0 需要，而 Rank > 0 不需要？

在 MUSA PCP（`xllm/core/layers/musa/pcp_context.cpp` 与 `xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp`）中，一条长度为 `num_rows` 的待 Prefill 序列按 256 对齐切分为：
- **Rank 0 (`cp_rank == 0`)**：处理序列前半段 `rows [0, first_rows)`；
- **Rank 1 (`cp_rank == 1`)**：处理序列后半段 `rows [first_rows, num_rows)`。

其中 `musa_pcp::receives_prefix_state(context)` 的定义为（`pcp_context.cpp:350`）：
```cpp
bool receives_prefix_state(const Context& context) {
  return context.cp_rank > 0;
}
```

对比 Rank 0 和 Rank 1 在 GDN 层（`Qwen3GatedDeltaNetBaseImpl::forward`）中对 `conv_cache[write_id]` 和 `ssm_cache[write_id]` 的读写行为：

### 4.1 `conv_cache[write_id]` 的读写依赖对比

在 `xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp:1066-1091` 及 `xllm/core/kernels/musa/causal_conv1d.cu:513-552` 中：

```cpp
// causal_conv1d.cu:513-532
if (load_initial_state) {
  if (params.prefix_x_ptr != nullptr) {
    // 路径 A：直接从跨 Rank 传来的 conv_prefix 读取前 3 个历史 token
    const input_t* prefix_x = reinterpret_cast<const input_t*>(params.prefix_x_ptr);
    col0 = static_cast<float>(prefix_x[0 * params.prefix_x_l_stride + feature * params.prefix_x_c_stride]);
    col1 = static_cast<float>(prefix_x[1 * params.prefix_x_l_stride + feature * params.prefix_x_c_stride]);
    col2 = static_cast<float>(prefix_x[2 * params.prefix_x_l_stride + feature * params.prefix_x_c_stride]);
  } else {
    // 路径 B：从本地 conv_cache[write_id] 读取前 3 个历史 token
    col0 = static_cast<float>(conv_state[0 * params.conv_state_l_stride]);
    col1 = static_cast<float>(conv_state[1 * params.conv_state_l_stride]);
    col2 = static_cast<float>(conv_state[2 * params.conv_state_l_stride]);
  }
}
// causal_conv1d.cu:535-545：当 seq_len >= 3 时，直接用当前分片末尾 3 个 token 覆写 conv_cache[write_id]
if (seq_len >= 3) {
  conv_state[0 * params.conv_state_l_stride] = x[(seq_start + seq_len - 3) * params.x_l_stride + ...];
  conv_state[1 * params.conv_state_l_stride] = x[(seq_start + seq_len - 2) * params.x_l_stride + ...];
  conv_state[2 * params.conv_state_l_stride] = x[(seq_start + seq_len - 1) * params.x_l_stride + ...];
}
```

- **Rank 0 (`cp_rank == 0`)**：
  - `receives_prefix_state` 为 `false`，因此 `conv_prefix = std::nullopt`（`params.prefix_x_ptr == nullptr`）。
  - 当命中 Prefix Cache（`load_initial_state == true`）时，走**路径 B**，**必须读取 `conv_cache[write_id]`** 中的前缀状态。
- **Rank 1 (`cp_rank == 1`)**：
  - `receives_prefix_state` 为 `true`，`conv_prefix = musa_pcp::take_prefix_rows(*pcp_context)`（非空，`params.prefix_x_ptr != nullptr`）。
  - 走**路径 A**，直接读取 Rank 0 发送过来的 `[3, qkv_dim]` 边界行，**完全不读取 `conv_cache[write_id]`**！
  - 同时，由于 PCP 分片按 `kRowAlignment = 256` 对齐，Rank 1 的 `seq_len >= 256 >= 3` 恒成立，Kernel 会直接用 Rank 1 分片的最后 3 行把 `conv_cache[write_id]` 完全覆写。

### 4.2 `ssm_cache[write_id]` 的读写依赖对比

在 `xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp:1263-1290` 中：

```cpp
if (use_mate_gdn_chunked_prefill) {
  torch::Tensor initial_state;
  if (pcp_context != nullptr &&
      musa_pcp::receives_prefix_state(*pcp_context)) {
    // Rank > 0：直接从 Rank 0 接收前半段算出的 final_state 作为 initial_state
    initial_state = musa_pcp::take_prefix_rows(*pcp_context);
  } else {
    // Rank 0：从本地 ssm_cache[write_id] 读取 Prefix Cache 恢复的初始状态
    initial_state =
        torch::index_select(ssm_cache, 0, linear_state_base_indices);
    initial_state.mul_(attn_metadata.has_initial_states.view({-1, 1, 1, 1})
                           .to(initial_state.scalar_type()));
  }
  mate_params.initial_state = initial_state;
}
torch::Tensor mate_final_state;
std::tie(core_attn_out, mate_final_state) =
    xllm::kernel::musa::mate_gated_delta_rule_prefill(mate_params);
const torch::Tensor final_state = mate_final_state.to(ssm_cache.dtype());
// 将当前分片算出的 final_state 覆写进本地 ssm_cache[write_id]
ssm_cache.index_put_({linear_state_base_indices}, final_state);
if (pcp_context != nullptr) {
  musa_pcp::send_prefix_rows(final_state, *pcp_context);
  musa_pcp::defer_final_state(conv_cache, logical_state_indices, *pcp_context);
  musa_pcp::defer_final_state(ssm_cache, linear_state_base_indices, *pcp_context);
}
```

- **Rank 0 (`cp_rank == 0`)**：
  - `receives_prefix_state` 为 `false`，走 `else` 分支，通过 `torch::index_select(ssm_cache, 0, linear_state_base_indices)` **从本地 `ssm_cache[write_id]` 读取 Prefix Cache 状态**；
  - 算完前半段序列后，将前半段末尾状态 `final_state` 通过 `send_prefix_rows` 发送给 Rank 1。
- **Rank 1 (`cp_rank == 1`)**：
  - `receives_prefix_state` 为 `true`，走 `if` 分支，直接将 Rank 0 发来的 `final_state` 作为 `mate_params.initial_state`，**完全不读取本地 `ssm_cache[write_id]`**；
  - 算完后半段序列后，直接执行 `ssm_cache.index_put_({linear_state_base_indices}, final_state)` **全量覆写本地 `ssm_cache[write_id]`**。

### 4.3 模型末尾 `musa_pcp::finish` 的状态汇总流向

在全部 64 层执行完毕后，`Qwen3HybridModelImplBase::forward`（`xllm/models/llm/qwen3_next_hybrid_base.h:251`）调用 `musa_pcp::finish`（`xllm/core/layers/musa/pcp_context.cpp:528-550`）：
```cpp
const int32_t root_rank = last_rank(context); // root_rank = 1 (Rank 1)
...
for (DeferredState& deferred : context.deferred_states) {
  broadcast_cache_rows(deferred.cache, deferred.indices, root_rank, context);
}
```
此时是以 **Rank 1 (`root_rank = 1`)** 为源，将 Rank 1 在 `write_id` 上算出的最终 `conv_cache[write_id]` 和 `ssm_cache[write_id]` 广播写回给 **Rank 0**。

汇总整个生命周期：

| 阶段 | Rank 0 (`cp_rank == 0`) 对 `write_id` 的操作 | Rank 1 (`cp_rank == 1`) 对 `write_id` 的操作 |
| :--- | :--- | :--- |
| 1. `prepare_linear_state_cache`（Step 开头） | `write_id <= copy(read_id)`（**后续会读**） | `write_id <= copy(read_id)`（**死写，后续不读**） |
| 2. GDN 层 `conv1d` | **读 `conv_cache[write_id]`** $\to$ 覆写 `conv_cache[write_id]` | 读 Rank 0 发来的 `conv_prefix`（**不读 `conv_cache`**）$\to$ 覆写 `conv_cache[write_id]` |
| 3. GDN 层 `mate_gdn_prefill` | **读 `ssm_cache[write_id]`** $\to$ 覆写 `ssm_cache[write_id]` $\to$ 发给 Rank 1 | 读 Rank 0 发来的 `final_state`（**不读 `ssm_cache`**）$\to$ 覆写 `ssm_cache[write_id]` |
| 4. 模型末尾 `musa_pcp::finish` | 接收 Rank 1 广播，覆写本地 `conv_cache[write_id]` 与 `ssm_cache[write_id]` | 作为 `root_rank` 读取刚写入的 `write_id` 广播给 Rank 0 |

由此可见，在 PCP 激活时，**Rank 1 在阶段 1 对 `conv_cache[write_id]` 和 `ssm_cache[write_id]` 的拷贝没有任何消费者，100% 属于冗余操作**。

---

## 5. 需要注意的边界条件与设计约束

如果要在代码层面针对 `cp > 1` 消除该开销，需要特别注意以下 3 个边界条件：

### 5.1 边界条件 1：Rank > 0 只能跳过 GPU 拷贝（`copy_linear_state_slot`），不能跳过主机端 `validity_mask` 更新

在 `restore_linear_state_slots`（`xllm/core/framework/kv_cache/linear_state_restore.cpp:166-190`）中：
```cpp
if (!reads_distinct_state) {
  copy_linear_state_slot(kv_caches, write_ids[row], source_ids[row]);
}
std::fill(mask_begin, mask_end, 1);
```
`std::fill(mask_begin, mask_end, 1)` 会把 `input_params.linear_state_validity_mask` 更新为 `1`。在 `Qwen3HybridModelImplBase::forward` 中，`AttentionMetadataBuilder::build` 先于 `musa_pcp::build_context` 执行，会校验并物化 `linear_state_validity_mask`。因此即使在 Rank > 0 跳过 `copy_linear_state_slot`，也**必须保留 `std::fill(mask_begin, mask_end, 1)`**（这与 `reads_distinct_state = true` 时的行为完全一致）。

### 5.2 边界条件 2：并非所有 `cp > 1` 下的 Prefill 请求都会激活 PCP（Fallback 场景）

在 `musa_pcp::build_context`（`xllm/core/layers/musa/pcp_context.cpp:242-279`）中，即使全局配置了 `cp_size == 2`，当遇到以下情况时，`pcp_context` 不会创建（保持 `std::nullopt`），自动回退到 **Replicated Prefill（各 Rank 独立跑完整序列）**：
1. **短序列 Prefill**：待计算的 Token 数 `num_rows <= kRowAlignment (256)`（此时 `balanced_first_rows` 返回 `0`，无法切分给两个 Rank 各至少 256 行）；
2. **多序列打包 Prefill Batch**：`q_seq_lens_vec.size() != 1`（当前 MUSA PCP 仅支持单序列 `batch_size == 1`）；
3. **特殊模式**：`is_spec_verify`、`prefill_without_cache` 或 `aux_capture_.enabled()`。

**影响**：当某个 Prefill Step 回退到非 PCP 路径时，Rank 1 也会从序列的第 0 个 Token 开始独立执行完整计算，此时 Rank 1 的 GDN 层会走 `pcp_context == nullptr` 分支，**需要从本地 `conv_cache` 和 `ssm_cache` 读取初始状态**。
- 如果采用“仅在确定进入 PCP 分支时跳过 Rank > 0 的 `copy_linear_state_slot`”，必须在 `prepare_linear_state_cache` 时使用与 `musa_pcp::build_context` 相同的判定条件（或直接采用 6.1 节的读写槽位分离方案，则无论是否进入 PCP 都不需要拷贝）。

### 5.3 边界条件 3：跨卡同步导致的木桶效应（为什么只优化 Rank 1 不够）

从第 2.3 节的 Profile 数据可以看到：
- 在 Step 1 中，Rank 1 花了 **`2997.470 us`** 做 `restore_linear_state_slots`，而 **Rank 0 在同一物理时刻也花了 `2628.533 us` 做完全相同的操作**。
- 在第 0 层 GDN 中，Rank 1 需要调用 `take_prefix_rows` 等待 Rank 0 发送 `conv_prefix` 和 `final_state`。
- 如果只跳过 Rank 1 的 `restore_linear_state_slots`（省掉 Rank 1 的 3.00 ms），Rank 1 虽然会提前 3.00 ms 进入第 0 层 GDN，但因为 Rank 0 仍在执行它自己的 `restore_linear_state_slots`（2.63 ms），**Rank 1 提前的这 ~2.63 ms 会在第 0 层 GDN 的 MCCL Broadcast 接收处变成同步等待**，端到端收益极其有限。

---

## 6. 优化方案建议

### 方案 A（推荐，治本方案）：在 MUSA 后端启用 GDN 读写槽位分离（`reads_distinct_state = true`），同时消除 Rank 0 和 Rank 1 的 D2D 拷贝

代码库中已经预留了完整的 `linear_state_read_ids` / `linear_state_read_indices` 基础设施（`xllm/core/runtime/worker_impl.cpp:866` 的 `// TODO: need remove when all device support none D2D`，目前 NPU Qwen3.5 与 MLU GLM5-Next 已经启用）：

1. **原理**：
   - 在 `WorkerImpl::prepare_linear_state_cache` 中，对 MUSA Qwen3.5/3.8 Prefill 将 `reads_distinct_state` 设为 `true`。
   - 此时 `restore_linear_state_slots` 在 **Rank 0 和 Rank 1 上都直接跳过 `copy_linear_state_slot`**（零 Kernel 发射、零 D2D 拷贝），仅在 CPU 上将 `validity_mask` 置为 `1`（耗时 `< 1 us`）。
2. **GDN 算子侧适配**：
   - **对于 `ssm_cache`（立即可用，无需改 Kernel）**：
     在 `xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp:1273` 中，Rank 0 读取 `initial_state` 本就是独立的 `torch::index_select`：
     ```cpp
     // 读取初始状态时使用 linear_state_read_base_indices (read_id)
     initial_state = torch::index_select(ssm_cache, 0, linear_state_read_base_indices);
     ...
     // 写回最终状态时使用 linear_state_base_indices (write_id)
     ssm_cache.index_put_({linear_state_base_indices}, final_state);
     ```
     仅此一行改动，即可在 Rank 0 和 Rank 1 上同时消除全部 48 层 `ssm_cache`（每层 2 MB，共 96 MB）的 D2D 拷贝！
   - **对于 `conv_cache`**：
     在 `causal_conv1d_fwd_token_major_kernel`（`xllm/core/kernels/musa/causal_conv1d.cu:503`）中增加可选的 `read_cache_indices_ptr` 参数（或当 `read_id != write_id` 时将 `conv_cache.index_select(0, read_indices)` 作为 `conv_prefix` 传入 `causal_conv1d_prefill`！注意：`causal_conv1d_prefill` **已经支持 `conv_prefix` 参数 `params.prefix_x_ptr`**，当传入 `conv_prefix` 时，Kernel 从 `prefix_x_ptr` 读取初始状态，而把最终状态写入 `cache_indices_ptr` 即 `write_id`！）。
3. **收益预估**：
   - **Rank 1**：每个 Prefill Step 消除 `2.0 ~ 3.2 ms`（11 个 Steps 累计 **27.30 ms**）；
   - **Rank 0**：每个 Prefill Step 同时消除 `2.4 ~ 4.1 ms`（11 个 Steps 累计 **30.34 ms**）；
   - **端到端**：不存在 Rank 1 等 Rank 0 的木桶效应，每个 Prefill Step 真实端到端延迟直接降低 **`2.5 ~ 3.0 ms`**。

### 方案 B（局部快速方案）：仅在 PCP 激活时跳过 Rank > 0 的 `copy_linear_state_slot`

如果仅针对 `cp_rank > 0` 跳过拷贝：
- 在 `WorkerImpl::prepare_linear_state_cache` 中，当检测到当前批次满足 `musa_pcp` 激活条件且 `parallel_args_.cp_group_->rank() > 0` 时，将 `reads_distinct_state` 置为 `true`（从而在 `restore_linear_state_slots` 中只更新 `validity_mask = 1` 而不调用 `copy_linear_state_slot`）。
- 若同时在 Rank 0 上将 48 层逐层串行 `select/narrow/copy_` 合并或异步化，也可减少 Rank 0 对 Rank 1 的阻塞。
