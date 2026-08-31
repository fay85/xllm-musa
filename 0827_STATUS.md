# 2026-08-27 Qwen3-0.6B software-beam / MUSA TopK 调试状态

> 目标读者：从零接手、并与现有 agent 协同调试的 agent。
>
> 本文记录截至 2026-08-27 00:10（Asia/Shanghai）的事实、证据、当时安全状态和下一步实验。
> **2026-08-27 16:56 下午更新：§11 记录已落地的 graph-on 修复与准入证据。§0–§10 仍是上午诊断原文，其中“CPU fallback 是当前生产安全状态”已被 §11 取代。第三人 review 请先读 §11，再回看前面的根因与约束。**

## 0. 一页结论

当前问题不是普通 Qwen3-0.6B greedy 推理，而是 `beam_width=128` 的 software-beam 路径：

```text
logprobs: FP32 [128, 151936]
TopK: k = 2 * beam_width = 256
TopK indices: int64 [128, 256]
```

已确认的故障边界：在真实 xLLM 集成路径中，`torch::topk` 返回后立刻检查，就能观察到 `indices` 中出现由 FP32 score 字节拼成的 int64 值。该值之后被 software-beam 当作 token id，最终在下一步 embedding `index_select` 中触发非法地址访问。

尚未确认的更底层根因：为什么相同 shape 的独立 TorchMUSA/muDNN TopK 测试稳定，而 xLLM 集成路径会间歇性返回错误 indices。当前重点怀疑异步执行、workspace 生命周期/复用或 graph executor 周边的内存交互，但 standalone 的 current-stream、caching-allocator 和双 stream 实验均未复现，因此不能把 workspace 假说写成最终结论。

当前生产安全状态：对 MUSA `k >= 246` 的 top-logprobs TopK 临时回退到 CPU reference；其余小 K 仍走 GPU。该路径通过多轮 1000/1000 稳定性测试，但 QPS 相比实验性 GPU + 全局 uncached allocator 低约 36.6%～38.7%。CPU fallback 只能作为保正确性的临时方案，不是最终性能修复。

当前没有本轮 xLLM 服务在运行；最新 dump 仍为：

```text
/data/feihu/xllm-git-master-python/core_2026-08-26_23:11:20.164_worker38092_67441.mudmp
```

在此之后的 standalone muDNN 测试和 CPU fallback 测试没有生成新 dump。

## 1. 权威环境与工作树状态

### 1.1 远端与容器

所有项目操作通过：

```bash
ssh dev92
```

权威路径：

```text
host repo:      /data/feihu/xllm-git-master-python
container:      xllm-musa2.9.1-sdk5.1-dev
container repo: /workspace/xllm-git-master-python
mount:          /data/feihu -> /workspace
```

容器与挂载已核验：

```bash
docker inspect -f '{{.Name}} {{range .Mounts}}{{.Source}}:{{.Destination}} {{end}}' \
  xllm-musa2.9.1-sdk5.1-dev
```

当前主要版本：

```text
MUSA SDK/mcc:   5.1.0
torch:          2.9.1
torch_musa:     2.9.1+a18d871
mate:           0.2.6
tvm_ffi:        0.1.11.post1+musa.1
tilelang_musa:  0.1.12+musa.2
```

注意：`import mate` 报告版本为 `0.2.6`，但当前模块路径显示为
`/workspace/mate_0.2.5/mate/__init__.py`。这是目录名遗留，不能只凭目录名判断实际版本；继续实验前应同时记录 `mate.__version__` 和 `mate.__file__`。

### 1.2 分支、HEAD 与损坏的 git index

```text
branch: feat/python-qwen35-musagraph
HEAD:   3f656fd87f9fdc63d88e53392a8123fa092d29a3
```

当前 `.git/index` 损坏，`git status`/`git diff` 会失败：

```text
error: bad signature 0x00000000
fatal: index file corrupt
```

不要擅自删除、重建或 reset index。工作树中有其他用户的未提交修改；在用户明确授权修复 index 之前，用源文件 SHA256、`cmp`、显式备份和目标文件列表管理实验。不要因为 `git status` 不可用而假设工作树干净。

本文只新增 `0827_STATUS.md`，没有修改本轮调试代码。

## 2. 当前源码与二进制状态

### 2.1 临时安全实现

文件：

```text
/data/feihu/xllm-git-master-python/xllm/core/framework/sampling/sampler.cpp
```

关键逻辑：

```cpp
constexpr int64_t kMusaMaxStableTopK = 245;

std::tuple<torch::Tensor, torch::Tensor> musa_safe_topk(
    const torch::Tensor& input,
    int64_t k) {
  if (k <= kMusaMaxStableTopK) {
    return input.topk(k, /*dim=*/-1);
  }

  auto cpu_input = input.to(torch::kCPU);
  return cpu_input.topk(k, /*dim=*/-1);
}
```

调用点仅用于 `params.max_top_logprobs > 0` 的 TopK；software-beam 的 `k=256` 会回退 CPU。当前文件 SHA256：

```text
e17590a60f75605dc157a97661d8edbee9b9ffc64fc223c4d072918a47bedca7
```

同时确认以下两个 runtime 文件已恢复到调试前内容：

```text
xllm/core/runtime/worker_impl.cpp
b7feb2793bc3ae32bea00c4ec7bc62497cddf43d06a30375f6bd16472e0f433b

xllm/core/runtime/llm_worker_impl.cpp
a567aefb86b61dd790aea794e44289dc3d33ff51b097aa9ac5688237f764c7c1
```

### 2.2 最后一次安全构建

```text
build log:
/data/feihu/xllm-git-master-python/build_logs/build_cuda_graph_musa_20260826_234524.log

binary:
/data/feihu/xllm-git-master-python/build/lib.linux-x86_64-cpython-310/xllm/xllm

binary timestamp:
2026-08-26 23:46:54.395951934 +0800

binary SHA256:
e4f0206ec16a176b59cb08ed4f713b7ed9da3f15b71b59563a459d0670f1bbc9
```

构建日志末尾包含完整 `xllm` executable link，不只是 object compile。

## 3. 原始现象与确定的证据链

### 3.1 最早 dump 的表面故障点

代表性文件：

```text
/data/feihu/xllm-git-master-python/core_2026-08-25_18:06:41.023_worker38092_12552.mudmp
```

关键 dispatch：

```text
IndexSelectVectorKernel<__mt_bfloat16,int,true,128>
grid={8,2,1}, block={16,64,1}
embedding input=0x10016800000
vocab=151936 (0x25180)
hidden=1024 (0x400)
indices=128
```

因此 dump 中看到的是 embedding lookup 消费了坏 token id，不一定是最初产生错误的算子。

最新 dump：

```text
/data/feihu/xllm-git-master-python/core_2026-08-26_23:11:20.164_worker38092_67441.mudmp
```

关键 dispatch：

```text
IndexSelectVectorKernel<__mt_bfloat16,int,true,128>
grid={8,1,1}, block={16,64,1}
out=0x100167fc800
index=0x10fda200520
embedding input=0x10016800000
vocab=151936
hidden=1024
indices=1
```

这仍然是下游 embedding 报告点。

### 3.2 真正把故障边界推进到 TopK 返回之后

在 `torch::topk` 返回后立即加入临时检查，真实 xLLM 输入出现：

```text
/data/feihu/validation_qwen06_mate026_20260825/topk_output_check/server/xllm_Qwen3-0.6B.log

MUSA_TOPK_INDEX_CORRUPTION
flat_index=1024 row=4 column=0
token_index=-4665941854583144686
low32_as_float=-0.472161
topk_value=-0.052917
vocab_size=151936
```

另一次在 `MUSA_LAUNCH_BLOCKING=1` 下仍出现：

```text
/data/feihu/validation_qwen06_mate026_20260826/topk_check_blocking/server/xllm_Qwen3-0.6B.log

MUSA_TOPK_INDEX_CORRUPTION
flat_index=16384 row=64 column=0
token_index=-4715057646774796599
low32_as_float=0.0156095
topk_value=2.21476
vocab_size=151936
```

第一例坏 int64 为 `0xbf3f3e9abef1bf12`，两个 32-bit half 解释成 float 分别约为 `-0.747049`、`-0.472161`。这说明 score/value 字节进入了本应为 int64 token index 的区域。

因此已确认：

1. host 侧最初的 prompt/token 输入合法；
2. software-beam 合并前，TopK indices 已出现越界值；
3. embedding `IndexSelectVectorKernel` 是坏 token 的消费者和最终报告点；
4. 故障边界已经缩小到集成路径的 TopK 调用/输出周围，而不是 tokenizer 或最初的 host token 构造。

仍不能仅凭这些证据断言是 muDNN kernel 单体缺陷，因为 standalone 同 shape 不能复现，且异步的上游写入也可能在这个边界显现。

### 3.3 已核验的真实 TopK layout

真实 xLLM 输入不是异常 stride/view：

```text
shape:          [128, 151936]
stride:         [151936, 1]
dtype:          Float32
contiguous:     true
storage_offset: 0
k:              256
sorted:         true
largest:        true
```

### 3.4 已排除或不足以解释问题的方向

- 负 placeholder token：host/device token 诊断没有发现最初输入为负；加入 D2H 诊断曾使 40/40 通过，但属于 timing perturbation，不能视为修复。
- `replace_token(... synchronize_stream=true)`：运行到 982/1000 后服务退出，不充分。
- output token clone/device synchronize：有单轮 1000/1000，但后续其他配置仍复现，不能作为稳定修复。
- sorted large TopK 改 unsorted + small sort：后续复现。
- 把 256 拆成 128+128 chunked TopK：后续复现。
- `MUSA_LAUNCH_BLOCKING=1`：仍直接捕获坏 indices，而且 QPS 明显下降，只适合定位。
- 只运行独立 `torch.topk`：1000 次均通过，不能覆盖集成条件。
- 只看某一轮 1000/1000：该问题具有间歇性，必须重复跑并检查新 dump 和输出签名。

## 4. TorchMUSA/muDNN 调用链与当前假说

容器内 TorchMUSA 源码：

```text
/home/torch_musa/torch_musa/csrc/aten/ops/Topk.cpp
/home/torch_musa/torch_musa/csrc/aten/mudnn/TopK.cpp
/home/torch_musa/torch_musa/csrc/aten/mudnn/Handle.cpp
/home/torch_musa/torch_musa/csrc/aten/utils/Utils.cpp
/home/torch_musa/torch_musa/csrc/core/MUSACachingAllocator.cpp
```

调用链：

```text
at::musa::TopkOut
  -> FormatContiguous(self)
  -> CreateMUTensor(input/values/indices)
  -> GetMudnnHandle() + current MUSA stream
  -> musa::dnn::TopK::Run(..., InternalMemAlloc)
  -> mudnnGetTopKWorkspaceSize
  -> InternalMemAlloc(ws_size)
  -> MUSACachingAllocator::raw_alloc
  -> mudnnTopK(...)
  -> MemoryHandler destructor
  -> MUSACachingAllocator::raw_delete
```

`InternalMemAlloc` 返回的是 raw pointer + deleter，不是带 `recordStream` 的 `DataPtr`。allocator 的 `free(Block*)` 只有在 `block->stream_uses` 非空时才插 event；否则直接回收到 cache。由此产生一个合理但尚未证实的假说：TopK 内部 workspace 在异步工作真正完成之前被 allocator 复用。

不能直接下结论的反证：下面的 direct muDNN 实验在 current stream 上，无论 direct `musaMalloc` 还是 `MUSACachingAllocator::raw_alloc/raw_delete`，都没有复现。

## 5. standalone / direct muDNN 实验

### 5.1 仓库内纯 Python 回归脚本

```text
/data/feihu/xllm-git-master-python/tests/python/test_mudnn_topk_regression.py
```

它固定覆盖：

```text
input:      FP32 [128,151936]
k:          256
iterations: 1000（可通过环境变量修改）
checks:     int64 dtype、shape、index range、values==gather、CPU candidate set
seed:       1234
```

该测试在默认模式和 `MUSA_LAUNCH_BLOCKING=1` 下均通过。结论只能是“独立 TorchMUSA TopK 未复现”，不能据此关闭集成问题。

运行示例：

```bash
ssh dev92
docker exec xllm-musa2.9.1-sdk5.1-dev bash -lc '
  cd /workspace/xllm-git-master-python
  MUSA_VISIBLE_DEVICES=0 MUDNN_TOPK_ITERATIONS=1000 \
    python3 -m pytest -q -s tests/python/test_mudnn_topk_regression.py
'
```

先用 `mthreads-gmi` 检查选定设备，不要覆盖其他人的运行。

### 5.2 direct muDNN C API 扩展

当前诊断源（临时文件，可能在容器重启后消失）：

```text
dev92 host: /tmp/codex_direct_mudnn_topk_ext_alloc.cpp
container:  /tmp/codex_direct_mudnn_topk_ext_alloc.cpp
```

扩展使用 muDNN C API，匹配 `[128,151936]`、`k=256`、FP32 input、int64 indices，并把 handle 绑定到：

```cpp
c10::musa::getCurrentMUSAStream().stream()
```

结果：

| workspace | stream | 调用数 | 越界 indices | elapsed |
|---|---|---:|---:|---:|
| `musaMalloc/musaFree` | current stream | 1000 | 0 | 1.731 s |
| caching allocator raw alloc/delete | current stream | 1000 | 0 | 1.620 s |
| direct alloc，双 stream 交错 | two pooled streams | 1200 | 0 | 未单独计时 |
| cached alloc，双 stream 交错 | two pooled streams | 1200 | 0 | 未单独计时 |

编译时需要链接：

```text
/workspace/mudnn_3.4.0/lib
/usr/local/musa/lib
/usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_python.so
```

并包含 `/home/torch_musa`，否则 `MUSAStream::stream()` 会缺符号。

重要边界：该扩展每次创建 descriptor/handle，执行模式仍不等同于 xLLM 长生命周期 executor。下一步应该把它嵌入更接近 xLLM 的重复 handle、真实 stream 和相邻分配序列，而不是继续只加随机 standalone case。

## 6. 集成压力测试与性能证据

共同协议：

```text
model:              Qwen3-0.6B
beam_width:         128
output_len:         3
concurrency:        1
input lengths:      1..1024，32 个固定 prompt
graph:              true
chunked_prefill:    false
schedule overlap:   false
MTP:                off
```

### 6.1 当前 CPU fallback 的长稳态结果

| artifact | 结果 | QPS | mean | p99 | unstable prompts |
|---|---:|---:|---:|---:|---:|
| `validation_qwen06_mate026_20260825/final_safe_repeat/result1000_a.json` | 1000/1000 | 5.4643 | 181.833 ms | 240.549 ms | 0 |
| `validation_qwen06_mate026_20260825/final_safe_repeat/result1000_b.json` | 1000/1000 | 5.2876 | 187.956 ms | 255.610 ms | 0 |

2026-08-26 最终源码对应的短 smoke：

```text
/data/feihu/validation_qwen06_mate026_20260826/final_safe/result200.json
200/200, QPS=6.5119, mean=152.487 ms, p99=232.364 ms, unstable=0
```

短 smoke 的 QPS 不可代替两轮 1000-request baseline。

### 6.2 原始 GPU TopK 的错误性能读数

一些 raw/unsorted/device-sync 单轮可到约 15～16 QPS，甚至出现 1000/1000；但重复或换诊断位置后仍会退出。任何有错误、服务提前停止或后续复现的 run，都不能作为有效性能基线。

例如：

```text
/data/feihu/validation_qwen06_mate026_20260826/topk_layout2/result1000.json
344/1000 后服务退出；文件里的 QPS=14.5625 只统计成功前窗口，不能使用。
```

### 6.3 全局关闭 caching allocator 的实验

环境：

```text
PYTORCH_NO_MUSA_MEMORY_CACHING=1
MUSA_LAUNCH_BLOCKING=0
```

结果：

```text
/data/feihu/validation_qwen06_mate026_20260826/topk_uncached_noblocking/result500.json
500/500
QPS=8.6224
mean=115.251 ms
p50=112.825 ms
p99=139.334 ms
unstable_prompt_count=1
```

相对 CPU fallback 的两轮长测，吞吐高约 57.8%～63.1%；反过来说 CPU fallback QPS 低约 36.6%～38.7%。

该配置不能直接部署：

1. 它影响进程内所有 MUSA allocation，不只是 TopK workspace；
2. 只有 500 requests；
3. 出现 1 个 prompt 的两个不同输出签名；
4. 尚未完成重复长测和更完整 correctness。

`MUSA_LAUNCH_BLOCKING=1` + uncached 的结果约为 QPS 6.09、mean 163.51 ms，仅说明全局同步开销很大。

## 7. dump 与符号定位方法

每次服务异常退出后，先记录启动前后的 dump 列表，不能只猜最后一个 kernel：

```bash
find /data/feihu/xllm-git-master-python -maxdepth 1 \
  -type f -name 'core_*.mudmp' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %f\n' | sort | tail -20
```

当前容器可用：

```text
/usr/local/musa/bin/llvm-objdump
/usr/local/musa/bin/llvm-nm
/usr/local/musa/bin/llvm-addr2line
/usr/local/musa/bin/llvm-strings
```

当前容器没有发现：

```text
musa-gdb
musaasm
mudmp / mu-dump CLI
```

当前二进制/库的参考符号：

```text
xllm fused_add_rmsnorm BF16 host stub: 0x2a33560
TorchMUSA IndexSelectRun:              0x1501b60
TorchMUSA BF16 IndexSelect device stub:0x150aa50
```

必须针对“本次运行实际二进制 SHA256”重新解析地址；这些地址不是跨 build 稳定 ABI。

示例：

```bash
docker exec xllm-musa2.9.1-sdk5.1-dev bash -lc '
  llvm-nm -C \
    /usr/local/lib/python3.10/dist-packages/torch_musa/lib/libmusa_kernels.so \
    | grep IndexSelectVectorKernel

  llvm-nm -C --defined-only \
    /workspace/xllm-git-master-python/build/lib.linux-x86_64-cpython-310/xllm/xllm \
    | grep fused_add_rmsnorm
'
```

`llvm-objdump` 能解析 ELF host launcher stub，但当前工具链不能把 dump 中的 GPU PC 完整反汇编到 device source line；不要把 host stub 的 `musaLaunchKernel` 调用误报为 GPU fault instruction。

## 8. 相关 repository skills 与强制工作流

新 agent 开始前先读：

```text
/data/feihu/xllm-git-master-python/AGENTS.md
```

### 8.1 MUSA 容器构建 skill

```text
/data/feihu/xllm-git-master-python/.agents/skills/musa-container-build/SKILL.md
```

该 skill 的核心要求：构建、链接、server、correctness 和 benchmark 都必须在容器内；不能在 `/data/feihu` host 环境直接构建，不能运行 bare Ninja。

skill 示例写的是普通 `xllm-git-master`，本任务必须使用用户指定的 Python tree：

```bash
ssh dev92
docker exec xllm-musa2.9.1-sdk5.1-dev bash -lc '
  set -euo pipefail
  cd /workspace/xllm-git-master-python
  MAX_JOBS=16 NINJA_TARGET=xllm ./_build_cuda_graph_musa.sh
  stat build/lib.linux-x86_64-cpython-310/xllm/xllm
'
```

当前 index 损坏导致 skill 中的 `git status`/`git diff --check` gate 无法执行；必须在报告中明确写为 blocked，不能伪称 clean。其余 container/build/final-link gate 仍然必须执行。

### 8.2 unit-test / 复现脚本 skill

```text
/data/feihu/xllm-git-master-python/.agents/skills/add-unit-test/SKILL.md
```

本轮新增的稳定测试资产是：

```text
tests/python/test_mudnn_topk_regression.py
```

它是 standalone 回归，不是 integrated reproduction。若新增 C++ test/CMake target，必须先按该 skill 检查最近测试结构和 CMake 约定。不要重新加入之前已删除的临时 C++ GTest，除非它能稳定复现集成故障。

### 8.3 源码修改和 review

修改或 review `xllm/` 前必须完整阅读：

```text
.agents/skills/code-review/SKILL.md
.agents/skills/code-review/references/custom-code-style.md
```

若以后需要提交，再读：

```text
.agents/skills/git-workflow/SKILL.md
```

但当前 index 已损坏，未获得用户授权前不要进行 commit/index 修复。

## 9. 下一步建议：按优先级执行

### P0：先建立可重复、可比较的 integrated gate

不要继续只跑 standalone。以当前 CPU fallback 二进制为 baseline，保存以下信息：

1. binary SHA256；
2. sampler/worker/llm_worker SHA256；
3. GPU id 和 `mthreads-gmi` 快照；
4. 完整 server env；
5. 32 个 prompt、seed、beam width、output length；
6. warmup 和两轮 1000-request measured run；
7. request success、prompt signature、QPS/mean/p50/p90/p99；
8. 启动前后的 dump 文件列表。

### P1：验证“TopK 专用 workspace 生命周期”而非全局关闭 cache

优先候选是一个窄范围 GPU 路径：

1. TopK handle 绑定当前 MUSA stream；
2. 只对 TopK workspace 使用 direct 或持久化 allocation；
3. 不设置进程级 `PYTORCH_NO_MUSA_MEMORY_CACHING=1`；
4. 避免每次 `musaMalloc/musaFree` 带来的全局同步；
5. workspace 的生命周期必须覆盖异步 TopK 完成；
6. debug build 在 TopK 返回后检查 dtype/range/value-gather 关系；
7. release candidate 删除同步 D2H 检查。

可以在 TorchMUSA `TopK.cpp` 做最小实验，也可以先在 xLLM 做 custom wrapper。无论选哪条路线，都要保留 current stream 语义，并与原始 `torch::topk` 做 matched A/B。

### P2：如果 targeted workspace 仍不稳定，转查 graph executor 上游写入

由于 standalone cached allocator 也通过，另一个重要分支是：TopK output storage 或相邻 storage 被 graph replay/其他异步 kernel 写坏。

建议记录：

- TopK values/indices/workspace 的地址、size、stream；
- 调用前后 allocator block 地址是否被快速复用；
- TopK 输出到 software-beam 消费之间的 kernel dispatch 顺序；
- graph capture/replay 是否持有或重用同一 storage；
- 禁用 graph 的 matched integrated run 是否能长时间稳定。

不要一次加入大量 D2H/synchronize；它会改变 timing，使问题暂时消失。优先使用 device-side guard、allocator trace 或低扰动地址日志。

### P3：最终准入条件

GPU 修复只有同时满足以下条件才能替换 CPU fallback：

1. standalone Python test 通过；
2. direct muDNN current-stream test 通过；
3. integrated software-beam 两轮独立启动，各 1000/1000；
4. `unstable_prompt_count=0`；
5. 没有新 dump；
6. eager/graph 至少各一轮 correctness；
7. matched performance 明显优于 CPU fallback；
8. 删除/关闭高开销 debug 同步和 D2H 检查；
9. 完整 `xllm` target 在权威容器内重新链接成功；
10. index 问题解决后补做 `git diff --check` 和 style review。

## 10. 不要重复的路线与协同约定

- 不要把 embedding `IndexSelectVectorKernel` 直接当成最初根因；它消费的是已经损坏的 token id。
- 不要用 standalone TopK PASS 宣告问题修复。
- 不要用提前退出 run 的 QPS 做性能结论。
- 不要默认开启 `MUSA_LAUNCH_BLOCKING=1` 或全局关闭 caching allocator。
- 不要擅自清理/重建 git index、reset 工作树或覆盖无关修改。
- 每次异常退出都检查新 dump，并基于该次 binary/library hash 重新解析符号。
- 对每个实验写清楚：只改变了什么、哪些保持一致、结果文件路径、是否生成新 dump。

建议两个 agent 分工：一个维护 integrated reproduction、结果表和 dump 证据；另一个只做 TopK workspace/current-stream 的最小实现与 microbench。每轮交换源码 SHA256 和结果路径，避免互相覆盖实验文件。

## 11. 2026-08-27 下午：已落地的 graph-on 修复（第三人 review）

> 作者：下午 A/B agent。用户标准：`qwen06_errorfile_client.py` 在 Qwen3-0.6B、`beam_width=128`、graph-on 上通过。
> 未连 GitHub；未修 `.git/index`；未 commit。权威树仍是 `/data/feihu/xllm-git-master-python`。

### 11.1 一页结论（取代 §0 的“CPU fallback 安全状态”）

Software-beam 在 graph-on 上原来有**两层独立故障**，必须都修，测试脚本才会过：

1. **TopK 产出损坏的 int64 indices**（低 32-bit 是 FP32 score 字节）。host 侧 `LOG(FATAL) MUSA_TOPK_INDEX_CORRUPTION` 能在 embedding 之前抓住。根因是 graph capture/replay 与 caching allocator 别名 TopK 的 values/indices/workspace，不是 muDNN 对 `[N, 151936] k=256` 本身算错。graph-off 同二进制不崩。
2. **piecewise prefill padding 把 pad query 的 KV 写进最后一个真实 token 的 cache slot**。pad query 在更后的位置，K/V 并不等于 last-token。该写会污染后续每一层。表现是：bucket 对齐长度（`1,4,8,16,…,1024`）稳定且与 graph-off 签名一致；非对齐长度（`2,3,7,15,…,959`）每次请求换签名，且永不匹配 graph-off。`reshape_paged_cache` 已对 `slot_id < 0` early-return，decode/packed padding 本来就用 `-1`。

没有把 TopK 退回 CPU。debug 仍保留：越界/损坏 indices 现场 `LOG(FATAL)`。coredump 有用；`run_xllm_musa.sh` 仍 `ulimit -c 0`。

用户标准已达到：graph-on + piecewise + 官方 client，独立重启后 **1000/1000、`unstable_prompt_count=0`、签名与 graph-off 一致、无新 dump**。

§9 P3 里**尚未关闭**的项：高开销 `musaDeviceSynchronize` + indices D2H 扫描仍在 wrapper 里（用户要求现场崩，不要盖住）；index 损坏故还没有 `git diff --check` / commit。

### 11.2 改了哪些文件

| 文件 | 作用 |
|---|---|
| `xllm/core/kernels/musa/musa_topk.cpp` | 新增。MUSA software-beam TopK：current stream 上的 muDNN C API；grow-only `musaMalloc` 的 input/values/indices/workspace（不在 in-flight kernel / captured graph 下 `musaFree`）；TopK 前后以及结果拷回 torch tensor 后 `musaDeviceSynchronize`；D2H 扫描 indices，越界则 `LOG(FATAL) MUSA_TOPK_INDEX_CORRUPTION`。无 CPU fallback。 |
| `xllm/core/kernels/musa/musa_ops_api.h` | 在 `random_sample` 旁声明 `topk(input, k)`。 |
| `xllm/core/kernels/musa/CMakeLists.txt` | `musa_topk.cpp` 加入 `_TORCH_MUSA_KERNEL_SRCS`。`libmudnn.so.3` 显式链到 `cuda_kernels`（C API）。**不要** `DEPS mudnn`：`_build_cuda_graph_musa.sh` 把 `libmudnn.so` 指到 `libmudnncxx.so`。 |
| `xllm/core/framework/sampling/sampler.cpp` | beam top-logprobs：`USE_MUSA` 下走 `xllm::kernel::musa::topk(logprobs, params.max_top_logprobs)`。已删除 `kMusaMaxStableTopK` / `musa_safe_topk` CPU 路径。 |
| `xllm/core/runtime/musa/musa_graph_executor_impl.cpp` | piecewise prefill pad 的 `new_cache_slots` 从“复制 last-token slot”改为 `fill_(-1)`。 |

`sampler.cpp` 随机采样 fast path 上未加保护的 `sample_logits.topk` **没动**（不是 beam top-logprobs 路径）。

`FORCE_CMAKE=1` 被损坏的 git index 挡住。`musa_topk.cpp.o` 曾手写进 `build.ninja`（抄 `random_sample.cpp` 规则），并在 `-lmudnn` 旁插入 `libmudnn.so.3`。改 CMakeLists 后要 `touch build.ninja`，避免 ninja 重跑 cmake。

### 11.3 关键实现要点

**TopK wrapper**（`musa_topk.cpp`）：

- 只接受 2-D FP32 last-dim、largest+sorted。
- thread-local `mudnnCreate` handle；每次 `mudnnSetStream(..., c10::musa::getCurrentMUSAStream().stream())`。这里没有 `torch::` 等价物，现有 musa 文件也用 `c10::musa`。
- `DeviceBuffer` 是 `final` class（struct 不允许 member function）。
- grow-only：容量不够就再 `musaMalloc`，旧块不释放，避免 graph 仍点名旧地址。
- 校验只查 index 是否落在 `[0, vocab)`。范围内的错 id 不会 FATAL；那是 §11.1 第 2 层（padding）的问题，不是这条检查的职责。

**Sampler**：

```text
xllm/core/framework/sampling/sampler.cpp
  #if defined(USE_MUSA)
    xllm::kernel::musa::topk(logprobs, params.max_top_logprobs);
  #else
    logprobs.topk(params.max_top_logprobs, /*dim=*/-1);
  #endif
```

`max_top_logprobs = 2 * beam_width = 256`，由 `sampling_params.cpp` 设置。MUSA 没有 `BeamSearcher`（NPU-only）；software-beam merge 在 host `sequences_group.cpp` 里消费 `top_tokens`。

**Pad-slot 修复**（`musa_graph_executor_impl.cpp`，`MusaGraphPersistentParam` 更新 `new_cache_slots` 处）：

旧逻辑（错误）：

```text
piecewise_prefill_pad:
  pad cache slots = last actual token's slot
  // 注释声称 pad K/V 与 last token 相同，overwrite 无害
```

新逻辑：

```text
piecewise_prefill_pad:
  pad cache slots.fill_(-1)
  // pad query 在更后的位置，K/V 不同；写入 last slot 会污染后续层
  // reshape_paged_cache_kernel: if (slot_id < 0) return;
```

`else` 分支未改：decode / packed piecewise 仍是 `-1`，其它仍是 `0`。

Qwen3 `qwen3.h` 里对 `new_cache_slots` 做 `floor_divide(64)/remainder(64)` stack 的是 `input_params.attention.device.new_cache_slots`。graph 路径的 `attn_metadata->slot_mapping` 来自 **flat** `persistent_new_cache_slots`。`flashinfer_attention.cpp` 要求 `slot_mapping.numel() == key.size(0)`，因此 `-1` 以 1-D int 进 `reshape_paged_cache`，不会先被 stack 成 `(block, offset)`。

### 11.4 构建

容器：`xllm-musa2.9.1-sdk5.1-dev`。只在容器内用 `_build_cuda_graph_musa.sh`。不要 host compile、不要 `FORCE_CMAKE=1`、不要 `FULL_RESET`、不要裸 ninja。

```bash
docker exec xllm-musa2.9.1-sdk5.1-dev bash -lc '
  cd /workspace/xllm-git-master-python
  FORCE_CMAKE=0 MAX_JOBS=16 NINJA_TARGET=xllm \
    CPATH=/usr/include/python3.10 ./_build_cuda_graph_musa.sh
'
```

| 时间 | SHA256 | 内容 |
|---|---|---|
| 2026-08-27 12:59:38 | `d167e87abcc25db08b7b0c118e8b54038d3b613dabaffe7a14281f618db38574` | 第一版 GPU wrapper，仍会 graph-on 崩 |
| 2026-08-27 14:20:45 | `5e8cdf9bf9b4eb71b44aeb89eb431f9236df7a2cd94a5bef36e15c898c2a4400` | isolated buffer + device sync；崩没了，graph-on `unstable=22` |
| 2026-08-27 14:32:25 | `7186c21ceceea68a9174176f743fd4dfde395f3297aea6b1f8b8ba0b629c4667` | 加上 pad-slot `-1`。当前准入二进制 |

最新构建日志：`build_logs/build_cuda_graph_musa_20260827_143055.log`
二进制：`build/lib.linux-x86_64-cpython-310/xllm/xllm`
`DT_NEEDED` 同时有 `libmudnn.so.3` 和 `libmudnncxx.so.3`。

### 11.5 测试协议（不要另写 client）

```text
client:     /data/feihu/qwen06_errorfile_client.py
workload:   /data/feihu/validation_qwen06_errorfile_20260824/workload.jsonl
model:      /data/nfs_shared/models/Qwen3-0.6B
beam:       128
output_len: 3
C:          1
warmup:     8
seed:       1234
block_size: 64
max_seqs:   128
chunked:    false
overlap req: false（MUSA Qwen3 仍会 force overlap on）
wrapper:    /data/feihu/xllm-qwen06-beam-wrapper.sh
            --enable_beam_search_kernel=true --beam_width=128
```

Qwen3 graph 必需：

```text
XLLM_QWEN3_ENABLE_GRAPH_EXPERIMENTAL=1
ENABLE_GRAPH=1
ENABLE_PREFILL_PIECEWISE_GRAPH=1
ENABLE_GRAPH_DECODE_NO_PADDING=0
XLLM_BIN=/workspace/xllm-qwen06-beam-wrapper.sh
```

client 的 `summary.protocol.graph` **写死为 true**，不能当真实 graph 开关。§6.1 的 CPU fallback 1000（`final_safe_repeat`）server log 实际是 `enable_graph: 0`。

### 11.6 A/B 证据

| 实验 | 二进制 | graph / piecewise | GPU / port | 结果 | 产物 |
|---|---|---|---|---|---|
| A 旧 wrapper graph-on | `d167e87…` | on / on | 5 / 19270 | warmup 后 input_len=7、bucket 8 capture 时 `MUSA_TOPK_INDEX_CORRUPTION` | `validation_qwen06_mate026_20260827/gpu_persistent_topk/server/` |
| B 同 wrapper graph-off | `d167e87…` | off / off | 5 / 19271 | 200/200，errors=0，unstable=0，QPS 6.91 | `.../ab_graph_off/result200.json` |
| C isolated TopK graph-on | `5e8cdf9…` | on / on | 6 / 19280 | 200/200，**无崩**，unstable=**22**；只有对齐长度匹配 B | `.../ab_graph_on_isolated/result200.json` |
| D pad-slot `-1` graph-on | `7186c21…` | on / on | 4 / 19283 | 200/200，unstable=0，**32/32 匹配 B**，QPS 9.55 | `.../ab_graph_on_padfix/result200.json` |
| D 同进程 1000 A | `7186c21…` | on / on | 4 / 19283 | 1000/1000，errors=0，unstable=**2**（各 1 次 outlier：len=1 与 len=639），998/1000 匹配 B，QPS 9.42 | `.../ab_graph_on_padfix/result1000_a.json` |
| D 同进程 1000 B | `7186c21…` | on / on | 4 / 19283 | 1000/1000，unstable=0，1000/1000 匹配 B，QPS 9.55 | `.../ab_graph_on_padfix/result1000_b.json` |
| D **独立重启 1000** | `7186c21…` | on / on | 4 / 19283 | **1000/1000，unstable=0，1000/1000 匹配 B**，QPS 10.25 | `.../ab_graph_on_padfix/result1000_restart.json` |

独立重启 server log：`validation_qwen06_mate026_20260827/ab_graph_on_padfix/server_b/xllm_Qwen3-0.6B.log`。

相对 §6.1 CPU fallback 1000（QPS ~5.29–5.46，且当时是 graph-off）：graph-on GPU 路径大约 **1.8×–1.9×**。不要用提前退出窗口的 QPS。

本轮 **没有新 `.mudmp`**。最新已知 dump 仍是：

```text
/data/feihu/xllm-git-master-python/core_2026-08-26_23:11:20.164_worker38092_67441.mudmp
```

### 11.7 对 review 的明确约束

- 不要把 CPU fallback 加回来当“修复”。
- 不要把 `MUSA_LAUNCH_BLOCKING=1` 或进程级 `PYTORCH_NO_MUSA_MEMORY_CACHING=1` 当修复。
- 不要修 / 重建 / reset `.git/index`。
- 不要改 `/data/feihu/xllm-git-master`、`xllm-HEAD`、`xllm-HEAD-PR`。
- 不要 `pkill -f`。本轮起过的具体 PID 需要停时再停。
- 27B 若在跑，避开 GPU 0/2。
- 构建只走 python 树 + 容器内 `_build_cuda_graph_musa.sh`。
- 关闭 debug sync / D2H 前，必须再跑一轮独立 graph-on 1000；那是 P3.8，不是这次用户标准。
- 1000 A 的两次 singleton outlier 没有在独立重启 1000 上复现。若 review 要卡 P3.4 的“每一轮都是 0”，应再开一轮独立启动，而不是据此回退 pad-slot 或 CPU fallback。

### 11.8 Review checklist

1. `musa_topk.cpp`：无 CPU fallback；无 `FLAGS_`；无 try/catch 兜底；`DeviceBuffer` 是 class；primitive 不用 `auto`；越界走 `LOG(FATAL)` / `CHECK`。
2. 链接的是 `libmudnn.so.3`（C），不是 cmake `DEPS mudnn` → `libmudnncxx`。
3. `sampler.cpp` 只改了 `max_top_logprobs > 0` 的 beam 路径。
4. pad-slot 只改 piecewise prefill 分支；decode/packed 的 `-1` 语义未扩大误伤。
5. graph 路径 `slot_mapping` 仍是 flat int，`-1` 能被 `paged_cache.cu` 跳过。
6. 用 §11.6 的 JSON 核对 `ok/errors/unstable_prompt_count`，不要信 client 里写死的 `"graph": true`。
7. 对照 §9 P3：1–7、9 对 graph-on 准入已满足（standalone / direct muDNN 仍以本文上午记录为准）；8 和 10 未做。
8. style：改 `xllm/` 前必须通读本地 `.agents/skills/code-review/references/custom-code-style.md`。用户调试期间禁止为对拍而去连 GitHub。
