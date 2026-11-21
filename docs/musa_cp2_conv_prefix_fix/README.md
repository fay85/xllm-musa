<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# MUSA CP2 + MTP 卷积 prefix 状态形状修复

本次修复解决 Qwen3.8-27B-FP8 在 CP2 Prefill、MTP K3 和 prefix cache 同时开启时，卷积边界历史写入扩展状态缓存造成的形状不匹配。修复只更新已提交的卷积历史，保留 MTP 的投机状态、原始缓存布局及独立读写槽语义。

**验证状态（2026-11-21）：**旧算子已复现同一错误；修复后7项回归测试通过；真实模型32条启动检查和47条专项请求全部回答正确。198题 GPQA Diamond C16 已另行启动，完整准确率和长时间稳定性仍待其完成，不能由这些短时检查推导。

## 1. 故障表现与传播链

发生问题的服务为一个 TP1/CP2 Prefill 和一个 TP1/CP1 Decode，客户端并发16，MTP K3、prefix cache、PCP multibatch 开启。

Prefill rank1 首先抛出：

```text
shape mismatch: value tensor of shape [10240, 3] cannot be broadcast
  to indexing result of shape [1, 10240, 6]
```

符号化调用栈包含：

```text
xllm::kernel::musa::causal_conv1d_prefill
Qwen3GatedDeltaNetBaseImpl::forward
Qwen3HybridDecoderLayerImplBase::forward
```

rank1 退出后，rank0 仍等待它参与 collective，之后出现 MCCL watchdog 超时。因此，本事件的首发故障是卷积缓存赋值的形状异常；后续 collective 超时是另一 rank 退出的结果。

两次已保留的旧 CP2 运行均出现过该异常，分别只完成85题、86题，不能作为完整198题准确率结果。

## 2. 根因：逻辑历史长度与物理状态容量不同

设卷积宽度为 `W`，历史长度为 `H = W - 1`，通道数为 `D`，本次写入序列数为 `B`。

| 对象 | 本例形状 | 含义 |
| --- | --- | --- |
| `weight` | `[10240, 4]` | 每通道4个卷积系数 |
| 2D `prefix_x` | `[3, 10240]` | 单序列的3个历史 token |
| 3D `prefix_x` | `[B, 3, 10240]` | 多序列的历史 token |
| 转置后的 `state_rows` | `[B, 10240, 3]` | 要写入的已提交历史 |
| MTP K3 的 `conv_state` | `[slots, 10240, 6]` | 前3列已提交历史，后3列投机槽 |
| `conv_state[cache_indices]` | `[B, 10240, 6]` | 被索引到的完整物理状态行 |

`causal_conv1d_prefill` 的旧路径在收到 CP2 边界 prefix 时执行：

```cpp
torch::Tensor mutable_conv_state = conv_state;
mutable_conv_state.index_put_({cache_indices}, state_rows);
```

这里写入的是完整物理状态行。未扩展状态时，最后一维恰好等于3，问题被掩盖；MTP 把容量扩展到6后，3列历史便无法赋值到6列目标。

该函数还有 token-major 优化路径。默认配置中，满足 `x.stride(1) == 1` 且卷积宽度4时可以直接走此路径，它没有上述整行赋值。只测这一路径会漏掉故障。新增测试使用通道步长2的真实 strided 输入，确定性进入原有非 token-major 路径，不依赖关闭 CP2 或 MTP 来复现。

## 3. 修复实现与状态不变量

修改文件：

- [gdn_prefill.cpp](../../xllm/core/kernels/musa/gdn_prefill.cpp)
- [musa_pcp_multibatch_test.cpp](../../tests/core/layers/musa_pcp_multibatch_test.cpp)

关键修改：

```cpp
const int64_t history_len = weight.size(1) - 1;
torch::Tensor committed_conv_state =
    conv_state.narrow(/*dim=*/2, /*start=*/0, history_len);
committed_conv_state.index_put_({cache_indices}, state_rows);
```

`narrow` 返回共享底层存储的 view。用于赋值的目标现在是 `[B, D, H]`；传给后续卷积内核的仍是完整 `conv_state`，其地址、物理容量及 stride 不变。

修复保持以下不变量：

1. 只写指定 `cache_indices` 的前 `H` 列，投机尾部 `[H, state_len)` 不改写。
2. 不修改无关槽，也不把独立的 `read_cache_indices` 槽误当成写入目标。
3. 显式 prefix 已落到写入槽后，继续沿用原有逻辑清除本次卷积的独立读取索引，让内核读取新恢复的历史。
4. 历史长度由卷积宽度推导，不把 MTP 扩展容量当成历史长度，也不固定假设所有状态缓存都只有3列。
5. 校验卷积权重和缓存维数、缓存容量、2D prefix 的单序列约束，以及 prefix 的历史长度、batch 和通道数。
6. 原有 token-major 路径保持不变；没有新增异常捕获、执行失败后的回退或请求重试。

没有关闭 CP2、PCP multibatch、MTP、prefix cache；没有把6列缓存截断成3列，没有分配替代缓存来掩盖布局问题。

## 4. 回归测试如何避免“只是不崩溃”

新增两个 GPU 测试：

### `PrefixPreservesExpandedMtpStateAndReadSlots`

覆盖12组组合：

- 物理状态长度3、6、10，分别检查基础历史及不同扩展容量。
- 三条变长序列，长度为1、2、7，包含输入短于历史窗口的边界。
- 连续输入与通道步长2的 strided 输入。
- SiLU 开启和关闭。
- 3D prefix；写槽1/3/5与独立读槽2/4/6分离。

状态长度10只表示扩展容量的算子级覆盖，不代表已经验证端到端 MTP K7 服务。

### `SingleSequencePrefixMatchesGpqaCrashShape`

覆盖4组组合：

- 通道数10240、物理状态长度6，匹配原始崩溃尺寸。
- 2D与3D prefix。
- 连续与 strided 输入。

两项测试合计16组实际 GPU 算子调用。卷积输出与独立 CPU float32 参考比较，参考计算显式拼接 prefix 与 query，并应用4个卷积系数、bias和可选 SiLU；转为 BF16 后使用 `rtol=1e-2, atol=1e-3` 比较。

完整状态缓存还与参考缓存逐元素精确比较。缓存预置为哨兵值，因此可以同时检查最终已提交历史是否正确，以及投机尾部、独立读槽和无关槽是否保持原值。测试没有把尾部数据清零、裁掉或忽略。

此外运行了4项既有 PCP 上下文/分片测试及1项既有卷积多序列拆分/合并测试，总计7项测试。

## 5. 构建和运行方法

### 已有 MUSA CMake 构建环境

在已经完成依赖配置、启用 MUSA 和 `BUILD_TESTING=ON` 的构建目录中，只构建相应目标。以下路径为示例，需要指向当前 checkout 对应的构建目录；源码、输出和可写缓存均置于容器 `/workspace` 下。

```bash
export CP2_BUILD_DIR=/workspace/build/xllm-cp2-prefix-fix
cmake --build "$CP2_BUILD_DIR" --target musa_pcp_multibatch_test -j 4
```

GPU 必须预先检查为空闲或已明确分配给本次验证。以下示例使用GPU4：

```bash
export MUSA_VISIBLE_DEVICES=4
export CUDA_VISIBLE_DEVICES=4
export MTHREADS_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_musa/lib:/usr/local/musa/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}

ctest --test-dir "$CP2_BUILD_DIR" --output-on-failure \
  -R '^(MusaPcpMultiBatchTest\.|MusaPcpMultiBatchGpuTest\.(PrefixPreservesExpandedMtpStateAndReadSlots|SingleSequencePrefixMatchesGpqaCrashShape|CausalConv1dAndMateGdnSplitMatchUnsplitBatch)$)'
```

若直接调用测试二进制，使用同一组7项测试的过滤器：

```text
MusaPcpMultiBatchTest.*:MusaPcpMultiBatchGpuTest.PrefixPreservesExpandedMtpStateAndReadSlots:MusaPcpMultiBatchGpuTest.SingleSequencePrefixMatchesGpqaCrashShape:MusaPcpMultiBatchGpuTest.CausalConv1dAndMateGdnSplitMatchUnsplitBatch
```

### 本次实际增量构建

为保留并行 GPQA 基线，本次没有覆盖正在运行的二进制，也未重建第三方依赖。复用已有编译参数和依赖，只编译测试源文件及修改后的 `gdn_prefill.cpp`，复制基础 `libcuda_kernels.a` 到独立目录，替换唯一的 `gdn_prefill.cpp.o` 成员后重新链接。

环境：`s5000_dev15 / xllm_1110_dev`。

| 记录 | 值 |
| --- | --- |
| 验证基础源码提交 | `b9ff5a8f979a84da528eefdca9f4f0fe1e5675da` |
| 原始修复提交 | `545bfb471471b28fa8aad237f5f23a0ecf60edf1` |
| 原始修复工作树 | `/workspace/xllm-cp2-conv-fix-20261121` |
| 证据目录 | `/workspace/cp2_conv_fix_20261121` |
| 基础二进制 SHA256 | `fce69d738a05fbafca5e0249ed23b34bb97a65cb0d716b7d8fb475a0eb4f9af6` |
| 修复二进制 | `/workspace/cp2_conv_fix_20261121/build/bin/xllm_cp2_conv_fix` |
| 修复二进制 SHA256 | `3f434c13d100dda020da36ddf75759214d38edd2efe17ea663e2ac2278d3c8ad` |

证据目录的 `build_fix.py`、`run_regression.py`、`build_result.json`、`source_provenance.json` 和 `build/jobs/*.json` 保存了实际命令、链接输入及源码 hash。这些本机路径依赖已有缓存，不是任意干净环境可直接执行的通用构建器；上面的 CMake 方式使用项目正式测试目标。

同一新增测试先链接旧基础算子，再链接修复算子：

| 验证 | 结果 |
| --- | --- |
| 旧算子 + 原始10240通道形状测试 | 测试进程返回1，复现同一3列→6列异常 |
| 新算子 + 选定7项测试 | 测试进程返回0，7项全部通过 |
| clang-format 20.1.6 | 两个C++文件检查通过 |
| `git diff --check` | 通过 |

日志和XML：`baseline_regression.log/.xml`、`fixed_regression.log/.xml`。重新构建旧算子复现时，应在隔离构建中保留新增测试，但使用修复前的生产实现；只运行修复前版本原本没有新增测试的测试集合，不能验证本次回归。

### 发布分支上的再次检查

本次发布基于远端 `feihu-1016-dev` 的 `4bcd18464f1a140a984d4892d19f8655ed91aec3`，使用独立工作树 `/workspace/xllm-cp2-publish-20261121`。移入后的生产文件与原始已验证版本逐字节一致；目标分支原有10项测试全部保留，新增上述2项测试，并按 clang-format 20.1.6 格式化。

发布前重新编译该工作树的 `gdn_prefill.cpp` 和完整测试文件。目标测试文件还包含分支已有的 compact-vocab API 检查；为匹配该API，从同一目标分支重编译了 `row_argmax.cu`，但本次提交没有修改这个文件，也没有移除或替换已有测试。随后在GPU4上再次运行所选7项回归，全部通过。

这次检查复用其余已构建依赖，不是整个发布分支的完整重构。详细命令、链接输入、日志及XML在 `/workspace/cp2_conv_fix_20261121/publish_validation`；首次混用旧缓存时的缺符号链接记录也保留，最终测试使用补齐目标API后的实际二进制。

## 6. 真实模型 CP2 + Decode 检查

拓扑为 GPU5/6 上 TP1/CP2/DP1 Prefill，以及 GPU7 上 TP1/CP1/DP1 Decode。模型为 `/data/models/Qwen3.8-27B-FP8`，MTP K3、prefix cache、PCP multibatch 开启。Decode graph 容量64，覆盖 `(1+K) * C16`；linear-state cache slots为16；Prefill piecewise graph与overlap关闭。

Decode 沿用此前诊断运行的 `libasan.so.6` 分配器预加载，Prefill未预加载；二进制没有完整 ASAN 编译插桩。这不是完整内存安全证明，也没有验证普通分配器配置。

| 检查 | 请求数 | 结果 |
| --- | ---: | --- |
| C16启动检查：数列输出与 thinking 算术 | 32 | 全部正确，Prefill/Decode记录和usage对应 |
| 冷请求 C1/2/4/8/16，输入至少512 token | 31 | 全部正确 |
| 重放最后16条请求，验证热缓存复用 | 16 | 全部正确 |

专项检查共47条请求。日志中的 `[PCP_PERF_BATCH]` 确认实际执行了 CP2 多序列，观察到最大实际 batch 为9；客户端并发16不等于引擎实际 batch 为16。热请求复用了14336个 prefix-cache token。Decode 投机提议1833 token、接受1829 token，说明 MTP 实际执行；简单数列的接受率不能代表 GPQA 或客户多模态负载。

检查结束时，两条 Prefill rank和Decode均存活，未出现该形状异常、ASAN错误报告或MCCL watchdog超时。增加的 `XLLM_PCP_PERF_LOG=1` 只用于确认执行路径；本轮没有性能提升结论。

真实模型证据在：

```text
/workspace/gpqa_pd_cp2_convfix_c16_20261121/
  smoke.json
  smoke_path_audit.json
  focused_smoke.json
  focused_smoke_audit.json
  focused_audit_correction.json
  frozen_evaluation_audit.json
  prefill_r0.log
  prefill_r1.log
  decode.log
```

专项审计曾误读缓存字段 `cached_tokens`，实际引擎终止记录使用 `num_prefix_cache_tokens`。更正后对同一47条响应及原始日志重新审计，没有重跑请求或改写答案。原始准备阶段错误记录保留，不能把该辅助脚本错误当成引擎缓存未命中。

## 7. GPQA 长测与尚未覆盖的结论

完整 GPQA Diamond 198题、C16重跑独立保存在上述运行目录。使用与前轮相同的prompt、选项排列、gold、`temperature=1/top_p=.95/top_k=20`、thinking/xhigh、`max_tokens=65536` 和每题 `seed=42+index`；每题一次请求，不重试、不补答。`frozen_evaluation_audit.json` 记录协议一致性。

首次启动评分前因新目录缺少CSV而退出，补齐并校验数据hash后启动正式评分；当时没有评分请求。准备阶段记录单独保留，不混入正式结果。

完成后应一起检查：

- 198条原始SSE与逐题结果，按198分母计分。
- 请求错误、截断、答案未解析和真正答错分别统计。
- 两rank及Decode的存活与异常日志。
- 实际MTP计数、prefix-cache计数、Prefill/Decode终止记录及usage对应。

本README记录的是原始修复提交及其验证二进制。将该补丁移入后续分支时，不能把这里的二进制hash当成新分支完整构建的hash，也不能据此宣称后续整个分支已完成长测。当前证据支持卷积prefix形状修复、数值/状态回归及短时真实请求正确性；完整准确率、长期稳定性、Prefill piecewise graph和性能结论需要各自相应验证。
