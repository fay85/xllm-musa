# MUSA 后端 KDA 算子接入说明

## 当前实现范围

当前提供通用的 C++ 算子接口，位于
[`kda.h`](../../xllm/core/kernels/musa/kda.h) 和
[`kda.cpp`](../../xllm/core/kernels/musa/kda.cpp)：

- `xllm::kernel::musa::flash_kda`：对应当前安装的 `flash_kda.fwd`。
- `xllm::kernel::musa::flash_kda_workspace_size`：对应 `flash_kda.get_workspace_size`，返回 `0`。

接口不依赖特定模型配置，没有固定模型的 batch、heads 或序列长度分桶。
这次接入提供算子封装、构建入口和测试，没有新增模型注册或模型层调用。
`mate.kda.gated_delta_rule_decode` 是另一套接口；`mate.chunk_kda` 额外提供的状态池索引、chunk 状态快照和 checkpoint 参数，也不属于这里映射的 `flash_kda.fwd` 参数范围。

## 与 GDN 一致的接入方式

运行链路如下：

```text
xLLM C++ 校验并整理张量
  → get_function(uri, function_name)
  → 从 FLASHINFER_OPS_PATH 加载预编译模块 .so
  → MATE prepare 内核
  → MATE recurrence 内核
  → 写入调用方的输出和最终状态
```

实现复用 `musa_tvmffi_stream.cpp` 中的模块与函数缓存、`to_ffi_tensor` 张量转换，以及 `TvmffiStreamGuard` 的 stream 绑定和同步。
张量通过 DLPack 传递设备指针和元数据，不把激活、状态或设备序列长度拷到 CPU 计算。
临时 workspace 在 guard 之前分配，guard 在这些张量释放前完成必要的 stream 依赖处理；dense 输出的 Torch 回写发生在 guard 退出之后。

与现有 GDN 的 MATE 路径相同，Python 只参与离线产物生成。xLLM 运行时不调用 Python、不触发 JIT，也不直接编译或 launch MATE kernel 模板。
`kda.cpp` 使用已有的 TVM-FFI、TorchMUSA 和 MUSA 构建依赖，不包含 MATE/MUTLASS kernel 头文件，也没有额外的 kernel 编译参数。
本算子采用 MATE，无 muDNN 或 muBLAS 替代分支。

## 参数和张量约定

以下约定以当前安装的 `flash_kda/flash_kda_interface.py`、`mate/kda.py` 和导出模块 ABI 为准。
`B` 表示 dense batch，`T` 表示每条序列长度，`S` 表示打包后的总 token 数，`N` 表示逻辑序列数。

| 参数 | 当前行为 |
| --- | --- |
| `q`、`k` | dense 为 `[B,T,Hqk,128]`，二者 shape、dtype 相同 |
| `v`、`g`、`out` | dense 为 `[B,T,Hv,128]`，要求 `Hv > 0` 且 `Hv % Hqk == 0` |
| 激活 dtype | `q/k/v/g/beta/out` 使用相同的 FP16 或 BF16 |
| `beta` | 原始 logits，dense 为 `[B,T,Hv]`，由 MATE 内核执行 sigmoid |
| `scale` | 可选；默认 `1 / sqrt(128)`，自定义值传给 MATE |
| `a_log`、`dt_bias` | 成对提供或同时省略；连续 FP32，shape 分别为 `[Hv]`、`[Hv,128]` |
| `lower_bound` | 传给 MATE，不固定为某个模型的数值 |
| `initial_state` | 可选，连续 `[N,Hv,128,128]`，V-first 布局；dtype 为激活 dtype 或 FP32 |
| `final_state` | 可选，调用方提供的输出 buffer；shape、dtype 与状态约定一致 |
| `cu_seqlens` | 可选，设备上的一维 int32/int64 累积序列长度，shape 为 `[N+1]` |
| workspace 查询 | 返回 `0` 表示不消费调用方显式 workspace；内部仍会分配临时显存 |

提供 `cu_seqlens` 时，Q/K 为 `[1,S,Hqk,128]`，V/G/out 为 `[1,S,Hv,128]`，beta 为 `[1,S,Hv]`；也接受省略外层 `1` 的 rank-3 激活和 rank-2 beta。
调用方需提供合法的累积长度：从 `0` 开始、非递减、最后一个值等于 `S`。封装不会为检查其数值而读取设备数据到 host。

没有 `cu_seqlens` 时，C++ 将 dense 输入整理为 packed 输入，并在设备上生成序列边界。
两种入口最终使用相同的 varlen/int64 FFI 模块；int32 序列长度在设备上转换。
外层 stride 可以不连续，激活的最内层维度要求连续；必要的 reshape/contiguous 可能产生设备拷贝。

省略初始状态表示零状态；省略最终状态表示不向调用方返回状态，当前内部仍分配最终状态 buffer。
提供的输出和最终状态被写入原 buffer；初始状态与最终状态可以使用同一 buffer。
空序列保留传入状态。Q/K L2 normalization 始终开启，与 `flash_kda.fwd` 一致。
有 gate 参数时，MATE 使用以下 log decay：

```text
lower_bound * sigmoid(exp(a_log) * (g + dt_bias))
```

没有 gate 参数时，当前安装的 MATE prepare 内核将 `g` 作为以 2 为底的 log decay。
状态 buffer 的存储 dtype 由调用方决定，chunk workspace 和中间计算精度由 MATE 内核决定。

## 为什么写死 chunk=16、head_dim=128

两个常量都来自当前 MATE chunk KDA 的编译与 workspace 契约，但含义不同。

### `kChunkSize = 16`

已安装的 `mate/jit/kda_ops.py` 定义：

```python
KDA_PREFILL_CHUNK_SIZE = 16
KDA_PREFILL_HEAD_DIM = 128
```

模块生成器将它们写入 prepare/recurrence 的编译模板。
因此，`16` 表示每个 chunk 处理的 token 数，用于计算 chunk 数量、临时矩阵尺寸和 workspace 偏移。
它不是最大序列长度，也不要求输入长度是 16 的倍数。尾部不足一个 chunk 由 MATE 内核处理；已有测试包含单 token、17 token 序列和跨 chunk 的 packed 输入。

### `kHeadDim = 128`

当前 `flash_kda.fwd` 转调 `mate.kda.chunk_kda`，后者显式检查 Q/K/V/G 最后一维必须为 `128`，否则报错：

```text
chunk_kda currently requires D=128.
```

因此，这是当前上游接口的真实 shape 限制，不是按某个模型新增的限制，也不是 KDA 算法本身只能使用 128。
另一套 decode 接口的支持范围不能用于推断这个 chunk 接口的支持范围。

### 对 FFI workspace 的影响

C++ 封装需要分配 MATE 两阶段共享的 workspace，因此必须使用与预编译模块一致的 chunk 和 head dimension。
当前每个 `(chunk record, value head)` 的布局如下，偏移与大小均以 FP16/BF16 元素为单位：

| 字段 | 元素偏移 | 元素数量 |
| --- | ---: | ---: |
| `decayed` | 0 | 4096 |
| `k_restored` | 4096 | 2048 |
| `p` | 6144 | 256 |
| `inverse` | 6400 | 256 |
| `total` | 6656 | 128 |
| 每条 record 的总跨度 | — | 6784 |

这些数值对应 MATE 的 `chunk_kda_workspace.hpp` 和 `mate.kda._allocate_kda_workspace`。
只修改 C++ 中的 `16` 或 `128`，不会改变已经编译好的内核，反而会造成 workspace shape、stride、offset 与内核不一致。

未来 MATE 支持其他 chunk 或 head dimension 时，需要同步更新模块生成、模块选择、workspace 契约和测试，不能只放宽输入检查。
当前产物脚本会在生成前检查 `(chunk, head_dim, offsets, record_stride)`，发现与预期契约不一致就停止导出。
当前 C++ 还选择并检查 MP31 目标，与所部署的 MATE 模块对应；它不是对所有未来 MATE 版本或设备能力的声明。

## 预编译产物与部署

当前脚本位于
[`compile_musa_kda.py`](compile_musa_kda.py)。
虽然目录带有模型名称，脚本本身不读取模型配置或权重，也不按模型 heads、batch 或序列长度生成专用模块。

在安装好 TorchMUSA、MATE 和编译工具链的环境中，于仓库根目录执行：

```bash
export FLASHINFER_OPS_PATH=/path/to/cached_ops
MATE_MUSA_ARCH_LIST=3.1 python tools/musa_glm5_3_vl_bench/compile_musa_kda.py --jobs 2
```

也可通过 `--ops-path /path/to/cached_ops` 指定导出目录。
脚本调用 MATE 官方的 `gen_kda_prepare_ops_spec` 和 `gen_kda_recurrence_ops_spec`，而不是复制或自行实现设备 kernel。
编译器版本适配由 MATE 自己的 builder 负责。

共生成 8 个模块：4 个 prepare 模块覆盖 FP16/BF16 与 gate 参数有/无，4 个 recurrence 模块覆盖 FP16/BF16 激活与对应激活 dtype/FP32 状态。
模块使用 varlen、int64 边界、Q/K 归一化、输入/输出状态均存在的配置；接口层的可选状态由 C++ buffer 分配表达。

加载路径为：

```text
$FLASHINFER_OPS_PATH/<function_name>/<function_name>.so
```

函数名模式为：

```text
prepare_kda_dtype_{fp16|bf16}_gate_{0|1}_varlen_1_normqk_1_seqlens_i64_arch_mp31
kda_recurrence_dtype_{fp16|bf16}_varlen_1_seqlens_i64_state_{fp16|bf16|fp32}_in_1_out_1_cs_0_arch_mp31
```

recurrence 的非 FP32 状态 dtype 必须与激活 dtype 对应。
运行环境需将 `FLASHINFER_OPS_PATH` 指向包含这些模块的目录，并能找到 TVM-FFI、MUSA runtime 和 TileLang loader。
必要时显式设置：

```bash
export XLLM_TILELANG_LIB=/path/to/tilelang/lib/libtilelang.so
```

模块缺失或 ABI 不匹配时，沿用现有 loader 的错误处理，不会在线编译或转为 host 计算。
升级 MATE 后应重新核对接口、函数命名和 workspace 布局，生成并验证匹配产物。

## 构建与验证

源文件加入 `xllm/core/kernels/musa/CMakeLists.txt` 的 `cuda_kernels` 目标。
测试通过 `tests/core/kernels/musa/CMakeLists.txt` 的 `cc_test` 注册。
使用仓库已有的 MUSA 构建环境，在准备好上述外部模块后执行：

```bash
cmake --build build/cmake.linux-x86_64-cpython-310 --target kda_test -j 4
MATE_DISABLE_JIT=1 build/lib.linux-x86_64-cpython-310/xllm/kda_test
```

2026-11-05 已有实机验证记录：MUSA SDK 5.1.0，使用当时安装的 MATE；8 个模块导出到 `/opt/mate027_cache/mp31/cached_ops_jd_k8d`，运行时以该目录为 `FLASHINFER_OPS_PATH`。
测试结果为 **3/3 通过**：

- 独立 CPU token recurrence 数值参考：覆盖 16 组激活 dtype、状态 dtype、dense/varlen、gate 参数组合，以及 GVA、非连续外层 stride、int32/int64 边界、空序列、尾 chunk 和省略可选状态的单 token 调用。
- Graph capture/replay：更新设备序列边界、Q 和初始状态，连续 3 次重放与 eager 结果一致。
- 默认 stream：连续 3 次调用，检查 FFI 输入输出依赖和临时 workspace 释放复用后的结果。

当时构建产物包含 `kda.cpp.o`，其编译命令没有 MATE/MUTLASS kernel 头文件路径或 KDA kernel 编译宏。
8 个模块的 `ldd` 检查没有未解析依赖；格式、Python 语法和 diff 检查通过。
这些是算子级验证记录，尚未进行由模型层调用该算子的 HTTP 推理验证。

## 实现参考

- 已安装的 `flash_kda/flash_kda_interface.py`：公开 wrapper 的参数及转调关系。
- 已安装的 `mate/kda.py`、`mate/jit/kda_ops.py`：shape 检查、生成配置、函数命名和 workspace 分配。
- 已安装的 `mate/flat/kda/chunk_kda_workspace.hpp` 与 prepare/recurrence FFI 模板：实际内存布局和导出 ABI。
- 当前仓库 `gdn_decode.cu`、`gdn_prefill.cpp`、`musa_tvmffi_stream.cpp`：模块加载、张量转换与 stream 接入约定。
- 本地 vLLM 的 `vllm/models/glm5next/common/kda.py::_flashkda_prefill`：raw gate/beta 与预分配 buffer 的语义映射；未照搬模型选择条件。
- 本地 vLLM 的 `vllm/third_party/flash_linear_attention/ops/kda.py`：V-first 状态与序列边界语义。
