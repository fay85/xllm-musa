# DeepSeek-V4-0731 MUSA 后续目标与验证记录

日期：2026-09-11。工作目录 `xllm0814/xllm-dsv4-musa-dspark`，分支
`dsv4/feat/musa-dspark-20260910`，本轮起点 `046fb526`。

## 目标及当前状态

1. 修复 target-only eager：两条连续的 2500/1500 严格全文 `1..750` 通过；
   格式化后重新编译并独立启动，再次通过 2500/1500 的 `1..750`，以及
   8192/1024 的 `1..512`。
2. Draft 全链路 Graph：尚未实现/验收。不能把 target q6 Graph 或单算子
   capture 通过写成 draft 全链路通过。
3. 多 batch：尚未测试。本轮计划递进 B2/B4/B8/B16，分别记录正确性和实际
   batch、客户端并发、输入/输出/总吞吐；必要时追加 8192/1024。

所有实验、启动器、单测构建脚本、源码 patch、二进制 SHA 和宿主机占用快照保留在：

```text
/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/dspark_goal_20260911
```

本轮未 push、未 reset GPU、未安装或升级共享 vcpkg/模型依赖。GPU 空档只使用
本任务拥有的占卡进程；停止进程前核对 PID/PGID/启动时间，宿主机
`10.20.34.28:22` 的 `mthreads-gmi` 用于确认资源是否释放。

## 一、Target-only eager 修复

### 已定位的问题

**默认流绑定不一致。** MUSA 默认流的 native handle 可以是空指针。
`bind_tvmffi_stream_to_current_torch_stream` 原来委托给通用 FFI helper，
在这种情况下改绑池化流，但直接调用方没有安装跨流等待。现在直接绑定真实当前流，
包括有效的空句柄；H8 DSA 的 current-stream 包装也调用这一接口。通用
`TvmffiStreamGuard` 的池化流策略不变。这两处修正单独不能解决完整 eager 精度。

**HC decode 的等待提交太晚。** `try_mate_hc_pre_decode` 中，FFI prenorm 在
池化流写 partials，native finish 在 Torch 流读取；原 guard 在 finish 启动后
才析构并提交 FFI→Torch 依赖。现在把 guard 限定在 FFI 调用范围内，让依赖先于
native finish 入队。Graph/非默认流和 prefill 的全 FFI 分支保持原有顺序。
这一步恢复了短请求正确性，但长输入跨页仍失败。

**换页时读取了尚未更新的 recurrent state。** Compressor/indexer 的状态附着于
SWA 物理块。新页刚分配时，旧状态在前一个 token 的页内；原 state-gather 总是
读取最新有效块，导致读取新页的零值/旧值。现在由 live input position 确定读页，
仍输出当前页的写 slots：读取上一时刻状态、计算更新、写入当前页。这样不需要
关闭 C4/C128 更新或冻结状态。

### 修改范围

| 文件 | 改动 |
| --- | --- |
| `xllm/core/kernels/musa/musa_tvmffi_stream.{h,cpp}` | 保留精确当前流绑定，包括默认流空句柄 |
| `xllm/core/kernels/musa/deepseek_v4_dsa_attention.cu` | H8 current-stream 包装使用上述接口 |
| `xllm/core/kernels/musa/deepseek_v4_hc_pre.cu` | 缩短 FFI guard 生命周期，先等待再启动 native finish |
| `xllm/core/kernels/musa/deepseek_v4_sp_pad.cu` | state gather 分离源页和目标写页 |
| `xllm/core/kernels/musa/musa_ops_api.h` | 私有 state-gather 接口增加可选 live positions |
| `xllm/core/layers/musa/deepseek_v4/deepseek_v4_attention.cpp` | 四个 q1 compressor/indexer 调用传入 live positions |
| `tests/core/kernels/musa/` | 默认流、非默认流、capture 流、HC 依赖及 C4/C128 跨页回归 |
| `tests/core/kernels/CMakeLists.txt` | 在已有 device 分发中追加 MUSA 测试目录 |

没有修改公共 sampler、sequence、worker service、公共模型算法或共享参数结构。
没有增加 MUSA 模型行为环境变量。当前 q1 修复不能自动证明多 batch 或任意压缩
状态恢复场景正确；其他设备/模型也没有用本次测试替代各自的运行回归。

### 实验链及失败记录

统一模型：`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`。
Target-only 配置：TP8/EP1/DP1/CP1、B1、Graph=false、无 draft、无 prefix、
无 schedule overlap、block size 128、max batch tokens 10240。

| 实验 | 结果 |
| --- | --- |
| `target_eager_baseline_01` | 原二进制两条 64/64 均失败，只输出数字 1 |
| `target_eager_direct_stream_02` | 只修正直接流 helper，仍失败；候选随后回退 |
| `target_eager_dsa_stream_03` | 增加 H8 包装修正，仍失败 |
| `target_eager_hc_handoff_04` | 64/64 严格 1..32、128/128 严格 1..64、64/256 严格 1..128 通过；2500 长输入仍失败 |
| 同上，2500/1500 | 只输出 1..32 后继续生成不可见 token，完整精度失败 |
| 同上，2558/160 | 只输出 1..3，失败接近总位置 2560 |
| `target_eager_previous_state_05` | 2558/160 严格 1..80 通过；两条连续 2500/1500 严格 1..750 通过 |

两条完整通过请求使用二进制 SHA256：

```text
ae41e9348d436c03b44babf36ea9f0ee3ec4f5d0706149dd1eeba2bf2f261b67
```

客户端诊断 TTFT/TPOT 分别为 `1240.065/61.509 ms`、
`1270.300/61.234 ms`。它们是 **target-only eager**，不是 DSpark Graph
性能；运行期间有 CPU 增量编译，不能作为干净性能比较窗口。

特别注意：失败请求可能只在开头产生可见内容，随后用 special/EOS token 耗尽预算。
例如失败的 2500/1500 请求 E2E 92.858 秒，但 last-content 约 5.003 秒；
其客户端 TPOT 2.381 ms 无效，绝不能算提速。精度不通过的请求不进入性能结论。

格式化后完整构建成功的二进制 SHA256：

```text
0e1893c23836f1b68f2192caa6e4b114461a965933978a246768d47208ef3bea
```

独立复验目录为 `target_eager_formatted_fresh_06`：

| 请求 | 严格结果 | 客户端 TTFT / TPOT |
| --- | --- | --- |
| 2500/1500，服务启动后的首条请求 | 750/750 PASS | 9386.441 / 61.330 ms |
| 8192/1024 | 512/512 PASS | 3736.439 / 71.457 ms |

两条均精确核对服务端 input/output usage，未启用 Graph、MTP/DSpark、prefix
或 overlap。上述为 target-only eager 正确性回归，首请求包含冷态加载开销，不能与
DSpark 的热态性能基线直接比较。

同一独立服务另外完成非计数 smoke：`17 × 19` 返回 `323`，指定只输出
`sapphire` 时返回完全一致的单词。它们是补充功能检查，不代表通用评测集精度对齐。

### 定向测试

四个保留测试均通过，无跳过：

- 延迟 FFI prenorm producer 后，HC output/post/comb 与非默认流参考一致。
- C4/C128 状态行数、int32/int64 table 和 position 下，位置 128 从旧页向
  新页写，位置 129 从新页读取；复用同一 metadata tensor。

直接流绑定的三个单测曾通过，但该候选导致 target Graph 长请求退化，已在
`b8dc72ba` 中回退；因此不把它们作为当前产品修复的一部分。

测试 XML：`state_gather_test.xml`；构建脚本：`build_stream_test.sh`。
测试已接入正常 `cc_test` 清单；本轮实际执行使用同一产品实现与已编译 native
object 的独立 runner，没有宣称完整跨设备测试套件通过。

测试环境设置过程中的失败也保留：缺少某个 TorchMUSA stream-guard 别名、
as-needed 丢弃 torch_python、未注册 PrivateUse1 hooks、GPU scalar 比较需要
pinned allocator、将 LD_PRELOAD 误施加给 timeout。最终使用现有 StreamGuard、
安装环境的 backend hooks，并在 CPU 比较 GPU 结果。没有替换模型算法或 mock kernel。

## 二、Draft 全链路 Graph

当前正式性能基线仍来自原文档的 **target q6 Graph + draft eager**：
独立服务六请求热窗均值 TTFT `215.821167 ms`、TPOT `8.188667 ms`，6/6 严格通过。
本轮没有把上述 eager 修复的延迟数字混入该基线。

`probe_graph_ops.py` 已分别验证安装环境 TorchMUSA 2.9.1 的 embedding、
index-select-out、HC repeat、五步 Markov greedy 和 masked attention：五个 case
均能 capture，改变 live inputs 后三次 replay 对照通过。它们只是算子级可行性
证据，尚不是完整 C++ draft/FFI/custom-allreduce 闭环。

后续需要覆盖 draft forward、LM head、Markov 和 accepted-context KV 写入，处理
动态位置、非因果 block mask、workspace 生命周期及接受/拒绝路径。原 target compact
Graph 的 compressor 历史冻结行为依旧属于已知边界，不能用计数通过代替一般任务等价性。

## 三、复现入口

原 DSpark 性能链路及环境依赖见
[完整复现文档](DEEPSEEK_V4_MUSA_DSPARK_REPRO.md)。本轮 target-only eager 的 NFS
实验启动器是 `dspark_goal_20260911/run_owned.sh`，它选择本分支 build 下的 xllm，
并保存源码状态、patch、二进制指纹、启动参数和宿主机快照。

使用前确认宿主机 GPU 空闲并只停止自己拥有的占卡程序；使用全新的实验名：

```bash
setsid -f bash \
  /data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/dspark_goal_20260911/run_owned.sh \
  target-eager my_eager_repro
```

该实验脚本的宿主机 SSH control socket 属于本轮环境，其他会话需自行建立授权
连接或调整脚本，不能当作可跨机器复制的凭据。服务健康后执行：

```bash
python3 /data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/dspark_20260910/benchmark_0731.py \
  --url http://127.0.0.1:18860/v1/chat/completions \
  --prompt-tokens 2500 --output-tokens 1500 --count-to 750 \
  --output /data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/dspark_goal_20260911/my_eager_repro/full_01.json
```

必须检查 `accuracy.strict_text_passed`、精确 usage，以及完整输出与结束时间；不能只
检查 HTTP 成功或构建成功。服务停止后核对宿主机 GPU 释放，再恢复自己拥有的占卡程序。
