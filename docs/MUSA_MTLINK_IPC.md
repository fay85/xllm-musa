<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# MUSA 同机 IPC / MTLink 传输

本分支把 Mooncake 的同机 transport 接到 MUSA IPC，并提供片段合并、Driver batch copy、可选 GPU copy kernel 和默认关闭的 PULL 队列、完成票据诊断。现有实验已经跑通同机 P/D 传输；端到端吞吐与 RDMA 的观测区间接近，传输阶段仍较慢，尚未证明稳定超过 RDMA。

## 适用范围与依赖

- P、D 必须在同一台主机，GPU 和驱动必须支持相应的 IPC / peer-access 能力。IPC 描述符带主机、进程实例、设备、allocation、generation 和范围检查，不能用于跨机传输。
- 已验证的平台为 MUSA SDK 5.1。Driver batch、可选 peer policy 及完成票据依赖该 SDK 的接口；没有旧 SDK 的构建或运行资格结论。
- 同机 KV 数据通过导出的 GPU allocation 和 IPC 映射复制，不需要 IB IP、RoCE 或 RDMA 数据通道。RPC、协调和 metadata 交换仍沿用现有控制面，服务之间仍须可达。启用 IPC 不是关闭整个系统的网络。
- 软件日志和字节校验可以证明执行了 IPC 路径；目前没有 MTLink Tx/Rx 硬件流量计数，不能据此断言每次复制的物理路由。
- xLLM、Mooncake 和调用方必须使用一致的 ABI、feature defines 和依赖。旧 Mooncake 静态库不能仅靠设置环境变量获得这些能力。

## 获取和准备源码

发布基线为 `feihu-1016-dev` 的 `9c84ea55b52f9564a2e2b9602224b77cfb744684`。Mooncake gitlink 保持为 `bb2c0bee9f7839789c947556341556b4c02aaeda`，没有引用无法从远端取得的本地子模块提交。

从仓库根目录执行：

```bash
git submodule update --init third_party/Mooncake
python3 scripts/mtlink_prepare_mooncake.py --apply
python3 scripts/mtlink_prepare_mooncake.py --check
```

补丁及逐文件前后 SHA256 位于 [`third_party/patches/mooncake_musa_ipc`](../third_party/patches/mooncake_musa_ipc/README.md)。脚本要求准确的 Mooncake 基线，重复应用时只校验；文件存在冲突时拒绝覆盖。可通过 `--mooncake-root /path/to/checkout` 验证独立的 pinned checkout。后续若强制重置或更新子模块，需重新执行 `--check`，必要时应用补丁。

这个 vendor 补丁覆盖 **Mooncake pinned commit 到完整发布状态**，因此也带上实验开始前已有的 MUSA 构建和兼容依赖：`common.cmake`、两个 store 源文件及 `device/ibgda/os.h`。它们不是本轮测得的新优化；只发布相对 dirty golden 的五个 MTLink 文件增量会遗漏这些依赖。原有 `.orig` 备份和其他无关工作树修改没有纳入补丁。

## 编译

先完成该分支常规 MUSA 构建环境和依赖准备，再向现有 CMake 配置增加以下参数；必须重编受这些宏影响的 Mooncake 对象并重新链接 xLLM：

```bash
cmake -S . -B build_musa \
  -DUSE_MUSA=ON -DUSE_CUDA=OFF \
  -DUSE_INTRA_NVLINK=ON
cmake --build build_musa --target xllm --parallel 4
```

`build_musa` 应替换为实际使用的构建目录。以上只列本功能所需选项，不能替代项目原有的 Torch/MUSA、模型算子和其他第三方依赖配置。`USE_INTRA_NVLINK` 默认关闭。

若要编译可选的 fused copy kernel，再添加：

```bash
-DUSE_MUSA_IPC_COPY_KERNEL=ON -DMOONCAKE_MUSA_ARCH=mp_31
```

kernel 选项默认关闭，要求同时启用 `USE_MUSA`、`USE_INTRA_NVLINK`；架构参数须与目标 GPU 匹配。它会把多个独立 copy span 转换为 64KiB tiles，使用对齐 vector copy，并处理非对齐和尾部字节。已有实验没有证明它比 Driver batch 带来稳定的端到端收益。

本次发布检查已完成：

- 公开 checkout 的 6 个改动 C++ 源文件重新编译通过，使用已有第三方及生成的协议头文件。
- [`tests/musa_mtlink`](../tests/musa_mtlink/README.md) 的 30/30 CPU 回归通过；使用实际生产 headers、ThreadPool/Folly，没有链接或启动 GPU 库。
- 干净 Mooncake pinned checkout 的补丁重建、重复应用、冲突拒绝通过；kernel ON/OFF 和非法组合检查通过，完整 `transfer_engine` 构建归档及 CMake consumer 最终 ELF 链接通过。该 ELF 未执行。

开发镜像裁掉的 vcpkg Debug 库会导致其 package config 在 Release 配置时也报错。Mooncake 检查使用隔离的依赖 metadata 和实际 Release 库，并补全现有 vcpkg/pybind11 include 与 MUSA 链接路径；没有修改原依赖或替换 CUDA/MUSA 类型、头文件。CPU 测试通过显式 Release prefix 选择同一套实际库。

逐源文件 hash 和本次检查边界见 [`mtlink_validation_20261109.json`](mtlink_validation_20261109.json)。本次没有重做完整 xLLM fresh build 或 GPU/模型测试；下述性能与字节验证仍来自历史隔离 overlay，不能视为本次公开 checkout 的新模型测试结果。

## 运行配置

所有参与本轮同机传输的 P/D 进程保持一致的 transport 设置。以下是已测试过的 Driver batch 候选配置：

```bash
export MC_INTRANODE_NVLINK=1
export MC_IPC_COALESCE=1
export MC_IPC_BATCH_COPY=1
export MC_IPC_BATCH_ENGINE=transfer
export MC_IPC_PEER_POLICY=default
export XLLM_MUSA_PD_COPY_QUEUE=0
export XLLM_MUSA_PD_RETIRE_TRACE=0
```

随后使用原有 P/D 启动命令、模型和路由配置启动服务。`MC_INTRANODE_NVLINK` 默认为关闭；开启后显式安装同机 transport，拒绝与 `MC_FORCE_TCP` / `MC_FORCE_MNNVL` 冲突或缺少编译支持的配置。xLLM 也拒绝将 IPC 与 network peer 混合提交。未开启 IPC 时保留原有 RDMA/TCP 选择。

`MC_IPC_BATCH_ENGINE=transfer` 使用 Driver batch；`runtime` 是另一条显式选择的 batch API 路径。`kernel` 必须具有上述 kernel 编译支持。batch 模式要求 `MC_IPC_BATCH_COPY=1` 和 `MC_IPC_COALESCE=1`，不会在执行失败后偷偷切换路径或重放请求。`MC_IPC_PEER_POLICY=unidir` 已做能力检查和实验，但没有稳定性能优势，未设为默认。

仅进行诊断时可设置：

```bash
export MC_IPC_VALIDATE_BYTES=1
export XLLM_MUSA_PD_RETIRE_TRACE=1
```

前者进行传输片段字节比对，后者增加 CPU epoch 和真实 GPU event 观测；额外校验、事件和日志的运行不能用于性能结论。常规性能测试应关闭诊断，并核对各服务实际二进制、环境变量和 transport 日志，而非只检查启动脚本。

## 已有验证与性能

以下结果来自 2026-11-09 同一台机器上的隔离实验产物。模型为 **Qwen3.8-27B-FP8**，4P/4D，全部 TP1；输入实际 20,486 tokens，其中 prefix cache 命中 16,384 tokens，含五张图，OSL 100、C=10、MTP K=2、P chunk=8192。它们不是无缓存的纯文本 20k workload，也不是 TP2 性能结果。

V9 同一二进制的固定五组筛选共 600 个正式请求；每组两轮，每轮 60 个请求，预热和诊断不计入指标：

| 传输 / PULL 队列 | QPS 两轮 | TPOT P90 两轮（ms） | 完整 PDRegion 均值两轮（ms） |
| --- | --- | --- | --- |
| RDMA / request | 3.142 / 3.116 | 15.274 / 15.527 | 49.368 / 49.371 |
| IPC / request | 3.090 / 3.038 | 16.087 / 15.954 | 59.047 / 60.383 |
| RDMA / copy | 3.076 / 3.187 | 15.080 / 15.923 | 49.523 / 53.517 |
| IPC / copy | 3.046 / 3.016 | 16.620 / 16.418 | 97.786 / 87.788 |
| RDMA / request 复测 | 3.031 / 3.200 | 15.215 / 15.229 | 47.838 / 47.301 |

PDRegion 是正式时间窗口内的完整传输统计，不是按请求 ID 精确关联的纯复制带宽。IPC 的 copy/enqueue/drain 部分约 40.5ms；copy 队列使 fence 等待从约 16–18ms 增至约 42ms。P 侧 TTFT 约 1.45–1.53s 主导端到端耗时，因此相近 QPS 不代表传输已经同样快。两轮筛选不足以证明统计显著性。

V8 同 bin 的 Driver / kernel / RDMA 对照中，kernel copy 部分约 40.23–40.26ms，Driver 约 40.50–40.51ms；kernel 完整 PDRegion 约 62.60–63.04ms，仍未超越 RDMA 的约 47–49ms。kernel QPS 为 3.059 / 3.059，没有稳定超过 RDMA 控制组。

正确性证据分层如下：

- IPC/kernel 微测覆盖 READ、WRITE、对齐、非对齐、tile 边界、越界拒绝和 generation 检查；六个 kernel case 的 24 次 launch、192 个 region 零字节差异。
- PULL queue helper 使用真实 ThreadPool/Folly 的 17 项 CPU 回归通过。单独启用 copy 队列未带来性能收益，所以 `XLLM_MUSA_PD_COPY_QUEUE` 仍默认关闭。
- V12b 实际模型 gate：13 个请求、39 个 transfer batch、20,623,589,376 bytes，逐片比对零差异；使用的历史实验 xLLM SHA256 为 `093a5562195ae4e86c065b1f2811ada3083afe73cadbb549a16b8b64d15745c7`。
- GPU 票据的真实 native 探针在 ordinary 和 NULL legacy-default stream 两种 fresh-process 模式通过 Pending→Ready、两路事件、byte/guard oracle、错误 context 拒绝及作用域恢复验证。NULL 读线程只临时借用票据记录时已有 context；不创建 context，也不覆盖错误的非空 context。
- 传输字节相同和事件正确不等于模型语义准确率。RDMA 控制组自身也存在文本 hash 变化；本轮没有新增完整 MMMU/GPQA gold-accuracy 结论，也没有据此豁免新路径的语义验证。

## 完成票据的边界

`XLLM_MUSA_PD_RETIRE_TRACE` 默认关闭。开启后，在外层执行者完整 step/MTP prelaunch 返回、结果槽 CV 等待之前记录 compute/prepare 两路 Driver event；PULL receiver 与外层 MTP 执行者通过显式 incarnation / 生命周期绑定关联。查询和清理在 CPU ledger 锁外进行，缺失、Unsupported、Pending、错误或不连续的票据都不能当成完成。

模型中实际观测到：某次 PULL 的 admission / enqueue ACK 已到 81，而完整提交与 GPU latest/frontier 仅到 79；事件 79 Ready，并不能覆盖已入队的 80/81。历史缓存 Ready 也不能算作本次查询成功。

因此本分支保留全部现有 GPU fence、最终 stream drain、future 和传输范围。`owner_closed=0`、`owner_generation_proof=0`；尚未证明物理 block / LINEAR slot 的旧 owner 已关闭、所有辅助写者完成或共享前缀未来读者已被排除。诊断票据目前不授权省略同步，也不实现 decode 前缀免传。
