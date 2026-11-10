---
title: "Engram 主机侧卸载（Host Offload）"
sidebar:
  order: 83
---

## MLU FP8 Engram

MLU 原生 DeepSeek-V4.1 默认将 Engram FP8 表及 scale 保存在 host 内存。未传参数或传入 `{}` 时均启用；可通过服务启动参数显式选择：

```bash
# 默认：表在 host，保留原有 TP/EP 行切分
xllm --engram_config='{"placement":"cpu","replication":"none"}'

# 同机模型 ranks 跨 DP/TP 共享一份完整 FP8 表及 scale
xllm --engram_config='{"placement":"cpu","replication":"per_node"}'

# 关闭卸载：表常驻 MLU device
xllm --engram_config='{"placement":"gpu"}'
```

host 路径复用已有 hash 计算：CPU 收集当前 token 命中的 FP8 行及 scale，经 pinned staging buffer 搬到 MLU，沿用 device 反量化、集合通信和投影。Graph replay 使用流上的 host callback 更新命中行，避免复用旧 token 的查表结果。大表本体无需锁页，仅 staging buffer 锁页。

`replication=per_node` 默认关闭，开启时要求 `placement=cpu`。加载时利用模型的 world 进程组收集 kernel boot ID，识别同机 ranks。每台主机的首个 rank 创建匿名 `memfd` 存储，其余同机 ranks 通过 `/proc/<pid>/fd/<fd>` 映射同一份存储。同一模型实例的同机 ranks 跨 attention DP/TP 组共享一张完整表，不同主机各保留一份完整副本。同机 ranks 必须处于同一 PID namespace，且有权限访问 leader 的文件描述符。使用不同 world 组的独立服务实例之间不共享表。

加载是模型 world 组上的集合操作。各 rank 仅复制按同机 rank 数分配给自己的 checkpoint 行，初始化等待所有 ranks 完成后才允许查表。单机 attention DP2 × TP4 时，8 个 ranks 分工填充同一份完整表。最后一个映射及文件描述符释放后，存储自动回收；无需命名共享文件或持久化转换缓存。Engram 层的 eager 和 graph 回归覆盖 DP2 × TP4；原生 MLU V4.1 整模型仍要求 DP1，模型及运行时的完整 DP 接入需后续完成。

MLU 原生配置只接受与 SGLang 同名的字符串选项：`placement`（`cpu` 或 `gpu`）和 `replication`（`none` 或 `per_node`）。默认值为 `placement=cpu`、`replication=none`；`placement=gpu` 对应 MLU device 存储。

共享表保留原始 FP8 E4M3 权重字节及 E8M0 scale 字节，不提前将整表展开成 FP16/BF16。各 rank 独立查询本轮全部命中行，仅将这些行及 scale 经自己的 pinned staging buffer 搬到 MLU 反量化；此模式跳过 Engram 查表结果的 AllReduce。Graph replay 复用同一份共享表和已有 host callback。

默认 host 分片模式下，双表在单机 TP=8 时约占每 rank 23.60 GiB，节点内合计约 188.83 GiB。机内共享模式下，每台主机保留一份完整物理表，共享范围不随 attention DP/TP 切分而缩小。各进程都映射完整共享表，直接累加进程 RSS 会重复统计共享页，应看 PSS 或节点物理内存；加载及运行缓冲另计。投影等小权重仍在 MLU。以下章节介绍 NPU 路径，其默认值和支持的表格式与 MLU 不同。

## NPU 概述

DeepSeek-V4.1 的 Engram n-gram 表在 W8A8（Eco-Tech int8）导出下每卡约 **27.7 GiB** 的常驻显存（可被卸载释放）。`engram_config.cpu_offload` 把这部分表搬到 **host 内存**（pageable CPU 内存），换取 HBM 预算，代价是每步一次 NPU→CPU 的 id 同步 + CPU int8 去量化 + 命中行的 H2D 回传。该路径由融合 CPU 查表算子（`xllm_ops.engram_int8_lookup_cpu`）承担，仅 int8 表可卸载。

DeepSeek-V4.1-Flash-1017 W8A8 的 INT8 Engram 表使用 UE8M0（safetensors `F8_E8M0`）group-32 scale，而较早的导出使用 FP32 scale。两者均支持：加载时只对当前 TP rank 的行分块解码 UE8M0，运行期仍使用 FP32 scale，因此 CPU 卸载与 DP 共享的配置、内存预算和查表算子合同不变。无需离线重写权重；直接将原始字节转成浮点数不是正确的 UE8M0 解码。

## 配置方式（`--engram-config`）

配置以 **原始 JSON 字符串** 经 C++ gflag `engram_config` 透传到 Python 执行器解析（`xllm/python/models/engram_config.py`），命令行参数写作 `--engram-config`：

```bash
xllm --engram-config '{"cpu_offload":true,"dp_shared_memory":false}'
```

NPU 路径**默认全关**：不传 `--engram-config` 时 `EngramConfig()` 为 `{"cpu_offload":false,"dp_shared_memory":false}`（等价 `{}`），走 HBM 路径；要开启卸载必须**显式传入**该参数（如 `--engram-config '{"cpu_offload":true}'`）。部署侧的启动脚本若需要默认开启，由该脚本自行下发参数，xLLM 不感知其形态。

## 开关语义

| 键 | 默认 | 含义 |
|---|---|---|
| `cpu_offload` | `false` | int8 Engram 表驻留 host 内存（卸载开关）。 |
| `dp_shared_memory` | `false` | 节点内 DP 副本共享 **同一份** host 表（省 host 内存）。**lookup 仍每个 rank 各执行一次**。 |

`dp_shared_memory` 共享的是**只读的 TP 分片表数据**（host 上的 int8 codes 与 fp32 scales），不是查找结果：各 rank 仍按**自身的输入与历史**生成 n-gram hash id，并各自完成 lookup + 去量化 + H2D，回传的行（staging 输出）也不共享。之所以不共享查找结果，是因为各 rank 的 hash id 本就不同，要共享就得先把 id 汇总到一处再广播回来，多出一轮通信与同步，因此本设计让每个 rank 直接查本地那份共享表。

解析是**严格**的：字段值必须是真正的布尔值，`"true"` / `0` / `1` 一律拒绝；未知字段、重复 JSON key、非法 JSON 都会报错。

### 合法配置（三选一）

```jsonc
{}                                              // 全关：走 HBM 路径（与旧默认不同，旧默认开）
{"cpu_offload": true}                           // 卸载到 host 内存
{"cpu_offload": true, "dp_shared_memory": true} // host 卸载 + 节点内 DP 共享 host 表
```

### 非法组合

```jsonc
{"dp_shared_memory": true}                       // 错误：shared 必须配 cpu_offload
{"cpu_offload": false, "dp_shared_memory": true} // 同上
```

`dp_shared_memory=true` 而 `cpu_offload=false` 会在解析期直接 `ValueError`。

## 约束

- **TP**：仅支持 `TP ∈ {1, 2, 4, 8}`，且 **hash 列数必须被 TP 整除**；hash 列数 = `(max_ngram_size - 1) * n_heads`（V4.1 默认 `n_heads=8`、`max_ngram_size=4` ⇒ **24 列**，TP 取 1/2/4/8 均整除）。非整除去量化/切分会直接拒绝。
- **PP / DCP**：engram 目前要求 `PP=1`、`DCP=1`。
- **权重格式**：仅支持 safetensors / dummy（`load_format ∈ {auto, safetensors, dummy}`）。**int8 专属**：只有 W8A8（int8）表有 CPU 解码路径；fp8/bf16 表请求 `cpu_offload` 会回退到 HBM，请求 `dp_shared_memory` 则直接报错。
- **DP 共享依赖**：`dp_shared_memory` 需要节点内 gloo 通信组（HCCL 无 object 集合通信）与 `/dev/shm` 空间（启动前按聚合 footprint 预检）。
- **PD 分离**：TODO —— 待 V4.1 Engram 接入 P/D 分离拓扑后，该拓扑将要求 `dp_shared_memory=true`；当前运行期尚未强制。

### 节点身份与共享内存前提

同一 DP 共享内存域内的各 rank 应报告一致的节点身份。`hostname` 仅用于预检分组（如 DP 共享的 `/dev/shm` 容量估算），**它本身不是共享成功的证明**：不同物理节点必须可区分，而同一物理机但处于不同 `/dev/shm` namespace（如各容器挂载独立 tmpfs）也不能共享。真正的证明是 attach 成功**且**从段尾读回的 token 与 leader 写入的一致（`_verify_segment_token`）；`/dev/shm` 容量检查同样只是预检。这里**不要求每个 rank 的 `hostname` 唯一**——同一节点上多 rank 重名是正常的。

## 相关实现

- 解析/校验：`xllm/python/models/engram_config.py`、`xllm/python/models/deepseek_v41.py`（`_validate_engram_config`）
- 卸载路径：`xllm/python/models/deepseek_v41_engram.py`
- DP 共享 host 表：`xllm/python/models/engram_shared_memory.py`
