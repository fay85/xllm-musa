# E43：固定 C1 owner 与 DeepEP LL 接入

这轮将 E42 节省显存的 FP8 非专家投影作为前置，目标是减少每个 Decoder 层的 TP Attention 与专家输出全归约，借鉴 SGLang B 的完整 Attention＋DeepEP low-latency 路径。E42 单独未提速，仍只作为前置保留，累计最佳是 E39。

## 实际计算分工

- D 端 rank 0 执行目标模型完整 H64 Attention、路由、共享专家和 mHC；所有 rank 都保留各自的 48 个路由专家，共 384 个。
- owner 通过 DeepEP dispatch 发出 token；其余 rank 用真实 0 行输入参加同一轮通信，仍执行分到本 rank 的专家计算，再用 combine 回传。
- combine 已完成路由权重及专家结果合并，不能重复 EP 求和；owner 只加一次完整共享专家。
- 每轮模型边界广播最终 hidden 与 aux hidden；embedding、Engram、目标 LM head 和 draft 仍沿用 TP8。这是固定并发 1 的 owner 布局，不能称为完整的通用 DP8 调度。
- 16 行 dispatch 容量是通信切块大小；超过容量的输入按块参加相同次数通信。不是减少实际请求 token。

## 代码层次

| 文件 | 实际改动 |
|---|---|
| `deep_ep_ll_runtime.h` | 原生进程初始化 vendor 通信接口、独立 Gloo rendezvous、LL buffer 生命周期与 Graph 边界 reset/barrier |
| `moe.h` | owner dispatch/专家 W4A8/combine，非 owner 0 行参与，避免重复归约 |
| `decoder.h` | 给非 owner 提供只执行专家通信的入口 |
| `text_model.h` | owner 执行目标模型，其他 rank 参加每层专家通信，保持层数和通信次数一致 |
| `deepseek_v41_native.h` | 显式默认关闭开关、TP8/EP8/权重/布局一致性检查，连接 Runtime、目标图及模型边界广播 |
| `decode_graph.h` | 捕获前后调用通信 reset/barrier；回放中没有 CPU barrier |

这里仅借助嵌入式 Python 对接已有 DeepEP vendor ABI，模型 forward、MATE 算子、专家数学和 Graph 仍在原生 C++ 路径执行；没有重新退回 Python 模型 forward。

## 已完成的直接检查

8 rank、40 次连续层通信，owner 行数 1/4/6，改变输入和路由；包含 eager 及 Graph 回放，共 192 项通过。E38 已完成真实专家权重与 LL 组合检查，本轮不重复其全部测试。H64/H8 Sparse 组合 8 场景，H64 12 形状及 4 次 Graph 回放通过。

原生启动曾把 MUSA tensor 标记为 CUDA dispatch key，导致 fill 失败；保持硬件、驱动不变，按已有 unified runner 方式优先预加载 libmusa_python.so，问题解决。两个失败实验自行退出，未强杀。生产构造函数加入 dispatch key 检查，防止后续错误环境进入通信。

## 完整模型 eager 冒烟

两道基础题答出 4.2 和 56；1 暖机＋2 条短请求全部实际 256/256、success。2 条 screen 均值 TPOT 36.331158 ms，TTFT 343.033963 ms，端到端输出 26.645907 token/s。不是正式性能对比，不替换 E39。

汇总脚本因只有 formal 阶段赋值 summary，在 smoke 结束后报 UnboundLocalError。原始请求成功且数据库完整，原始 exit1/error 保留，只离线补写 reconciled_summary 和来源更正，没有重复请求或改写原始数据。

D 端观测最大 77334 MiB/卡；这是冒烟后的驻留观测，不是连续采样的峰值。16 个模型进程全部正常 exit0，08:45:49 双机 GPU 已释放，无 SIGKILL。

Graph 同 ELF 候选已于 08:46 启动，结果待真实完成。显存预算与 eager 通过均不能代替 Graph 验收。

**仅通过基础功能检查，未完成模型精度验收。**
