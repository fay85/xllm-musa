# DeepSeek-V4 MUSA 重复/死代码清理与验证（2026-10-29）

## 基线和范围

- Git 基线：`feihu-1016-dev@f8dd0663279e9085365ade309d9823b7b438c4e6`。
- 本轮基于已发布的新镜像版本，不导入旧 `refine_merge_20261026/source` 的未提交修改。
- 本轮是保守等价重构，不修改模型、Graph key、speculative 参数、通信算法、算子 ABI。
- 主 MUSA 源码和 CMake 选择的 Decode overlay 同步清理。运行中的源码和原二进制未覆盖。

## 已修改

1. 删除主 Attention 与 Decode overlay 的 FP8 activation roundtrip 空函数及全部调用；保留 BF16 KV 的行为。
2. 删除空 PrefillProfileState、begin/record/finish profile、空 output-projection sync，以及确定恒 false 的 trace/config 空壳和不可达日志。
3. 主模型使用现有 `deepseek_v4_build_cache_specs`，删除构造函数内的重复实现；保留 MTP 调用。
4. 主模型复用 `load_hc_head_state_dict`，保留 DSpark 子类的调用。
5. 两套 metadata collector 共用非 cache 字段枚举，保留 Graph 专有的 cache-group alias 处理和原访问顺序。
6. 删除 MUSA 专用模型头中提前 return 后不会执行的 CUDA sparse/QLI metadata 构建。
7. 提取 compressor/indexer 共用的 C4/C128 tail、prefix snapshot 和 Decode 状态初始化；C4 的重叠窗口、C128 的部分块及匹配 shape 的 Graph buffer 地址保持不变。
8. 合并 MoE FP4/FP8 expert loader 骨架，按格式选择原有存储、weight/scale 校验和分配函数；保持 w1/w3/w2 的分片与发布顺序。
9. 简化 PCP split 中不可达的短请求空分支，保持现有合法输入的 split 结果。
10. 按仓库要求对所有修改的 C++ 文件运行 clang-format 20.1.6。部分 MoE 文件含原有格式差异，因此 diff 行数不等于实际重构规模。

## 明确保留的路径

这不是将全部相似代码强行合并：

- compressor/indexer 的 projection、pool、RoPE/Hadamard 和 persistent gather dispatch 仍有专有分支；
- 单请求/多请求 Indexer 参数并不完全相同，保留原 dispatch；
- Attention/MoE allreduce 的 layer、workspace、同步保护条件不同，保留原策略；
- legacy Graph weak-symbol 开关和 rank7 dump 属于可达诊断入口，不按死代码删除；
- `XLLM_DSV4_STABLE_MULTI_GRAPH_MAX_SEQS` 在本基线源码无读取。候选启动移除此无效环境变量，不改原计划；恢复原服务时仍使用原计划。

## 构建证据

独立目录：

```text
/data/nfs_shared/jd/dsv4_cleanup_20261029
  base.tar
  changed.txt
  source/
  build_cleanup.py
  build_r3.log
  prefill_r3/{commands.json,link.json,*.compile.log,link.log,objects/,archives/}
  decode_r3/{commands.json,link.json,*.compile.log,link.log,objects/,archives/}
  bin/{prefill,decode}/{xllm,libasio.so}
  prefill.{ldd,help}.log
  decode.{ldd,help}.log
```

编译环境：90 / `xllm_musa_mt_main`，沿用新镜像 MATE 0.2.7、Torch-MUSA 2.9 及 r14 AOT。

复用旧 P/D 编译命令、未改动对象和依赖，重编改动 translation units 与 Ninja 依赖记录命中的模型头消费者。修改后的对象只写独立目录，相关静态库复制后替换成员，再链接独立二进制；没有覆盖原 build cache。

初次脚本因消费者名称检查退出；下一次发现新旧 include 混用后修正路径。有效结果以 `build_r3.log`、实际对象、链接日志及运行进程为准，失败日志保留。

| 产物 | SHA256 |
|---|---|
| Prefill | `cc94e7c56490003b14bfe9c5e3492d9415d9a7c0ef46c4987bf3492fa98f8aa9` |
| Decode | `faa5b89f5e2f8064dc9ede7f7aae7193cab41733826e27aaf913229dfedf3e70` |

P/D 均完成编译、链接、`ldd -r` 和 `--help` 检查。

## 单元测试

新增 `deepseek_v4_compressor_state_test`，在 `tests/core/layers/CMakeLists.txt` 注册 CPU_ONLY target。

本次直接复用镜像 Torch 和 vcpkg GTest 编译运行同一测试源码：

```bash
docker exec xllm_musa_mt_main bash /data/nfs_shared/jd/dsv4_cleanup_20261029/unit_test.sh
```

5/5 通过：24 组 C4/C128 边界组合、匹配 buffer 地址与数值复用、空状态初始化、单边 shape 不匹配、snapshot 与输入不共用存储。

## 端到端 A/B

- 拓扑：P94 + P96（PCP CP2），D90，各角色沿用 r14 参数；Control 未重启。
- 模型：`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`。
- 相同冻结输入：`inputs/payloads.jsonl`；相同 benchmark 和严格检查，不以 HTTP 200 代替验收。
- 20K/100、B1/B2 各两轮、每轮 32 个正式请求，另有 4 个 primer/warm 请求。正式请求必须 cached_tokens=16000、输出 token 数及正文完全匹配。
- GSM8K 使用 r14 的前 20 个固定 HTTP payload（单轮 boxed、temperature=0、max_tokens=1024），比较原回答正文、boxed 结果和 stop 状态；不是本次全量 GSM8K 验收。
- 原始记录：`smoke_baseline/`、`smoke_candidate/`；角色日志和 PID/start-ticks/exe 清单：`runtime/`。

| 检查 | 清理前 r14 | 清理候选 |
|---|---:|---:|
| 20K/100 严格门禁（含预热） | 132/132 | 132/132 |
| B1 正式请求 | 64/64 | 64/64 |
| B2 正式请求 | 64/64 | 64/64 |
| 正式请求 cached_tokens | 全部 16000 | 全部 16000 |
| GSM8K 正常 stop / 原答案 boxed 一致 | 20/20 | 20/20 |
| GSM8K 正文与原 r14 逐字一致 | 20/20 | 20/20 |

候选与本轮 baseline 的 20 条 GSM8K 正文也全部逐字一致。测试窗口内未发现 hang、OOM 或异常退出；未执行全量 1319 题、长稳、FP4 模型验收。

正式请求均值（不是累计吞吐倒数）：

| 并发 | 清理前 TTFT ms | 候选 TTFT ms | 清理前 TPOT ms | 候选 TPOT ms | 清理前输出 tok/s | 候选输出 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 529.851 | 539.034 | 9.470 | 9.513 | 68.084 | 67.457 |
| B2 | 525.820 | 531.417 | 11.348 | 11.623 | 119.490 | 117.182 |

B1 TTFT +1.73%、TPOT +0.46%；B2 TTFT +1.06%、TPOT +2.43%。本轮没有性能收益，存在小幅变慢；单组先后 A/B 不足以将差异归因于重构，也不证明严格性能等价。本次按有限冒烟范围接受正确性清理，不宣称性能优化。

输入文件 SHA256：
`9b240146f7f310402a53f90313d0a8df8f58772895954c2df25d7b5546468de9`。

汇总：`/data/nfs_shared/jd/dsv4_cleanup_20261029/results.json`。
验证脚本和依赖复用命令保存在同目录的 `build_cleanup.py`、`unit_test.sh`、`smoke.py`、`runtime.py`。脚本创建独立目录，不能对已存在的运行目录盲目重跑。

## 运维边界

只按本项目的 PID、start ticks、exe 清单停止并切换 P/D；保留原日志和 maps。没有 reset GPU，没有停止其他容器或 Control。
验收结束后恢复原 r14 服务，清理候选作为独立产物保留。

恢复检查：三个角色均重新就绪，Decode 已重新连接两个 Prefill；固定 GSM8K 第 0 题 HTTP 推理正常 stop，正文与原结果一致（boxed 18）。证据为 `restored/` 和 `restored_probe.json`。
提交前核对 11 个源码/测试/配置文件：10 个与测试副本字节 SHA256 相同，PCP 文件只差 LF/CRLF；未将未验证的源码改动混入提交。
