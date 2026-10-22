<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->
# MUSA DSpark 1P1D：30K/1024 B1 与 B16 验收结果

2026-10-06。正式同版本配对通过：B1 平均 TPOT <9ms，16并发输出吞吐 >B1×4.5。
16:39 最终快照/input哈希校验通过，P/D各8个PID与验收binary匹配，模型接口正常；目标验收完成。
完整过程见 [实验记录](DSV4_PD_V128_PORT_AND_30K_GOAL_20261005.md)。
拓扑、统计口径和具体代码改动见 [1P1D性能与优化说明](DSV4_PD_30K_1P1D_OPTIMIZATION_EXPLAIN_20261006.md)。

## 结果与边界

| 正式场景 | 严格正确 | TTFT均值 ms | TPOT均值 ms | 实际输出 tok/s | 完整窗口 s |
|---|---:|---:|---:|---:|---:|
| B1 | 64/64 | 2196.170133 | **8.779710** | **91.535690** | 715.961170 |
| B16并发 | 64/64 | 15129.773655 | 19.853986 | **428.193054** | 153.052460 |

- 两侧各输出65536 token；比值 **4.677880851765037**，高于4.5。
- B1正式64条的TPOT范围为8.504149～8.973826ms；平均值及本轮每条正式请求均低于9ms。
- 两组各16条预热也全部正确，不混入正式窗口。实际prompt/completion分别为30000/1024，cached全部为0。
- 每题一次，无重试、错误或usage差异；校验完整512个目标整数，不只检查HTTP成功或首尾数字。
- B16是客户端并发上限16，不保证每一步物理decode batch恒16。硬件为8P+8D，共16卡。
- B16单请求TPOT并非<10ms，本目标是聚合吞吐。此次不是GSM8K或自然语言长上下文全量验收。

对照上一合格5bd版本：B16输出405.917509→428.193054tok/s（约+5.49%），B1输出90.752892→91.535690tok/s；没有降低B1分母。

## 采用的改动

原Indexer分块循环重复创建valid长度、因果score mask及无效index mask。
新路径在**每次helper调用内**一次创建，分块通过narrow view读取，减少小算子启动和临时tensor创建。
不跨层或跨请求缓存；不改变GEMM形状、FP32评分算术、sorted TopK、有效KV范围或原状态保护。

开关：`XLLM_DSV4_PREFILL_INDEXER_MASK_ONCE=1`。默认关闭，限定长Prefill、query≤2048、非Graph capture。
真实模型TopK为512；已分别完成TopK2048补充覆盖及实际TopK512单测。模型日志验证9MiB mask实际命中。
单位测试先进行CPU/GPU逐位比较，再进行同binary单开关模型回归；并未仅凭微测速度采用候选。

同caf二进制开关两组各24/24正确；开启配置还通过64条连续请求、2条后置B1、5条Prefix重复/换题，共95/95。
Prefix实际命中序列为0/19456/19456/0/19456。此前无收益的因果评分裁剪、MCCL通道调整未加入最终配置，失败和负收益记录均保留。

## 当前部署及工件

根目录：`/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005`

| 项目 | 配置/位置 |
|---|---|
| Prefill | 10.121.38.94 / wjm_xllm，8卡，`PREFILL_indexer_mask_once_pair_b16_r1` |
| Decode | 10.121.38.90 / wjm_xllm，8卡，`DECODE_reopen_r1` |
| Control | 10.121.38.92 / cmz，CPU服务 |
| 模型 | `/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt` |
| 并行/推测 | 每侧TP8/EP1/DP1/CP1，DSpark N6 / target q7 |
| Prefill | 16K chunk/budget，容量16，memory0.81，Graph true，Overlap false，Prefix true |
| Decode | Graph/Overlap true，容量16，原已验证配置不变 |
| 请求入口 | `http://10.121.38.92:29888/v1/chat/completions` |
| P binary | `candidates/indexer_mask_once/xllm` |
| D binary | `candidates/reopen_decode/xllm` |
| P核心启动脚本 | `pd-role-indexer-mask-once.sh` |
| 独立编译目录 | 28 / xllm_musa_build_0914：`/tmp/pd_indexer_mask_once_model_20261006` |
| source包 | `pd_indexer_mask_once_sources.tar.gz`，含CPP/header、编译/链接命令和依赖hash |
| 原始输入 | `logs/dspark_30k_1k_cases_20260926/cases_pd_model_id.jsonl` |

P SHA256：`caf3db8f775224045f9b8143f1b8a3491fb4637229fd5ab18194e00b60506b43`

D SHA256：`5ea86017debbd94ba751a8ce38c74c1e2e9d39487b4ef3fcf662a24cedbaac93`

只重编Attention TU，复用已有编译归档、Torch/MUSA、AOT和通信依赖；其余layers成员hash核验不变。
没有提交/推送Git分支，运行的是上述独立可追溯产物。Windows源码/文档不直接参与Linux推理。

## 证据与复核

- 冻结计划与输入：`indexer_mask_once_pair_20261006/plan.json`、`inputs.jsonl`、`manifest.sha256`。
- 正式审计：`indexer_mask_once_pair_20261006/paired_audit.json`，`evidence_valid=true`、`paired_performance_pass=true`。
- 正式原始数据：`logs/indexer_mask_once_pair_{b1,b16}_{warmup,formal}/` 内的raw、attempts、config和summary。
- 正式快照：`evidence_indexer_mask_once_pair_20261006`；前置回归快照：`evidence_indexer_mask_once_round_20261006`。
- 单测：`indexer_mask_once_unit_r1/run_96_r1`、`indexer_mask_once_k512_unit_r1/run_96_r1`。
- 最终检查：`final_pair_snapshot_verify_20261006.log`、`final_pair_inputs_verify_20261006.log`、`final_runtime_{PREFILL,DECODE}_20261006.log`及各自SHA。
- 正式运行脚本：`run_indexer_mask_once_pair_arm.sh`；审计源码：`audit_indexer_mask_once_pair.py`。

只读复核（在共享目录可见的Linux容器里）：

```bash
R=/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005
cat "$R/indexer_mask_once_pair_20261006/paired_audit.json"
sha256sum -c "$R/evidence_indexer_mask_once_pair_20261006/snapshot.sha256"
sha256sum -c "$R/evidence_indexer_mask_once_pair_20261006/inputs.sha256"
```

原实验脚本禁止覆盖已存在的输出目录，不能直接重跑并覆盖这次结果。
全程未reset GPU；仅精确停止本实验匹配exe/PID的Prefill，D与Control在本次配对中没有重启。
完成后保留90/94服务运行。以原始工件和hash为准，不以当前仓库HEAD代替运行版本。
