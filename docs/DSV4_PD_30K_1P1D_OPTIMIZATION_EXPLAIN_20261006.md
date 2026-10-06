<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->
# DSpark 30K/1024：这是哪种1P1D性能，主要改了哪里？

本文解释2026-10-06已完成的验收结果。数值来自冻结的正式请求和审计记录，本次编写说明没有重新跑性能，也没有修改运行服务。

## 1. 确认：这是两台机器、共16卡的1P1D端到端性能

| 角色 | 机器/容器 | 用途 |
|---|---|---|
| Prefill（P） | `10.121.38.94 / wjm_xllm`，8张MUSA卡 | 处理输入，准备KV及DSpark所需状态 |
| Decode（D） | `10.121.38.90 / wjm_xllm`，8张MUSA卡 | 接收状态，继续生成输出 |
| Control | `10.121.38.92 / cmz`，CPU服务 | 请求路由、P/D协调和结果返回 |

客户端请求入口是 `http://10.121.38.92:29888/v1/chat/completions`。
P/D之间使用Mooncake PULL：Decode从Prefill拉取所需KV/状态。

因此，这不是单机8卡成绩，也不是绕过Prefill后只测Decode的成绩。
统计的是客户端看到的完整请求过程，包括窗口内的路由、Prefill、跨机交接、Decode和排队开销；这些阶段可能重叠，不能简单相加。

**B1和B16都使用同一套8P+8D硬件。** B1/B16表示客户端并发上限1/16，不是GPU数量，也不保证每一步实际decode batch都恰好等于16。

## 2. 正式结果及测试口径

每条请求为**30000输入token、1024输出token**。两组各16条预热、64条正式请求；模型、二进制和核心配置相同，仅客户端并发不同。
每组使用新的P实例，Prefix Cache虽然开启，但正式请求实际命中均为0。

| 场景 | 严格正确 | 平均TTFT | 平均TPOT | 端到端输出吞吐 |
|---|---:|---:|---:|---:|
| 1P1D / B1 | 64/64 | 2196.170 ms | **8.779710 ms** | **91.535690 tok/s** |
| 1P1D / B16并发 | 64/64 | 15129.774 ms | 19.853986 ms | **428.193054 tok/s** |

吞吐按实际完成的输出token除以完整测试窗口计算：

```text
B1：  65536 / 715.961169736秒 = 91.535690440 tok/s
B16： 65536 / 153.052459527秒 = 428.193053562 tok/s
倍率：428.193053562 / 91.535690440 = 4.677880852倍
```

两个目标均满足：B1平均TPOT<9ms；B16聚合输出吞吐≥B1的4.5倍。
B1正式64条的单条TPOT也全部低于9ms，范围为8.504149～8.973826ms。

需要区分：

- **4.67788倍是B16并发相对B1的吞吐倍率**，不是这次代码修改让旧版本加速了4.68倍。
- B16平均单请求TPOT为19.854ms，并非低于10ms；本次满足的是聚合吞吐目标。
- 本次严格精度指完整512个目标整数、原payload、usage及结束状态均正确，不代表重新完成了GSM8K全量或通用长上下文精度验收。
- 不能与旧2.5K/1.5K、热Prefix命中场景的约220ms TTFT直接比较；本次输入是30K，正式请求没有Prefix命中。

## 3. 最后突破4.5倍门槛，主要改的是Prefill Indexer

最后一轮是从已验证的`5bd45dcb`二进制派生到`caf3db8f`。
这一步**没有更换Decode二进制**，主要减少了Prefill中Indexer反复准备辅助数据的开销。

Indexer需要给每个query判断哪些历史位置可见，并屏蔽无效位置。这里有三类辅助数据：

| 数据 | 通俗解释 |
|---|---|
| valid长度 | 当前query实际能看到多少个压缩KV位置 |
| score mask | 把query不能看到的未来位置屏蔽掉，避免被TopK选中 |
| index mask | 可用位置不足时，把TopK结果中无效的尾部索引标成`-1` |

长输入需要分成多个小块计算Indexer。旧实现每处理一块，都会重新构造该块的valid长度和两类mask。
这些都是整数/布尔运算，但长序列、多个layer反复执行，会累积小算子启动和临时tensor创建开销。

### 修改前后

```text
修改前：一次Indexer调用
  分块0：原评分 → 构造/应用valid与score mask → 原TopK → 构造/应用index mask
  分块1：原评分 → 再次构造/应用上述数据 → 原TopK与索引屏蔽
  分块2：原评分 → 再次构造/应用上述数据 → 原TopK与索引屏蔽
  ……

修改后：一次Indexer调用
  先为本次query范围生成完整valid和两类mask
    ├─ 分块0：原评分 → 用score mask view → 原TopK → 用index mask view
    ├─ 分块1：原评分 → 用score mask view → 原TopK → 用index mask view
    └─ 分块2：原评分 → 用score mask view → 原TopK → 用index mask view
```

也就是：**把循环里反复做的准备工作移到循环外，各块读取对应部分，不重复生成。**

这里的复用只发生在**一次helper调用内部**，不是把上一请求的mask缓存给下一请求，也不是跨layer共享可变状态。
取view不复制整块mask。代价是完整mask需要多占一点临时空间：实际`q=2048、keys=4096、TopK=512`时，两类mask共9MiB。

### 没有改什么

- 没有裁剪有效KV，没有改变query/key、GEMM形状、FP32评分公式或规约方式。
- 没有改变sorted TopK、选中索引的顺序规则，模型Indexer TopK仍是**512**。
- 没有关闭服务的Graph、Prefix Cache，也没有降低输出长度或更换测试输入。
- 没有在本次增量中修改MoE、RoPE计算、PD传输协议或Decode执行代码。
- TopK=512是Indexer选取位置的数量；DSpark仍为N6/target-q7，两者不是同一个参数。

新增开关：`XLLM_DSV4_PREFILL_INDEXER_MASK_ONCE=1`，默认关闭。
入口限定在已验证的BF16输入/FP32评分长Prefill路径，query≤2048、绝对位置合法且当前不是Graph capture。
这表示**新优化不进入Graph capture**，不是把整个服务的Graph关掉；Decode仍保持原Graph/Overlap配置。

## 4. 具体代码改在哪里

下表左侧是对应主仓库的代码位置，右侧是本次改动。实际运行使用独立构建快照，未提交/推送Git分支。

| 对应代码位置 | 改动 |
|---|---|
| `xllm/core/layers/musa/deepseek_v4/deepseek_v4_indexer_reference.h` | 新增`dsv4_prefill_indexer_masks()`；为`dsv4_tiled_indexer_topk()`增加默认关闭的`prepare_masks_once`；调用内一次创建mask，分块内用`narrow`取view |
| `xllm/core/layers/musa/deepseek_v4/deepseek_v4_attention.cpp` | 读取新开关、限制适用范围，将参数传入Indexer；记录实际query、keys和mask大小，确认优化真的命中 |
| `deepseek_v4_indexer_mask_once_test.cpp`及测试接入patch | 覆盖完整mask、分块view、冻结输入、评分/索引逐位比较，以及动态输入replay |

实际源码快照位于28机器的`xllm_musa_build_0914`容器：

```text
/tmp/pd_indexer_mask_once_model_20261006/
├── mask_once_deepseek_v4_attention.cpp
├── deepseek_v4_indexer_reference.h
├── header/core/layers/musa/deepseek_v4/deepseek_v4_indexer_reference.h
├── compile.command
├── link.sh
└── manifest.sha256
```

Windows编辑/核对副本：
`C:\Users\jiaming.wang\code\xllm-musa\.codex-pd-n6q7-20260927\recheck\indexer_mask_once\`。
它不是Linux服务直接加载的目录。

只重编了Attention编译单元，复用已有编译结果和依赖；其它layers归档成员逐个hash确认不变，未重建外部AOT/MATE库。

## 5. 最后这一步实际提升了多少

下面比较的是同样30K/1024、各64条正式请求的两个已验证版本。

| 指标 | 前一基线5bd | 最终caf |
|---|---:|---:|
| B1平均TTFT | 2289.399 ms | 2196.170 ms |
| B1平均TPOT | 8.783904 ms | 8.779710 ms |
| B1输出吞吐 | 90.752892 tok/s | 91.535690 tok/s |
| B16输出吞吐 | 405.917509 tok/s | 428.193054 tok/s |
| B16/B1倍率 | 4.472777657 | **4.677880852** |

B16输出吞吐实测提高约**5.49%**，B1 TTFT减少约93ms，B1 TPOT基本持平。
因此，这一步主要是让Prefill更快、整条PD流水线吞吐更高，不是再次重做Decode加速。

B16单请求TPOT由约17.988ms升至19.854ms，这个不利变化也保留。
不能把“整体吞吐增加”理解为“每个请求生成都更快”，也不能用`16/平均TPOT`计算本表的吞吐。

## 6. 更早的改动：属于已继承基线，不是最后这一个patch

当前版本还包含此前已验证的工作，不能把整个优化过程归结为最后的mask改动：

| 方向 | 已有基础 |
|---|---|
| 长Prefill计算 | 16K分块、compact MoE、Indexer/Gate受控BF16输入与FP32输出、HC分段等 |
| MoE与通信 | SwiGLU量化/结果合并优化；匹配SP Prefill条件时，用ReduceScatter直接得到本rank分片，避免再处理完整结果 |
| 重复数据准备与搬运 | 已有Attention相关metadata prepare-once；主Attention及Indexer的RoPE结果拼接优化 |
| Gate正确性 | 修复默认Torch流与FFI流之间缺少依赖的问题，保留`TvmffiStreamGuard`，避免尚未写完的数据被后续读取 |
| PD重连与失败收尾 | 同端点重启后刷新Mooncake descriptor/address/rkey并处理旧peer连接；handoff失败及时结束请求，避免长时间挂住 |

这些改动各自有前置实验，收益不能简单相加。尤其ReduceScatter涉及不同的浮点规约顺序，不能把整条历史优化链都称为逐位等价；本次mask增量才是保持原浮点评分路径不变。
此前因果评分裁剪、MCCL通道调参等没有显示模型收益的候选，未加入最终caf配置。

## 7. 如何确认不是“变快但算错了”

1. 单测先核对CPU整数mask参考，再核对评分、索引和动态replay输出。除补充TopK2048覆盖外，实际模型TopK512也单独验证；没有把旧2048结果截取前512作为参考。
2. 同一caf二进制只切新开关，关闭/开启各24条请求全部严格通过。
3. 开启配置通过连续64条、后置2条B1及5条Prefix重复/换题，总计95/95；Prefix实际命中序列为`0/19456/19456/0/19456`。
4. 最后另起新P实例，完成同版本B1/B16正式配对：各16预热+64正式，无重试，无usage差异，正式cached全部为0。

以上证明本次固定30K计数任务的正确性与性能，不把计数通过扩大解释为GSM8K全量准确率。

## 8. 运行产物与复核入口

所有相对路径基于：
`/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005`。

- P：`candidates/indexer_mask_once/xllm`，SHA256：`caf3db8f775224045f9b8143f1b8a3491fb4637229fd5ab18194e00b60506b43`。
- D：`candidates/reopen_decode/xllm`，SHA256：`5ea86017debbd94ba751a8ce38c74c1e2e9d39487b4ef3fcf662a24cedbaac93`。
- 源码/构建归档：`pd_indexer_mask_once_sources.tar.gz`。
- 实际配置：`PREFILL_indexer_mask_once_pair_b16_r1/launch_manifest.txt`、`DECODE_reopen_r1/launch_manifest.txt`。
- 最终审计：`indexer_mask_once_pair_20261006/paired_audit.json`，两个字段`evidence_valid`和`paired_performance_pass`均为true。
- 冻结证据：`evidence_indexer_mask_once_pair_20261006`、`evidence_indexer_mask_once_round_20261006`。

详见[验收结果与只读复核命令](DSV4_PD_30K_B1_B16_GUARD_RESULT_20261006.md)和[完整实验记录](DSV4_PD_V128_PORT_AND_30K_GOAL_20261005.md)。
原实验目录已冻结，不要直接重跑脚本覆盖结果；本说明没有改动二进制、服务或验收数据。
