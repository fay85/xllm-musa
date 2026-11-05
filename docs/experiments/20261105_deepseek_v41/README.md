# DeepSeek-V4.1-Flash 0909：初次两机 PD 完成情况

**本轮“两个简单题 + 两组各 5 次 eager 性能”的初步部署目标已完成。** P=dev4、D=dev7，各 8 张 S5000；attention TP8，专家 EP8/MoE-TP1。当前实现存在明确的量化与高性能算子接入边界，见下文与详细模块说明。

## 结果

| 实际输入/输出 token | 正式次数 | 平均 TTFT(s) | 平均 TPOT(ms) | 平均端到端(s) | 平均端到端输出 token/s |
|---|---:|---:|---:|---:|---:|
| 256/256 | 5 | 5.9598 | 988.6838 | 258.0742 | 0.9920 |
| 2048/256 | 5 | 29.2549 | 981.8722 | 279.6323 | 0.9155 |

每组单独预热一次不计分，C1、temperature=0、top_p=1、chat 非思考模式、prefix cache 关闭、ignore_eos=true 固定输出长度。工具为原版 EvalScope 1.14.1。全部正式请求都核对了原生 request_id、P prefill、D success/sequence 计数；不是仅检查请求中的 max_tokens。TPOT 按 `(总时延−TTFT)/(输出 token−1)`；端到端输出速度按 `输出 token/总时延`。它们是服务指标，不能当作单个 kernel 耗时。

基础题通过真实两机链路：`1.3 + 2.9 → 4.2`（21/4 token），`7 × 8 → 56`（17/2 token），都正常 stop。这些基本题不替代完整精度评测。两组测速 prompt 为解释工程实验记录的作用，附 padding 后由官方 encoder 构造精确输入长度，完整文本与 token IDs 已保留。

## 可以经得起追问的运行证据

- Run：`pd_dev4_dev7_v2_nic9`；性能子目录 `performance_v2`，失败的 `performance_v1` 是客户端缺依赖导入失败，没有算作一次请求。
- 两个基础题、两个预热、十个正式请求，共 14 个 request_id 的 P/D 成功记录齐全；D 八个 rank 各完成 14 次非空缓存拉取。
- 源码计算实现锁在 `1a4a8a398962b2b8e500c99d81a5f03c7ac73924`，二进制 SHA256 为 `7265c5997a5877d548178a3c80522034e3aeac6e5d3358763878588bf8bbfbb9`。最终证据提交可在其后追加，但没有在测量期间换模型计算代码。
- 镜像 `registry.mthreads.com/presale/devtech/xllm:1102-dev`，MATE0.2.7，Torch/TorchMUSA2.9.1。镜像 digest、依赖与模型来源见 `raw/evidence/tested_version_matrix.json`。
- 旧 run `pd_dev4_dev7_v1` 自动 RDMA 选路失败后，正常退出并归档。第二轮只把本任务 Mooncake 白名单设为 `mlx5_9`；没有改主机网卡/驱动。当前成绩受这个单 NIC 配置限定。
- 16 个 rank 和两个推理容器全部正常退出码 0。各主机退出后 GPU 状态的原始输出保留；没有 SIGKILL 或服务器重启。

## 当前模块接入情况

已接通 xLLM 模型注册、Python 执行器、MUSA eager CSA2 metadata、原始权重加载、TP/EP 的 MCCL 通信以及真实 P/D 状态传输。CSA2、Mega-mHC、Engram、路由专家和共享专家都参与了完整模型运行。Engram tokenizer hash 已与官方 129280 项完整映射对齐；专家保留原始 packed MXFP4 存储，仅对活跃专家在 GPU 临时展开为 BF16，满足当前八卡显存约束。

**尚未接入 MATE MegaMoE serving，也未实现与官方完全一致的 block-32 FP8 激活量化和 indexer FP4 量化语义。** MATE FP4FP8 primitive 已有有限真实单专家投影验证，可作为后续接入依据；其 per-token activation scale 与官方 block-32 的区别还需明确。CSA2/RoPE/Mega-mHC 当前主要是 TorchMUSA 张量路径，不能称为都用了 MATE 融合实现。FP16/BF16 RMSNorm 可调用 MATE，FP32 使用 GPU reference。没有将专家矩阵计算回退到 CPU；少量路由元数据会同步到 CPU 做循环控制。

详细到每个模块、文件层次、失败尝试、验证范围与待办，见 [各模块实现说明](适配说明.md)。复现入口见 [复现与结果阅读说明](复现与结果阅读说明.md)。本轮没有扩展 MathQA100、trace、graph 或性能优化 A/B。

## 数据在哪里

- 本地当前文件夹：`王家明/20261105跑通PD分离/`。
- NFS：`/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离/`。
- 原始模型只读：`/data/nfs_shared/models/DeepSeek-V4.1-Flash`；48 个分片已核对官方初始版本 SHA。0909 是沿用的简称，不把它写成已经严格证明的官方发布日期。
- 共同仓库分支：[xiaopangDeepseeekV41Flash](https://github.com/fay85/xllm-musa/tree/xiaopangDeepseeekV41Flash)。没有向个人 fork 提交，也没有提 MR/PR。
- Git 中结果：`docs/experiments/20261105_deepseek_v41/`；私有 MATE 源码、权重、镜像及凭据不进入 Git。

1017 权重下载是独立任务；本轮正式测速一直使用 0909，没有中途换成 1017。下载完成与否应查看独立资产任务的验收报告。

## 可直接复制给王家明的汇报（同时作为 commit message）

```text
test(musa): 我已跑通 V4.1 Flash 0909 双机 PD 并完成两组 TP8 eager 测速

家明，我已基于共同仓库 1103-merged-dev（0cfd8270），在个人分支
xiaopangDeepseeekV41Flash 完成这次初始部署。P 使用 dev4，D 使用 dev7，
各 8 张 S5000；attention TP8，专家 EP8 / MoE-TP1，eager、并发 1。
两台都是启动前确认八卡空闲后使用，没有停止外部任务或修改主机驱动。

我先通过真实 Router/P/D 链路验证了 1.3+2.9=4.2、7×8=56。
随后使用原版 EvalScope 1.14.1，各组预热 1 次、正式 5 次：
- 256/256：平均 TTFT 5.9598 s，TPOT 988.6838 ms，
  端到端 258.0742 s，端到端输出 0.9920 token/s。
- 2048/256：平均 TTFT 29.2549 s，TPOT 981.8722 ms，
  端到端 279.6323 s，端到端输出 0.9155 token/s。
10 次正式请求的真实输入/输出长度均与设定一致；预热不进入均值。
TPOT=(总时延-TTFT)/(输出 token-1)，端到端输出速度=输出 token/总时延。

我完成了 MUSA 模型注册/执行入口、MCCL TP/EP、原始 packed MXFP4
专家加载和按活跃专家在 GPU 临时展开计算，并修正 Engram tokenizer hash：
129280 项映射和官方逐项一致。CSA2、Mega-mHC、Engram、shared expert
已参与这次真实推理；构建合并缺口、缓存组映射和正常退出问题也已处理。

这个版本的专家计算仍是 BF16 参考实现，尚未接入 MATE MegaMoE；
MATE FP4FP8 primitive 已做有限的真实单专家投影验证，但完整 serving、
激活 block-32 量化、indexer 官方量化语义和 CSA2/mHC 融合仍需后续接入。
MATE FP16/BF16 RMSNorm 有独立验证，其他 reference 张量计算在 MUSA GPU 上。
两道基础题仅满足本轮轻量准入，不能作为官方完整精度对齐；
本次也没有测 graph、DSpark/MTP、高并发或优化收益。

第一轮 P/D 因自动选择管理网 RDMA 设备失败，我归档后正常退出，
第二轮仅给本任务 Mooncake 设置 MC_TE_FILTERS=mlx5_9 后通过。
所以这些数字对应单 NIC 初始配置，不能代表多 NIC 或融合后的最佳性能。
原生定向回归 40 项通过；综合缓存回归 189/264 通过、75 项失败保留，
主要涉及旧 V4/MTP 路径，不宣称整个合并仓库全量回归通过。

本次 14 个请求（2 基础+2 预热+10 正式）均有匹配 P/D 成功记录；
D 各 rank 均有 14 次非空缓存拉取。结束后 16 个 rank、两个推理容器
全部正常退出码 0，没有使用 SIGKILL 或重启服务器。

原始请求/响应、SQLite、P/D rank 日志、逐次 CSV、简要结果、实现说明
已放入 docs/experiments/20261105_deepseek_v41/；NFS 完整工作目录为
/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离/。
我只提交该个人分支，不提 MR/PR。后续先明确量化数值合同，
再接入 MATE 专家高性能路径并进行同条件回归与测速。
```
