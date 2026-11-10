# V4.1 后续算子接入与 C++ 对照：下一轮执行说明

编写日期：2026-11-10。本文件用于后续用户明确启动的 goal；编写文件本身不代表已经启动新 goal。先读同目录 `今日工作总结.md`，再以原始结果文件为准确认阶段一、二的最终状态。

## 0. 本轮结束后的起跑点

A/B/C/D/E 的 256/256 与 2048/256 两组均已完成 1+5，原语接入和对应性能证据都已保存，后续不重复这些未改动的实验。共享 SwiGLU 在 D 侧有约 0.6%–0.7% 的本轮 TPOT 收益；post/pre 未获得整模型增量收益。最后加入 metadata 后，相对 post+pre 配置，256/256、2048/256 的 TPOT 分别下降 3.55%、4.23%（负值表示变慢）。

下一轮最先处理真实整模型的同前缀数值缺口：修复或绕开当前错误的原始 token 输入验证入口，保留本轮已经产生分歧的固定 prompt 和前缀，定位首次 hidden/logits 差异。C/D 等输出变化没有充分归因，不应直接全局打开 mHC。源码默认 reference，已编译融合实现以开关选择。之后再按第 5 节推进专家高性能路径，避免把新量化差异和本轮未定位的差异叠在一起。

## 1. 要完成什么

在共同仓库 `fay85/xllm-musa` 的个人分支 `xiaopangDeepseeekV41Flash` 上，继续基于 1103 补齐 DeepSeek-V4.1-Flash-1017 的算子和必要模型语义。先利用家明 V4 的现有实现，再结合 SGLang 与京东交付实现确认接口。每项改动都要先检查数值，再测性能，保留失败记录。不要把“函数能调用”当成“模型语义已经正确”。

Python 模型主线继续推进；C++ 路线先做使用相同 GPU 算子的局部对照。只有有必要且原生模型具备完整文本前向/P-D 续算后，才作整模型双分支比较。不把局部 C++ 调用测试称为原生 C++ 模型部署。

## 2. 固定基线与参考

| 项目 | 固定值/位置 |
|---|---|
| 共同仓库 | https://github.com/fay85/xllm-musa |
| 我方分支 | https://github.com/fay85/xllm-musa/tree/xiaopangDeepseeekV41Flash |
| 1103 父基线 | `0cfd82700e164e8aa7951064b652925dd6b24d18` |
| 本次开始时的 SwiGLU 提交 | `34815f6ac23de6c3145a3392b7649874a37d4f27`；后续以今日总结记录的新提交为准 |
| 家明参考 | `feat/dsv4-pcp-mate027-verified-20261029` / `f8dd0663279e9085365ade309d9823b7b438c4e6` |
| 原版 1017 模型 | `/data/nfs_shared/models/DeepSeek-V4.1-Flash-1017`，HF commit `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| 当前镜像 | `registry.mthreads.com/presale/devtech/xllm:1102-dev`；镜像 ID 和依赖快照见 manifest |
| MATE | 当前实际运行环境为 0.2.7；不能凭之前 GLM 用过 0.2.8 就写成此次也是 0.2.8 |
| 本地源码 | `../20261105跑通PD分离/代码/xllm-musa` |
| 参考调研 | `../20261110三项参考与路线规划/` 内五份 Markdown 和证据 |
| 交付指南 | `../DeepSeek V4.1 Flash-1017 部署测试指南（11_5）.md` |

注意：家明的固定参考提交不是上述 1103 的直接祖先，不能按提交日期推断继承关系。逐文件核对已经整合的实现。不要整条合入 1110，也不要默默切换权重。

## 3. 机器、目录和操作边界

主场地是 sj9（P，10.121.32.7）和 sj10（D，10.121.32.9），各 8 GPU，1P1D。使用原版 1017。sj11/sj12 不在此次授权范围。只有 sj9/sj10 被外部占用时才考虑 dev4–7 的空闲两台和 0909，且报告必须分开。

sj 的 NFS 是 `10.121.34.78:/data/nfs_shared`；dev 的 NFS 是另一套存储。路径同名不代表同盘。

当前本轮目录：

```text
/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops/
  00_manifest/              # 模型、镜像、依赖、旧容器归档、源码对应关系
  01_sources/               # 每次运行冻结的源码，不在运行时覆盖
  artifacts/                # 构建产物、模型只读视图、隔离测速环境
  scripts/                  # 启动、停止、验收、测速
  runs/<configuration>/     # 每个 rank、控制服务、supervisor 原始日志
  02_correctness/           # 基础问答、数值回归、P/D 检查
  03_benchmarks/            # EvalScope 逐请求数据库和汇总
  04_reports/               # 结论与交接
```

本轮容器为 `xiaopang1110_v41_prefill_sj9`、`xiaopang1110_v41_decode_sj10`。先核对它们是否仍在执行当前请求，不根据名字或 Up 时间停任何其他容器。两个旧 GLM 容器已得到用户明确许可，归档并正常停止；该许可不扩展到别人的任务。

CPU 编译可使用 dev6 的 `xiaopang1105_dsv41_build_dev6`。单卡测试须先确认具体 GPU 空闲，容器名称包含 xiaopang、任务及版本。不可修改驱动、重启服务器、重启 Docker；退出困难先收集证据，不能循环 stop/kill。正常 TERM 后等待并检查线程状态，不自动升级 SIGKILL。

访问入口与凭据通过 `../../sj9_12_登陆工具/sj_access.py` 和已有访问文档读取，不写入提交。不要重复建立不必要的 VPN 会话。

## 4. 先收尾阶段一、二尚未通过的验收项

先读取今日总结和每个配置的状态；已通过、未改动的项目不重复从头跑。

1. 确认 SwiGLU A/B 两组输入输出均有 **1 次暖机 + 5 次成功正式请求**，实际 native usage 为 256/256、2048/256。A 是 P/D 都 reference；B 是 P reference、D v4_fused。不能把服务已起来、客户端退出码 0 当成测速完成。
2. mHC 顺序为 post → 消费 carried pre 的 pre-combine → metadata/Sinkhorn。每次只改变一项，记录累计配置。三项都分别保留原语计时、真实层、小模型及模型级结果。若某项未通过，只回退该项，保留前面的已通过配置。
3. tiny fixture 要同时检查整段 prefill 与逐步 decode、累计配置间 hidden/logits、greedy token 一致性。它是随机小模型，不能替代 1017 真权重验收。
4. 用原版 1017 的层 0/1/14/39、attn/ffn mHC 参数做实际宽度 H=5120、hc_mult=4 的数值回归。保存权重子集来源、张量名与 SHA。
5. 对原版模型的固定 token 前缀，补齐整段 prefill、逐步 decode、跨 P/D 续算的一致性证据。若目前只有基础问答通过与随机小模型结果，明确写出缺口，不能宣称全模型逐层数值已经验收。
6. 补测 Chat raw-ID 控制也未通过：21 个 ID 对应 `1.3+2.9`，messages 问 `7×8`，实际输入 17 token、回答 56，说明客户端 token_ids 未作为最终输入；输出 usage=4 而 logprob 项仅 3。不要继续直接使用这一入口作固定前缀对照。固定前缀探针的 Completion 接口曾返回 V2 identity 错误，Chat 与 Completion 对 attempt_seq 的传递不一致，详见 `阶段一二/固定前缀探针与V2协议缺口.md`。先单独修复协议及失败请求资源回收，再复做这项验证；不要把 HTTP 错误当作模型数值错误，也不能在当前性能请求前反复发送已知不兼容探针。
7. 原始 MUSA PyTorch Sinkhorn 在饱和输入上曾有列求和精度问题。保留诊断输入和 CPU FP64 对照；不能把有误的旧结果作为唯一真值，也不能只为让测试通过放宽阈值。此 CPU oracle 仅属于测试，不是推理 CPU 回退。

本次代码通过 `XLLM_DSV41_MUSA_SHARED_SWIGLU` 和 `XLLM_DSV41_MUSA_HC_{POST,PRE,METADATA}` 独立选择 `reference` / `v4_fused`，在模型构造时固定。更换配置应正常结束旧 rank 后按冻结配置重启，不修改运行中的文件来冒充同一版本。

阶段二已出现需要继续核查的具体样本：256/256 性能 prompt 的 B/C 请求完全相同，输出在第 54 个字符后由 `preserves...` 变为 `creates...`。各配置自身五次相同。见 `阶段一二/证据/C_256_output_divergence.json`。这是开放式回答，不能直接判错；但在声称数值等价或将小幅吞吐变化全部归因于算子前，需要相同前缀的数值检查。优先保留这些固定输入，避免只换另一道恰好一致的题。

## 5. Python 主线后续待办：按收益、依赖与可验证性排序

### P0：确定可用的专家 GEMM 数值接口

**来源：** 家明 V4 的 grouped GEMM/GEMV、GPU 路由与专家通信；官方 V4.1 reference；京东 SGLang 交付实现。

当前最突出的性能缺口不是 SwiGLU：路由专家仍逐步将原始 packed FP4 解码为 BF16，逐专家执行，并将分组元数据带回 CPU。先把真实接口查清楚，避免继续围绕小算子优化而不碰主耗时。

要查：原版 FP4 E2M1、E8M0 scale、K group32、FP8 block32 激活的 GEMM/分组 GEMM 是否已有 MUSA 实现，支持哪些 M/N/K、累加精度、输出类型、layout、stream 和 workspace。分别看 decode 小 M 和 prefill 大 M。

从交付镜像提取 `/workspace/sglang` 及实际 MATE、MUTLASS、TileLang、TVM-FFI、DeepEP/mtshmem 依赖，记录 digest/Git SHA/源码树 SHA。只拉适用的一套镜像；本机中转大文件校验成功后清理。

**关键分叉：** `MooreThreads/DeepSeek-V4.1-Flash-1017-W4AFP8-mt` 的 routed/MTP 权重是重新量化的 signed INT4 group128；它不能直接替代原始 FP4 group32。若只有交付 INT4 路径可用，把它列成独立 Q 方案，另做精度验收，禁止混入原版 O 路线的 A/B 表。

若接口确实缺失，给算子同事的请求应包含一个最小输入样例、张量 shape/dtype/scale、官方方程、现有函数签名、预期误差和目标 shape，而不是笼统说“请支持 MegaMoE”。记录缺口后继续 P1/P2，不被外部协助卡死。

### P1：GPU 路由与专家分组、combine

**来源：** 家明 V4；交付 SGLang；官方参考。

按顺序替换分数/选专家、token 分组、专家计算、结果汇合。先核对 V4.1 的路由公式、用于选择的 bias、实际乘回输出的权重、top-k 和缩放，不能照搬其他 GLM/V4 的参数。

测试必须覆盖：一个 rank 没有 token、极不均匀路由、相同专家重复命中、跨 rank token 分发、所有 rank 的 collective 顺序、共享专家只计算/归约一次。记录 Attention TP8、路由 EP8/专家内 TP1、共享专家的实际分片方式，别统一写“全部 TP8”。

优先消除 `.cpu().tolist()` 和每 token 重复解量化。若为了缓存 BF16 权重增加显存，先估算各卡峰值，不能靠触发 OOM 试错。

### P2：CSA2 中可单独验收的算子

**来源：** 家明 Indexer/Sparse Attention、公开 SGLang、交付镜像、官方 V4.1 reference。

先接 RoPE、norm、score/top-k 等无状态或状态少的原语；随后接压缩 KV、物理页索引、跨层 KV/Top-K 复用。记录 Indexer 的官方量化语义、scale 和存储格式，不能把目前 BF16/INT8 近似计算写成官方 FP4 实现。

覆盖空页、跨页、压缩边界、SWA 窗口边界、多请求隔离、prefill/decode 混合和 P/D 状态传递。对有状态算子同时比较输出及 cache 内容，不能只看最后一个 token。

### P3：Engram、CED 和完整 Mega-mHC

**Engram：** 保留已经核验的 tokenizer hash 和表加载。逐项优化查表、gate/norm、卷积/状态更新、CPU 大表到 GPU 的传输。CPU 表放置是内存策略，不等于把模型主体算子放到 CPU。必须测时间和峰值内存。

**CED：** 参考 bounded replay，单独实现并验证 Decoder Prefill 的尾部 SWA Replay、跨层 cache 复用和请求续算。不能只改一个开关。长 prompt、P/D 切换、复用层的状态必须与参考一致。

**Mega-mHC：** 当前先接独立 post/pre/metadata；完整融合仍要保持 shifted pre_mix：当前子层计算出的 pre 给后一个子层使用。不要直接以 V4 整体 forward 替换 V4.1。FP32 投影、RMS 统计和 Sinkhorn 的融合须另测精度，不与算子接口改动同时混在一个实验里。

### P4：Graph、DSpark/MTP 与其他并行

算子覆盖和状态一致性稳定后再做。Graph 检查动态分配、shape、地址稳定、collective 捕获；可先分段，但标清 eager 区段。DSpark/MTP 需要草稿权重、验证、接受率和拒绝时状态回滚，不能只打开 flag。

后续并发、CP/DP/TP、custom-allreduce 等都作为独立实验。先验证适用的拓扑与形状；不要将跨机 RDMA、单机 MCCL 和自定义归约混写成一个通信机制。

## 6. C++ 路线：先做公平、有限的对照

拟用分支 `xiaopangV41FlashCpp`，共同父基线仍为 1103 的 `0cfd8270...`。是否已创建以 Git 实际状态为准，不在文档里虚构。

**第一步，局部调用对照：** 同一输入/权重/输出、同一编译 GPU kernel、同一 stream，分别从 Python binding 与 C++ 直接调用共享 SwiGLU/完整共享 MLP。两边都计入独立 gate/up 的拼接与分配；再测同一 mHC 原语。记录 host wall time 与 GPU event time、暖机、重复次数及同步边界。这只能说明该局部调用开销。

**第二步，需要时实现原生文本模型：** 新增 CausalLM 注册和 factory，Decoder、权重加载、shifted mHC、CSA2、Engram、LM head、P/D cache/state。参考家明 V4 的组织，但按 V4.1 语义实现。仅调用现有 Python forward 的包装不算原生 C++ 模型。

**第三步，算子进度对齐再比较整模型：** 两路线共用相同原版 1017、同一量化语义、TP/EP、共享专家分片、缓存、Graph、所有算子实现和依赖版本。对齐必要的 Folly/Provider/服务通信修复。先做基础题及固定前缀数值，再测两组 1+5。

如果 C++ 比 Python 更快，分析是 host 调度、语言边界、内存分配还是配置不同；如果差异落在波动内，也如实记录。不能预设 C++ 必须更快，也不能拿原版 FP4 的 Python 路线与 INT4 重量化的 C++ 路线解释语言收益。

## 7. 统一性能协议与证据要求

- 场景：1P1D，各 8 卡；eager；C=1；temperature=0；chat 模式；关闭推测解码和跨请求 prefix cache。
- 两组：实际输入/输出 256/256、2048/256；各 1 次暖机 + 5 次正式请求。请求与 token ID/hash 固定；输出足额需要 ignore_eos，但正确性题不能靠强制长度判断是否答对。
- 工具：当前使用未修改的 EvalScope 1.14.1 perf。固定隔离环境的依赖版本；不要改变模型服务的 Python 环境来修客户端依赖。
- 指标：逐请求 TTFT、TPOT、端到端时延、输出 token/s；组 QPS、总输出吞吐；均值、中位数和范围。明确 TPOT 计算口径及最后结束事件是否计入。
- 检查成功标记、HTTP/SSE 错误、实际 prompt/completion usage、结束原因。暖机不混入正式统计，失败请求不当作高吞吐样本。
- 保存命令、环境、源码 commit+dirty patch、二进制 SHA、镜像 digest、模型 SHA/版本、完整 rank 日志、原始 SSE/数据库与汇总。若工具数据库没有完整 SSE，另保留验证请求的逐块时间记录，并写清各证据覆盖范围。
- 对小幅提升不要只看均值；交错重复或 A→B→A 验证稳定性。先完成必要比较，只有波动/失败/新改动才扩大测试。
- 本轮不承诺 60% SOL。理论上限和算式尚需确认；不要将未统一硬件、权重/量化与负载的 H20/H200 数字当作此次目标验收值。

## 8. 完成时交付

更新本地与 NFS 的结构化报告；每个算子列来源、原实现、改动文件、计算所在设备、数值范围、微基准、模型级结果、未通过项与保留/回退决策。失败也留日志。

只向共同仓库的我方个人分支提交 commit，不向 fork 提交，不创建 MR/PR。commit message 用第一人称向家明汇报：做了什么、具体参数/数字、没有完成什么、下一步是什么。不能将真实算子接入来源都写成 MATE：家明仓库的自有 MUSA kernel 与 MATE 库调用应分别说明。

每次提交前执行仓库要求的代码复核；C++ 用 clang-format **20.1.6** 格式化，记录实际编译和回归结果。最后列出仍运行的我方容器及用途，清理本机中转大文件；不删除原始证据或模型权重。
