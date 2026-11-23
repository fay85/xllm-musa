# GLM-5.3-Flash SGLang Decode Trace 对齐分析

## 结论

本次比较使用相同的 20,000 输入、16 输出、并发 1、同一份 prompt 文件。prompt 文件 SHA256 是 `9f2335fb9b1f52a61d61b6652ae3e75bc266c7a2b1b3b697ac99ce2b0812d409`。

| 配置 | TTFT | TPOT | 证据 |
|---|---:|---:|---|
| SGLang EAGLE，原始配置 | 798.747 ms | 22.309 ms | `sglang_pd/results/20261122_073126_810914/result.json` |
| SGLang EAGLE，带 profile | 778.947 ms | 24.245 ms | `sglang_pd/results/20261122_075134_555016/result.json` |
| SGLang 关闭 speculative，对照 | 743.045 ms | 34.998 ms | `pd_logs/sglang-alignment/no-spec/results/20261122_091255_175667/result.json` |
| xLLM 现有 decode | 约 37--39 ms | 现有 `PD_RUN.md` 和 xLLM trace | `pd_logs/decode-opt`、`pd_logs/profiles` |

SGLang 与 xLLM 的非 speculative 单 token 路径已经接近，xLLM 比 SGLang 对照组约慢 2--4 ms。SGLang 的 22 ms 主要来自 EAGLE 一轮验证后平均接受多个 token。不能通过继续删除少量 host synchronize 把 xLLM 的 35--39 ms 单 token路径直接变成 22 ms。

## SGLang trace 的事实

原始 trace 目录是：

`/data/nfs_shared/albert/Projects/scripts/LLM_inference/GLM-5.3-Flash/sglang_pd/results/20261122_075134_555016/traces`

- 8 个 Prefill rank 和 8 个 Decode rank 都有压缩 trace。
- Decode rank 0--7 的 kernel event 数是 12618、其中 rank 5 是 13061。Prefill 每个 rank 是 2557 左右。
- Decode trace 不是每个输出 token 一个 SSE 片段。rank 5 能看到 6 个 `draft`、6 个 `step[TARGET_VERIFY bs=1]`、6 个 `draft_extend` 和 6 个 `scheduler.run_batch` CPU scope。它表示 speculative 的 draft/verify 调度回合，不能用 SSE 片段数量直接推断 accepted token 数。
- rank 5 的 34 个 GLM KDA 层在 6 个回合中合计出现 204 次 `fused_sigmoid_gating_delta_rule_update_kernel`。mHC 的 `mhc_pre_big_fuse_with_norm_tilelang_kernel` 出现 540 次，`mhc_fused_post_pre_fma_tilelang_kernel` 出现 270 次。
- DSA decode、DeepEP low-latency dispatch/combine、FP8 grouped GEMM 和 mHC/KDA fused kernel 都在同一个 decode trace 中。不能只看某一个 KDA kernel 的耗时判断整个 TPOT。
- 22 ms 的统计来自输出 token 的时间跨度。profile 会改变 TPOT，带 profile 的 24.245 ms 应作为 profile 上界，非 profile 22.309 ms 才是基准。

## SGLang 源码对应关系

源码根目录是 `/data/nfs_shared/albert/Projects/sglang/python/sglang/srt`。

### Decode graph 和同步

- `model_executor/runner/decode_cuda_graph_runner.py` 282--318 行按 batch bucket 计算 capture/compile/max batch。
- 355--407 行建立固定地址的 token、position、KV location、sequence length、DP token 数、KDA state index buffer。
- 541--621 行把实际 batch 选择到 graph bucket，不满足条件时回到 eager。
- 845--1062 行 warmup、构造 metadata、初始化 TBO/DeepEP/PP buffer 后捕获完整 decode graph。
- 1088--1266 行 replay 前只更新 registry 和 device metadata，实际 token 行异步拷贝到稳定 buffer。
- `model_executor/runner_backend/full_cuda_graph_backend.py` 77--208 行只用 capture/replay stream 的 wait。MUSA replay 前后是 stream-local event ordering，不是每步 device-wide synchronize。
- `batch_overlap/two_batch_overlap.py` 506--845 行切分两个 ForwardBatch，872--1057 行跨层交错执行。并发 1 时 TBO 的收益有限，不能把它当成 22 ms 的主要来源。

### GLM5、KDA、DSA、MoE

- `models/glm5_next.py` 1083--1186 行使用 LayerCommunicator/mHC communicator，1313--1438 行接入 speculative forward 和 decode 运行路径。
- `layers/radix_linear_attention.py` 78--160 行把 KDA decode 交给 attention backend，并使用固定 output buffer。
- `layers/attention/dsa_backend.py` 3380--3556 行选择 tilelang、FlashMLA/FlashInfer 等稀疏 decode backend，3691--3750 行使用 FP8 KV 和 page metadata。
- `layers/moe/mega_moe.py` 199--234 行把 shared expert 放在 alt stream，237--389 行执行 FP8 MegaMoE，402--428 行使用固定 recipe/layout。
- `server_args.py` 999--1162 行定义 TP、EP、DP attention、DCP。原始 Decode 启动命令同时使用 TP8、EP8、DP1、DP attention、DP LM head、DeepEP low-latency、linear-attn triton、DSA tilelang。
- 原始 SGLang Decode 还开启 `num-continuous-decode-steps=2`、EAGLE 5 steps、topk 1、draft tokens 6。这个配置允许一次服务回合验证一批 draft token。

## 当前 xLLM 的对应关系

当前 xLLM P/D 组合是 Prefill CP8/EP8，Decode TP8/CP1，Decode routed expert 通过 EP8 分片，Decode graph 已开启，schedule overlap 已开启。xLLM 现有 trace 的单步主要工作包括 KDA recurrence、mHC、MegaMoE stage1/stage2、TP collective 和 KV restore。

最重要的结构差异是：

1. xLLM 当前 Options 日志是 `num_speculative_tokens: 0`，没有 draft/verify 回合。
2. xLLM MUSA 的 GLM5 KDA 在 `core/layers/musa/glm5_next/glm5_next_linear_attention.cpp` 490--494 行显式禁止 spec verify，438 行的模型 graph 也禁止 `is_spec_verify`。
3. `python/models/glm5_next_mtp.py` 已经存在，但 `python/model_platform_support.py` 将 `glm5_next_mtp` 标记为 NPU-only。MUSA 构建不能用 Python executor，`models/models.h` 也没有 MUSA GLM5 MTP native registration。
4. 因此给 xLLM 增加 `--draft_model`、`--num_speculative_tokens=6` 只能证明配置能传入，不能证明 MTP 已实现。

## 本次修改

`tools/musa_glm5_3_vl_bench/start_service.sh` 新增可选参数：

- `GLM5_DRAFT_MODEL_PATH`
- `GLM5_NUM_SPECULATIVE_TOKENS`
- `GLM5_SPECULATIVE_ALGORITHM`

它们默认关闭，只有 `GLM5_NUM_SPECULATIVE_TOKENS>0` 才向 xLLM 传递 speculative flags。这样可以复用同一个启动脚本进行 A/B，不会改变原有 P/D 默认行为。

用 `GLM5_NUM_SPECULATIVE_TOKENS=6` 实测时，当前远程 binary 在 draft worker 创建阶段报：

```
Unsupported model type: glm5_next_mtp
Check failed: model_ != nullptr Failed to create model.
```

本次实验已停止全部失败 worker，sj7/sj8 当前无运行中的 xLLM 或 SGLang 服务。

## 对齐实施顺序

### 第一阶段：把单 token 路径对齐到 SGLang no-spec

1. 保留 Decode TP8/EP8，先不要在并发 1 引入 TBO 或 DP attention 改拓扑。
2. 以 no-spec SGLang 的 34.998 ms 为基准，逐个比较 xLLM trace 中 KDA、mHC、MegaMoE、TP collective、KV restore。
3. 确认 graph replay 使用固定 token/position/KDA state/page table 地址，host 侧只做异步 H2D 和 stream event。
4. 对 token=1 的 KDA 做专用 decode kernel，把当前的 state index select、chunk recurrence、final-state scatter 和不必要的中间 tensor 写回收敛到单个路径。Prefill 继续使用现有 chunk/prefill kernel。
5. 对 MegaMoE stage1/stage2 和 shared expert 对齐 FP8 layout、dispatch/combine buffer 及 alt-stream event，避免 host wait。
6. 在相同请求上比较 8 个 rank 的 GPU elapsed、collective 时间和 CPU launch gap。只有这些数据下降才接受改动。

### 第二阶段：实现 GLM5 MTP native MUSA

需要补齐以下能力，不能只改 launcher：

1. 为 `glm5_next_mtp` 注册 MUSA native model 或为 MUSA Python backend 实现等价的 MTP draft executor。
2. 加载 checkpoint 的 appended MTP layer、enorm/hnorm/eh_proj、draft embedding/lm_head 和 indexer 权重。
3. 为 decode 预留 `num_speculative_tokens+1` 的 KDA/MLA cache checkpoint，完成 accepted/rejected token 的 KDA state、conv state 和 MLA KV rollback。
4. 将 target verify 改为固定 width=7 的 graph bucket，验证 token ids、logits、KDA state、MLA cache 和 P/D KV transfer。
5. 在 P/D 模式中明确首 token 的 bootstrap 和 D 侧 verify 的 ownership，保证 P 不执行 draft worker。
6. 用 SGLang 的 6 回合 trace 对照 accepted length、verify width、每回合 GPU elapsed。目标是保持单回合约 35 ms，同时让平均每回合提交至少 1.6 个 token，才能接近 22 ms TPOT。

### 第三阶段：并发优化

TBO、DP attention、DCP 和更激进的跨层 overlap 放到并发 8/16。并发 1 时它们会增加 metadata 和调度复杂度，不能作为当前 22 ms 差距的第一修复项。

## 复现命令

SGLang no-spec 对照产物：

```
pd_logs/sglang-alignment/no-spec/results/20261122_091255_175667/result.json
pd_logs/sglang-alignment/no-spec/results/20261122_091312_156438/traces/
```

SGLang 原始产物：

```
/data/nfs_shared/albert/Projects/scripts/LLM_inference/GLM-5.3-Flash/sglang_pd/results/20261122_073126_810914/
/data/nfs_shared/albert/Projects/scripts/LLM_inference/GLM-5.3-Flash/sglang_pd/results/20261122_075134_555016/
```

xLLM speculative flags 仅用于后续 native MTP 完成后的实验：

```
GLM5_DRAFT_MODEL_PATH=/data/nfs_shared/models/GLM-5.3-Flash \
GLM5_NUM_SPECULATIVE_TOKENS=6 \
GLM5_SPECULATIVE_ALGORITHM=MTP \
bash start_decode.sh start
```


