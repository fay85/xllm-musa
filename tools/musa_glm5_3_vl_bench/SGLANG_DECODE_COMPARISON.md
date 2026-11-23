# SGLang Decode 实现对照与 xLLM 借鉴建议

## 结论

本文对照容器中的 /data/nfs_shared/albert/Projects/sglang 源码，重点查看 GLM5 decode graph、KDA/DSA、MegaMoE、批间 overlap 和并行策略。该目录当前不是可用的 Git checkout，所以本文按源码快照的文件和行号记录。本次没有在 SGLang 服务上运行同一份 20k/16 请求，性能数字采用当前 xLLM MUSA trace。

SGLang 的 decode 优化是一个执行栈：

1. 按 batch size bucket 捕获完整 decode graph，输入和 metadata 使用稳定地址。
2. replay 前只更新设备 buffer，用 stream event 表达 producer 到 graph stream 的依赖，避免每步全局同步。
3. 用 TBO 把 batch 切成两个 micro-batch，跨层交错计算、MoE dispatch/combine 和通信。
4. 使用专门的 DSA/MLA decode kernel、FP8 MegaMoE、mHC fused kernel 和 shared-expert 多流 overlap。
5. 并发提高后再使用 DP attention、EP/A2A、DCP 和 speculative decode 扩展吞吐或有效 token 数。

对当前 xLLM 的借鉴顺序是稳定 graph replay、decode 专用 kernel、MoE 和 TP collective 优化。TBO、DP/DCP 和 speculative decode 需要先解决 xLLM 的调度、状态回滚和 P/D 拓扑契约，不能只增加一个启动参数。

## SGLang 的实现

### Decode graph

文件是 python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py。

- 282～318 行计算 capture_bs、compile_bs、max_bs 和 max_num_token。
- 355～407 行创建 DecodeInputBuffers 和 CudaGraphBufferRegistry。token、position、request/cache location、sequence length、DP token 数和 mamba/KDA 索引都放在固定 buffer。
- 541～621 行将实际 batch 映射到已捕获 graph bucket。不能满足 bucket 或动态条件的 batch 走 eager。
- 845～1062 行先 warmup/autotune，再按大到小的 batch 捕获 graph，capture 期间同时初始化 attention metadata、TBO、DeepEP 和 PP buffer。
- 1088～1266 行 replay 前写入 registry、计算 graph key、更新 attention metadata，只拷贝实际 token 行。无效 pad 行由 planner 预先置为 sentinel。
- 1273～1350 行 replay 后按实际 token 数切片输出，并用 read-done event 发布共享 cache 读取完成。

完整 graph backend 在 python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py。

- 77～93 行分配全局 graph memory pool，并保存 capture/replay stream。
- 102～156 行两次 warmup 后捕获完整 model forward。
- 187～208 行 MUSA replay 前执行 replay_stream.wait_stream(current_stream)，replay 后执行 current_stream.wait_stream(replay_stream)。这是 stream-local ordering，不是 device-wide synchronize。

python/sglang/srt/model_executor/cuda_graph_config.py 30～169 行定义了 decode/prefill 独立的 graph 配置。decode 默认 full，也支持 breakable。breakable backend 在 python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py 56～249 行把 attention 或 mamba 动态边界切成多个 graph segment，并在 replay 侧重算动态 metadata。

### TBO

文件是 python/sglang/srt/batch_overlap/two_batch_overlap.py。

- 64～96 行对 decode 每条序列按 1 token 计算两个子 batch 的切分位置。
- 338～393 行在 capture/replay 时写入固定的 split metadata。
- 395～471 行收集 DP ranks 是否都能运行 TBO。低延迟 A2A MoE 等不兼容组合会关闭 TBO。
- 506～845 行切出两个 ForwardBatch 子视图，维护 token range、padding、sequence metadata 和 speculative mask。
- 872～940 行调用 execute_overlapped_operations，以 delta_stages 交错执行两批操作。
- 948～1057 行只对 hidden state、residual 和 positions 做切片/padding。

GLM5 调用点在 python/sglang/srt/models/glm5_next.py 1363～1415 行。模型先执行 dense 层，再把后续层交给 TBO。TBO 主要服务 batch 大于 1 或持续请求，单并发 1 可能只有切分开销。

### GLM5 attention、通信和融合

python/sglang/srt/models/glm5_next.py 12～18 行使用 FusedRMSNormGated 及 MUSA mHC 的 hc_pre、hc_post、hc_post_pre。66 行使用 RadixLinearAttention。python/sglang/srt/layers/radix_linear_attention.py 78～160 行将 KDA decode 转给 attention backend，piecewise graph 使用固定 output buffer。

Glm5NextDecoderLayer.forward 在 glm5_next.py 1083～1186 行通过 LayerCommunicator、MHCLayerCommunicator 或 DSA/CP communicator 处理 attention 和 MLP 输入输出。MLP all-reduce 可以标记为和下一层融合，DP padding 可使用 reduce-scatter。

DSA decode 入口在 python/sglang/srt/layers/attention/dsa_backend.py 3380～3556 行。它根据配置选择 flashmla_sparse、flashinfer_sparse_mla、flashmla_kv、tilelang、fa3 或 aiter。3461～3478 行用 fused top-k/page table 生成稀疏页索引，避免全 KV 扫描。flashmla_kv 路径 3691～3750 行使用 FP8 KV、page size 64 和预先准备的 FlashMLA metadata。

### MegaMoE

文件是 python/sglang/srt/layers/moe/mega_moe.py。

- 90～129 行按 EP group、token 上限和模型形状缓存 symmetric buffer。
- 199～234 行在 graph capture 中可把 shared experts 放到 alt_stream，同时执行 routed MegaMoE，然后用 stream wait 合并。
- 237～389 行完成 router/top-k、pre-dispatch 和 MUSA MATE fp8_fp8_mega_moe。MUSA 权重要求 FP8 E4M3，scale 要求 FP32。402～428 行转换为 MegaMoE layout，kernel 使用 recipe=(1,1,32)、swiglu、activation clamp 和 fast math。
- 490～503 行支持 P/D 角色专门构造权重。644～670 行在 PD decode 中释放 regular MoE 权重，prefill 侧可释放 interleaved 权重。

### 并行策略

python/sglang/srt/server_args.py 999～1162 行定义 TP、PP、DP、attention CP、DCP 和 DP-attention。

- DP attention 用 DP 复制 attention、TP 处理 FFN，适合并发请求。
- DCP 通过 dcp-comm-backend a2a 交换输出和 LSE，减少 all-gather/reduce-scatter。dcp-replicate-q-proj 用少量重复 Q projection 换掉每层 Q all-gather。
- EP 配合 moe-a2a-backend 选择 DeepEP、Mooncake、NIXL、Mori、MegaMoE 等 dispatch/combine 路径。deepep-mode auto 在 decode 选 low-latency，在 prefill 选 normal。
- enable-two-batch-overlap 在 2871～2883 行开启 TBO。

P/D 约束在 python/sglang/srt/arg_groups/pd_disaggregation_hook.py 30～98 行。decode DCP 要求 Mooncake/NIXL，decode 默认使用 chunk cache，radix cache 需要显式开启并受额外限制。P/D cache layout、transfer backend 和 decode 并行组必须一起设计。

## 与当前 xLLM 对应关系

当前 xLLM 是 sj7 Prefill CP8/EP8，sj8 Decode TP8/CP1，routed experts 在 8 卡分片。Decode graph 已把热态 TPOT 从 eager 57.51 ms 降到复用主机槽位副本后的 39.32 ms，证据在 PD_RUN.md 和 pd_logs/decode-opt/comparison.json。

| 主题 | 当前 xLLM | SGLang 做法 | 建议 |
|---|---|---|---|
| Graph | MUSA graph 已覆盖 GLM KDA/MLA，graph 内仍约 35 ms GPU 工作 | bucket、stable buffer、replay stream fence、full/breakable graph | 保留现有 graph，补齐统一 graph key、buffer registry、stream event 和动态 metadata |
| 每步同步 | 启动脚本显式 enable_schedule_overlap=false，VLM NO_SYNC 路径只有调度 overlap 开启才使用 | replay 只做 stream-local wait | 先对 Decode 做 enable_schedule_overlap A/B，验证 KDA state、输出和 PD transfer 生命周期 |
| Attention | 已有 MUSA KDA/MLA/MATE kernel，trace 仍有 prepare/index/norm 小 kernel | DSA decode 选择 sparse MLA、FlashMLA、FlashInfer、TileLang，metadata 可预计算 | 优先融合 token=1 的 prepare、state update 和 projection |
| MoE | MegaMoE stage1/stage2 和 TP reduce 是主要设备时间 | MATE MegaMoE、FP8 layout、pre-dispatch、shared-expert alt stream、PD 权重裁剪 | 先对齐 stage1/stage2/scale/copy，再试 shared-expert overlap 和 token=1 kernel |
| 并行 | D 是 TP8、CP1、逻辑 EP8 | DCP、DP attention、EP/A2A 多种布局 | 单并发先比较 TP4/TP8 attention/shared-expert，保持 routed EP8。DCP/DP 放到并发实验 |
| 跨请求流水 | scheduler overlap 关闭 | TBO 两批切分和跨层交错 | C=8/C=16 稳定后再移植，C=1 先不引入 |
| 有效 token 数 | 未启用 MTP/speculative | EAGLE/MTP/DFlash 配合 graph 和状态回滚 | 作为 10～20 ms/token 的后续路线，先验证 KDA/MLA 状态回滚和 P/D 首 token |

## 建议实施顺序

### 阶段 A：稳定 graph replay

1. 按 batch size 和 decode token width 生成 graph key，预捕获 1、2、4、8、16 等 bucket。
2. 将 KDA state slots、MLA page table、sequence length、out cache location 和 output rows 放入固定地址 registry。host mirror 只做异步 H2D，不每步 item()/D2H。
3. 将输入更新、graph replay 和输出消费拆成 stream-local 阶段，用 event wait 表达依赖。全局 synchronize 只保留给 graph capture、显式 profiling 和诊断。
4. profile_bench.sh 同时记录 graph launch、GPU completion、短输出同步和 8 个 rank collective。

C=1、20k/16、热态 8 次请求必须与 eager 的 token ID 对齐，KDA state slot 切换测试通过，8 个 Decode rank 无 collective 错误。

### 阶段 B：设备关键路径和并行

1. 用 eager trace 标出每层 MegaMoE stage1/stage2、TP reduce、mHC、KDA prepare/recurrence 和 MLA decode。
2. 对 MegaMoE 比较 FP8 layout、pre-dispatch、中间 tensor 写回次数和 TP reduce 大小，优先做 token=1 固定形状 kernel。
3. 融合 KDA/MLA 的 norm、gate、conv/state update、index select 和 output projection，并分别验证 eager、capture、replay。
4. 在 MUSA 上试 shared-expert alt stream，使用 event 代替 host wait，确认真实 GPU overlap。
5. 保持 routed EP8，比较 attention/shared-expert TP8 和 TP4 的单请求延迟、8 rank collective、显存和输出。

### 阶段 C：并发和有效 token

1. C=8/C=16 稳定后移植 TBO 的两个子 batch metadata 和 layer operation schedule，先支持 decode-only、固定宽度。
2. 验证 DP attention 或 DCP。DCP 先做 KV owner、LSE reduction、P/D manifest 和 page mapping 的离线单测。
3. 最后接 MTP/EAGLE/DFlash，检查 accepted token、KDA state、MLA KV、reject rollback、P/D 首 token 和 graph replay。

## 统一验证和 trace

固定同一份 20,000 输入、16 输出、纯文本、并发 1，至少两次预热后测 8 次。并行/TBO 另测 C=8、C=16。记录：

- 客户端 TTFT、TPOT、端到端时间和 D 侧 KV restore。
- P/D 相同 request ID 的 token 数、token ID、state slot 和 transfer regions。
- 8 个 Decode rank 的 GPU trace。graph trace 用于确认 graph launch 和 stream fence，关键 kernel 从 eager trace 统计。
- graph capture 首次耗时、steady-state GPU elapsed、CPU launch gap、collective 时间、峰值显存和服务错误。
- KDA 状态切换、非法 slot、边界 sequence length、graph replay 与 eager 的 token/state 对比。

当前 xLLM trace 显示 graph launch 只有亚毫秒量级，graph 后 GPU 计算约 35 ms/步，另有约 0.05～0.21 ms 输出消费同步。继续删 synchronize 不会单独得到 10～20 ms。这个目标需要减少 MoE、通信和小 kernel 的设备关键路径，或用 MTP 将一次验证摊到多个接受 token。

## 源码索引

- decode graph runner：python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py 282～318、355～410、541～621、845～1062、1088～1350
- MUSA full graph：python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py 77～208
- breakable graph：python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py 56～249
- TBO：python/sglang/srt/batch_overlap/two_batch_overlap.py 338～471、506～845、872～1057
- GLM5 model：python/sglang/srt/models/glm5_next.py 1083～1186、1313～1438
- KDA/Radix attention：python/sglang/srt/layers/radix_linear_attention.py 78～160
- DSA decode：python/sglang/srt/layers/attention/dsa_backend.py 3380～3556、3691～3750
- MUSA MegaMoE：python/sglang/srt/layers/moe/mega_moe.py 199～389、402～428、490～503、644～692
- TP/DP/DCP/EP/TBO flags：python/sglang/srt/server_args.py 999～1162、2281～2325、2871～2883
- P/D constraints：python/sglang/srt/arg_groups/pd_disaggregation_hook.py 30～98

