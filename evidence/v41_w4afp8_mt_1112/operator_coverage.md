# C++ 路线算子覆盖、有效选路及交付版差异

更新时间：2026-11-12，StageI正式性能与前后trace验收完成。本文覆盖文本 C1/eager、P/D 各 TP8 的当前实现；完整数据见 `optimization_stage_i/README.md`。

## 实现身份与判断标准

分支为共同仓库 `fay85/xllm-musa` 的 `xiaopangV41FlashCppFromSglangtoXllm`，直接从 1103 的 `0cfd82700e164e8aa7951064b652925dd6b24d18` 建立。参考家明 V4 提交 `f8dd0663279e9085365ade309d9823b7b438c4e6`、京东交付镜像内 SGLang `7c97bb0e6d8b0f8fd41cd797a77b54f7162c79f6` 及官方原版 inference/model.py；交付镜像与源码清单保存在 `01_参考/`。

模型是 MooreThreads 官方转换版 `DeepSeek-V4.1-Flash-1017-W4AFP8-mt`，ModelScope commit `a41b72782c27dbd505c84c13fcefaabb1fe2f273`。专家实际为有符号 INT4 packed-I8、group128 scale。不能因为 config 中仍出现 fp4 就将其误称为原版 E2M1/group32 专家。权重格式、激活量化、累加精度和输出归约须分别核对。

当前生产前向由 `xllm/models/llm/musa/deepseek_v41_native.h` 及 `core/layers/musa/deepseek_v41/` 的 C++ 类组织。C++ 直接调用 MUSA kernel、MATE/TVM-FFI 或离线编译 AOT。Python 用于离线构建、数值参考、服务进程管理和 HTTP 测试，不承担生产模型的逐层 forward。CPU 仍承担请求、采样/Tokenizer和控制逻辑，这不等于把 Attention/MoE 数学回退到 CPU。

判定“接入”同时要求：有实际生产调用、依赖和产物固定、目标形状数值测试通过、目标 GPU 测试通过。只找到接口、仅完成编译或仅两道问答通过都不足以证明完整精度。

## 已接入的主要算子族

路径均相对于仓库；表中的测试是算子级证据，不等同于评测集精度验收。

| 模块 | 当前有效实现与文件 | 来源/接入要点 | 已验证范围 |
|---|---|---|---|
| 模型、层组织、权重加载 | `models/llm/musa/deepseek_v41_native.h`、`deepseek_v41_checkpoint.h`；`layers/musa/deepseek_v41/text_model.h`、`decoder.h` | 保留 xLLM C++ 执行/调度，真实加载 40 主层；不借用 Python 模型 forward | sj9/sj10 共16个真实进程，实际 exe、每rank40层核对；短题及2048多块输入通过 |
| 路由专家 W4A8 | `w4a8_experts.h`、`deepseek_v41_w4a8.cu`、`deepseek_v41_w4a8_gemm.cpp` | 交付 MATE S4/FP8 group128 grouped GEMM；一次加载保留 packed 专家，GPU 路由、排序/分组、SwiGLU量化、合并 | 真实权重2专家，M1/17/128/256/512，DEV及双SJ通过；最大相对RMS约0.07613% |
| MoE 路由与归约 | `moe.h`、`deepseek_v41_gate_topk.cu` | top6、sqrt-softplus、routing scale1.5、clamp10等按目标配置；EP8每rank48专家；Attention TP8；共享专家replicated只加一次 | 基础/长请求与八rank实际运行；逐项专家数值记录；不是 DeepEP all-to-all 吞吐验证 |
| 共享专家/常规线性 | `feed_forward.h`、`deepseek_v41_dense_block32.*` | FP8/block32 激活 QDQ 后 BF16 MUBLAS；共享SwiGLU融合；加载时固定权重布局，不在每token逐专家解量化 | 双SJ真实7类线性形状数值对比；特化候选更慢，未启用 |
| Mega-mHC | `hyper_connection.h`、`deepseek_v41_mhc_prenorm.*` | 使用交付 MATE m32/m64 prepare，保留 V4.1 shifted pre_mix、Sinkhorn 及子层系数传递；原生GPU post/combine | 真实Attention/FFN权重、M1/17/128/256/512共10类，双SJ通过，最大相对RMS约0.04649% |
| RMSNorm、RoPE | `rms_norm.h`、`attention_projection.h`、`deepseek_v41_attention_quant.*` | 复用已有原生GPU norm/partial-RoPE；融合 RoPE、FP4 QDQ、需要的E4M3 scale边界 | RMS96例/目标机；RoPE/量化36例逐位一致，包含小/正常/大幅值 |
| Attention主路径 | `attention.h`、`csa2_attention.h`、`deepseek_v41_sparse_attention.*` | MATE稀疏Attention，真实TP8 local H8/D512；sink、因果与SWA/压缩pick语义保留 | 12组有限/NaN无效cache组合，双SJ通过，相对RMS约0.212%～0.238%；全无效pick精确零 |
| CSA2 / Indexer | `indexer.h`、`deepseek_v41_indexer_score.*`、离线`build_indexer_aot.py` | 参考交付Triton评分，显式保留BF16 dot/product/head-sum舍入；稳定TopK、候选复用/跨层source | 评分24例；TopK144例，K1/64/256/512；实际目标K512额外计时/验证；双SJ通过 |
| C2 compressor | `compressor.h`、`deepseek_v41_pair_pool.*` | FP32 GPU投影不改；成对softmax池化及尾部状态提交各一个有序kernel；保留奇数跨块状态 | 分页池化64例、完整类8例/目标机；此前257-token奇边界4类回归 |
| Engram lookup | `engram_embedding.h`、`deepseek_v41_engram_gather.*` | 原FP8/UE8M0表GPU常驻；离线Triton AOT融合查表+解码；本地head分片再TP all-gather | 每目标机48例，包含NaN/边界/Mask；原表未转换为全量BF16 |
| Engram hash/history | `engram_hash.h`、`deepseek_v41_engram_hash.*` | int64 hash、真实目标Tokenizer映射；先读整段history，再提交最后写入，避免ring覆盖提前破坏lookback | 每目标机68例，独立CPU参考的hash/mask/history精确相同；新Tokenizer压缩词表99092 |
| Engram注入 | `engram_injection.h`、`deepseek_v41_engram_gate.*` | GPU门控、norm、投影及残差注入；与hash/lookup合用；无CPU计算替代 | 继承原生数值测试及本轮完整问答；未宣称完整评测集精度 |
| 输出低秩投影 wo_a | `attention_projection.h` 调用已有 `deepseek_v4_wo_a_m1` | 复用家明/1103已存在的单token GEMV；M1/G1/4096→1024 才选，其他形状仍用已定义BF16 einsum；QDQ顺序不改 | 真实layer0 TP8 rank0权重60例，最大相对RMS0.003175%；DEV及双SJ通过，局部约0.055→0.020ms |
| SWA单token写回 | `paged_attention.h`、`deepseek_v4_dsa_cache_ops.cu` | 单行复用已有scatter kernel，显式native MUSA current stream；多行stable latest-wins不变 | 双SJ各60项逐位回归；同场景每commit 35→1 kernel；完整两组正式复测 |
| P/D缓存与跨层复用 | `request_metadata.h`、`token_cache.h`、`paged_attention.h`、framework/kv_cache、disagg_pd_service | SWA、压缩KV/Indexer状态、Engram lookback等按mapping group交给Mooncake；共享source层及TopK按forward生命周期传递 | StageI双端16个实际exe SHA匹配；每rank40层/TP8/EP8；D每rank至少15条非空pull；两组正式性能通过 |

上述 MATE 扩展产物使用交付镜像中对应源码/头文件离线编译；服务基础镜像仍是 `xllm:1102-dev` 的 Torch/MUSA2.9.1、MATE0.2.7、TVM-FFI0.1.11.post1+musa.1。没有把整个环境静默替换成所谓“最新MATE”，也没有依靠在线 Python JIT 才能生成生产算子。具体so/mubin和构建命令、编译头文件、SHA在运行包manifest和原始构建日志中。

## 与交付 SGLang 不同、但不能混为“缺失模型结构”的部分

### Dense FP8/block32

已保留block32激活量化/反量化数值边界，GEMM载体为BF16。这与交付MUSA代码中的部分BF16/F32 GEMM策略一致，并不因为专家改成W4A8就自动获得所有Dense FP8原生矩阵指令。我们试过7个真实形状的AOT特化，均数值通过但比现有路径慢，因此没有上线。新补的wo_a是此前7形状未覆盖的一类；该类实测较快才接入。不能把“没有启用更慢候选”写成需要MATE提供缺失接口的外部卡点。

### EP通信策略

当前同一请求的token在各EP rank上都有，每rank仅计算其48个专家，合并后一次F32 all-reduce；共享专家为replicated，放在归约后相加。它能表达当前C1策略，不应再对已归约输出求和。交付SGLang的token分发、DeepEP/all-to-all、可选FP8通信属于另一套完整的数据归属策略，不能单改ep_size或通信dtype就替换。后续需在同一token归属和量化边界下做通信数值及性能回归，不是缺少模型的路由/共享专家。

### CED与尾部SWA replay

当前执行官方参考的完整40层chunk前向，具备SWA与跨层压缩KV复用。交付SGLang另有 `enable_decoder_swa_bounded_replay`：从late layer21起对extend输入只保留尾部512行，并将可见SWA起点限制到tail_start，因为更早的late KV没有写入。该逻辑位于`models/deepseek_v4.py`与`layers/attention/deepseek_v4_backend.py`，部署指南确实有打开它的命令。

这是一项调度、metadata、状态缓存共同变化的执行策略，不是一个可直接dlopen的MATE算子。官方原版model.py的Transformer.forward仍对所有40层执行完整当前chunk。当前xLLM chunk256小于512，直接按当前chunk裁尾不会省行；如要跨chunk仅replay最终尾部，需要改变延后执行/状态传递，必须先明确对长输入logits的语义影响。当前没有证据说明“不接该策略就漏算一个必需模块”，也没有证据宣称它已在我方生效。此处明确记录为未接的可选执行优化，后续有条件设计数值回归，而非甩给MATE的缺失算子。

### 更大范围融合、Graph、MTP和多模态

已接入主要算子族，不等于已穷尽SGLang所有融合变体。跨子层mHC融合、paged cache metadata/gather/scatter、通信/计算重叠仍可继续优化；此轮按要求先保留eager，基于真实trace选择一个小范围改动归因。MTP=0、文本请求，无vision输入；不能对外宣称完成全部多模态或投机解码覆盖。它们不是此次两组文本性能的隐含开启项。

## 本轮遇到的问题归属

- W4算子首次误差大：我方离线头文件/编译参数未与交付对应，按实际交付固定后数值通过；不能以最初失败断言MATE不支持。
- 稀疏Attention的无效pick引用NaN cache：交付kernel的乘零并不能消去NaN，改用已有robust descriptor的越界零填充，再做12类数值回归。补丁和原始失败均保留。
- Engram hash失败：我方Triton ASTSource signature dict顺序错误，加上cache hash不区分顺序，导致错误ABI和旧产物复用；修正离线有序签名与局部cache key后68类通过。不是MUSA整数store硬件缺陷。
- EvalScope首个暖机失败：我方给了根URL，工具自动补错API路径。改为完整`/v1/chat/completions`；失败记录保留，正式测量另建v2目录。

当前这些问题都已在我方解决，没有据此向外部索要资源。现已完成两组正式性能、双端40层/16rank真实trace、单token SWA写回优化及同条件复测；第一提交已推送，第二提交收录优化证据。H200原版1017参考与Torch/Perfetto完成，231/232我方容器正常停止；H20 GPU任务按用户要求延期。SOL暂无测算表，按用户确认记录方法/参数，不宣称已达到60%。2048固定prompt下输出不完全重复、默认无progress配置下长输入通信超时仍需后续隔离，详见新报告。
