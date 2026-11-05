<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
     Licensed under the Apache License, Version 2.0. -->

# GLM-5.3-Flash sparse attention：xLLM MUSA + MATE + muDNN 实现方案

核对日期：2026-11-05。本文是接入方案及有限算子验证，未实现新的 xLLM 模型或算子。范围是文本 decoder 的 DSA/IndexPool 层；目录名中的 VL 不表示本文已验证视觉编码器、KDA、MoE 或整个 VL 模型。



> 实现核对补充：安装版 MATE 的 paged Indexer cache 虽为逻辑 `[P,64,1,132]`，物理上每页是 `64*128` 个FP8 payload字节，后接64个FP32 scale，并非逐pool的inline132-byte record。完整层现已接入，当前接口、权重/cache契约与验证结果见 [musa_sparse_attention_integration.md](musa_sparse_attention_integration.md)。本文其余部分保留为最初静态调研记录。

## 1. 结论

建议保留 **NoPE latent MLA + 独立 IndexPool**，由 **MATE 承担 sparse attention 和 FP8 index scoring，muDNN/现有 xLLM 封装承担投影、归一化、BMM、Top-K，MUSA 设备 kernel 承担 GLM 专用压池、tail、Hadamard、量化和索引转换**。

最关键的发现是 MATE 已有专用 `sparse_mla_nope_decode`，支持 **Q512 + 每 token 528-byte inline FP8 cache**；不必把 GLM 的 NoPE latent 强行改成 DeepSeek-V4 的 448+64 cache。MATE 的 BF16 sparse prefill 也有 Q/K/V latent 均为512维的入口。两者都还需要 xLLM C++ 的离线导出、workspace/metadata 和调用封装，Python API 的存在不等于当前 xLLM 已经接通。

当前 xLLM MUSA **没有本次检索可确认的原生 GLM5-Next sparse 层注册/调用链**。现有 GLM 配置 loader、Python 实现、MLU 原生实现可作为模型/状态语义参考；现有 MUSA DeepSeek-V4 实现可以复用 FFI、stream、workspace 和调度方式，但其专用 attention/cache/indexer 参数不能原样套用。

推荐先实现 BF16 latent sparse 路径作为数值基线，再接专用 FP8 NoPE decode 与 FP8 indexer；短上下文 dense MHA、prefill 融合、图执行和跨设备扩展分阶段加入。不要首先追求与 vLLM 相同的 launch 数。

## 2. 参考与当前环境

| 项目 | 本次核对结果 |
| --- | --- |
| 用户 vLLM 研究 | [VLLM_GLM_5_3_FLASH_SPARSE_ATTENTION_ZH.md](/data/nfs_shared/albert/Projects/scripts/xllm/GLM-5.3-Flash/VLLM_GLM_5_3_FLASH_SPARSE_ATTENTION_ZH.md) |
| xLLM checkout | `/data/albert/Projects/xllm-musa`，`4b49ec74d5d37f024aa9bb89e0c2fd1dc870bb6d` |
| vLLM checkout | `/data/albert/Projects/vllm`，`ebac032b92cc55c2232deced9217cd5122648ce8`，与用户文档一致 |
| MATE checkout | `/data/albert/Projects/mate`，`9d3260d633597f08fc9d5d4dc73344872241c34d`，检查时工作区干净 |
| 已安装 MATE | `0.2.8.post1`，`/usr/local/lib/python3.10/dist-packages/mate`；checkout JIT 缓存版本为 `0.3.0`，与安装包不等同 |
| Torch / TorchMUSA | 均为 `2.9.1` |
| TVM-FFI | distribution 名为 `apache-tvm-ffi`，`0.1.11.post1+musa.1` |
| 设备 | `mthreads-gmi` 可见8张 MTT S5000，驱动 `3.3.8-server`；测试选择 MP31 路径 |
| 已有构建 | `build/cmake.linux-x86_64-cpython-310`，`USE_MUSA=ON`、Release；二进制 `build/lib.linux-x86_64-cpython-310/xllm/xllm` |
| 动态库 | `ldd` 解析到 `/workspace/mudnn_3.4.0/lib/libmudnn*.so.3` 和 `tvm_ffi/lib/libtvm_ffi.so`，未见 `not found` |
| muDNN 头文件 | `/usr/local/musa/include/mudnncxx/`；`mudnn_version.h` 宏为3.1.0，`/workspace/mudnn_3.4.0/include` 的对应宏也为3.1.0；目录名不能作为 ABI 版本证明 |
| 当前 FFI 模块目录 | `FLASHINFER_OPS_PATH=/data/feihu/mate027_cache/mp31/cached_ops_jd`，检查时目录不存在，尚不能据此认为 C++ 可加载目标模块 |

MATE 在线文档地址为 <https://mate-docs.mthreads.com/0.2.8/index.html>，本次访问失败；下面关于能力的判断来自本地源码、已安装包和测试，未引用在线页面内容。已读技能随附 `mate_api.md`、`muDNN_api.md` 以及 dtype/status 枚举文档；实际接口同时核对本机头文件。

安装包与 checkout 的 `flashmla.py`、sparse prefill dispatcher、`glm_nope.py` 有差异：安装包 prefill 使用 `sparse_mla_model1_fwd_pipelined.py`，checkout 使用 `mp31/sparse_mla_v4_fwd_generic.py` 等架构分发；`glm_nope.py` 的 scheduled kernel import 路径也不同。checkout 新的 `ops/indexer/impls/mutlass/indexer_mp31.py` 在安装包中不存在，安装包提供的是 `deep_gemm.py` 的 FP8 scoring API。接入时必须固定其中一套源、头文件、生成器和 `.so`，不能混用名称/ABI。

## 3. 必须保持的模型语义

本方案按用户研究中的 Flash 配置：hidden=4096，主 heads=64，Q latent=1536，KV latent=512，原始 QK/V head dim=256，RoPE dim=0；index heads=32，index dim=128，pool rate=4，token Top-K=2048。最终以实际 checkpoint 的 `text_config` 为准；当前 loader 的 `index_kpool` 默认是1、compress 默认 false，不能依赖缺省值启用 rate=4。

令 `H=64/TP`，`C_j:[512]`，`UK_h/UV_h:[512,256]`：

```text
q_raw = q_a_proj(x)                  [T,1536]，RMSNorm
C     = kv_a_proj(x)                 [T,512]，RMSNorm
Q     = q_b_proj(q_raw)              [T,H,256]
Qlat_h = Q_h @ UK_h.T                [T,H,512]
score_jh = Qlat_h @ C_j.T / 16
Z_h   = softmax(score over selected tokens) @ C   [T,H,512]
O_h   = Z_h @ UV_h                   [T,H,256]
output = o_proj(flatten(O))          [T,4096]
```

主 scale 必须显式传 **0.0625**，不要使用512维 API 的默认 `1/sqrt(512)`。主干和 indexer 都不运行 RoPE。UK/UV 从 `kv_b_proj` 按 head 拆分，TP 仅切主 heads 及相关权重，indexer 投影按 vLLM 语义复制；最终输出投影遵守现有 RowParallelLinear 通信约定。量化权重必须先按其 scale/layout 正确派生权重，不能直接 reinterpret 量化存储。

IndexPool 的独立通路为：

```text
q_raw_norm -> wq_b -> IQ [T,32,128]
x -> wk -> LayerNorm -> IK [T,128]
x -> weights_proj(FP32) -> W [T,32]
x -> compress_gate -> G [T,128]
pool_K[d] = sum_j softmax_j(G[j,d]+APE[j,d]) * IK[j,d]，j=0..3
IQ / pool_K -> normalized Hadamard128 -> FP8 + scale
pool_score[p] = sum_h W[h] * ReLU(dot(IQ[h],pool_K[p]))
Top512 pools -> 展开2048个原始 token -> 追加0..3个未成池 tail token
```

index score 的缩放是 `128^-0.5 * 32^-0.5 = 1/64`，再按接口约定折入 Q/K 量化 scale，不能套用主 scale。ReLU 必须在每 head 点积之后、head 加权求和之前。保留 head weight 的 FP32 计算；pool 后 BF16 舍入、Hadamard 后 BF16 舍入及 Q 侧独立舍入顺序按用户文档对应 vLLM kernel 核对，不能因 Hadamard 在实数域保持内积就删除它。

完整 pool 只在该 query 的因果可见范围内形成；prefill 一次压完所有输入后仍要逐 query 限制 pool 可见性，避免把未来 token 压入可见 pool。主 MLA cache 始终每原始 token 一条，不压缩为每 pool 一条。

## 4. 仓库内参考及复用边界

| 参考 | 关键位置/接口 | 可复用与 MUSA 差异 |
| --- | --- | --- |
| 公共配置 | [glm5_next.h](../../xllm/models/llm/glm5_next.h)，`glm5_next_args::load_model_args` | 已加载 NoPE、IndexPool、层 schedule、VL 配置；不等于 native MUSA 模型已注册 |
| MLU 原生模型 | [mlu/glm5_next.h](../../xllm/models/llm/mlu/glm5_next.h)，`Glm5NextModelImpl`/注册；`models/models.h` 的 `USE_MLU` 分支 | 模块/权重/Top-K 共享关系参考；MUSA 分支目前没有包含这个模型 |
| MLU KPool | [glm5_next_kpool_indexer.cpp](../../xllm/core/layers/mlu/glm5_next/glm5_next_kpool_indexer.cpp)，`Glm5NextKPoolIndexerImpl`、`update_cache`、`select_pools` | 已有投影、tail、分页 scoring、Top-K、展开、长度接口；设备检查固定 MLU，不能直接调用 |
| MLU NoPE MLA | [deepseek_v2_attention.cpp](../../xllm/core/layers/mlu/deepseek_v2_attention.cpp)，Q 吸收、`resolve_dsa_topk_state`、`build_mla_attention_metadata`、`project_output` | 显式跳过不支持 NoPE 的 fused MLA；可参考 UK/UV BMM 和 sparse metadata 生命周期；MLU block table 语义要适配 MATE |
| MLU kernel | `core/kernels/mlu/triton_kernel/glm5_next_kpool{,_select,_expand}.py` | pooling/tail/expand 的仓库内对照；其 Triton target、舍入/非FP8路径不能直接视为 vLLM FP8 等价 |
| Python 模型 | [glm5_next.py](../../xllm/python/models/glm5_next.py)，`Glm5NextIndexer`/`Glm5NextKdaAttention`；`glm5_next_kpool.py` | 模型权重名、schedule、tail、Top-K 共享及测试参考；含 NPU 专用调用，不能靠切 device 宣称支持 MUSA |
| 公共 cache | `core/framework/kv_cache/{kv_cache_shape,indexed_kv_cache_impl,linear_state_restore}.*` | 已有 index cache、KPOOL_TAIL、状态槽及恢复入口；需增加/核对 MUSA 物理 layout 分支，保持 prefix-cache/请求释放/槽复用语义 |
| MUSA DSV4 | [deepseek_v4_dsa_attention.cu](../../xllm/core/kernels/musa/deepseek_v4_dsa_attention.cu)、`layers/musa/deepseek_v4/` | 复用 device metadata、预分配、FFI 调用模式；DSV4 compressed/SWA 两 cache、固定512+128候选、特定 scale 不适用 GLM |
| MUSA 通用设施 | [musa_tvmffi_stream.h](../../xllm/core/kernels/musa/musa_tvmffi_stream.h)，`get_function`、`to_ffi_tensor`、`TvmffiStreamGuard`；`attention_runner.*` | 复用 module/function 缓存、DLPack、stream 依赖及 piecewise runner；不新建 Python 在线执行器 |

其他 backend 检索覆盖 `models/`、`core/layers/`、`core/kernels/` 的 GLM5、KPool、MLA、indexer 以及 CMake/test 入口；同模型最直接的 native 参考是 MLU，另有 Python/NPU 路径。没有把 CUDA/DCU 的通用 MLA 当作完整 GLM IndexPool 实现。

MLU 测试参考包括 `tests/core/layers/mlu/glm5_next_kpool_indexer_test.cpp`、`glm5_next_decoder_layer_test.cpp` 和 `tests/models/glm5_next_mlu_test.cpp`；它们的存在仅说明参考用例可迁移，未在 MUSA 运行。

## 5. MATE 的实际可用入口

### 5.1 BF16 sparse prefill：最适合首版正确性基线

调用链：`mate.flashmla.flash_mla_sparse_fwd` → `flashmla_sparse_prefill` → `sparse_mla_prefill_fwd` → MP31 `sparse_mla_v4_fwd_generic.py::sparse_mla_fwd_interface_model1`（或安装包对应 MODEL1 factory）。

输入约定：Q `[T,H,512]` BF16；KV `[N,1,512]` BF16；indices `[T,1,2176]` int32；可选 `topk_length:[T]` int32；输出 `[T,H,512]`。候选宽度须64对齐，2176也满足 vLLM 的128对齐。GLM 无 attention sink，传空。每行选择独立，不能用 V4 seq-pack 中“取最后一行候选、多个 query 共享”的优化替代 GLM IndexPool。

对分页 BF16 主 cache，可以直接 flatten 为物理 slot 空间 `[pages*page_size,1,512]`，把每行逻辑 token 转成物理 slot；前提是实际 layout/stride 与接口匹配。这样可避免每 query gather 一份 KV。长 prefill 对 query 分 chunk、复用有界临时区；不要物化 `[T,H,N,512]`。

这个 BF16 sparse 接口在算法上也可处理少量 query 的 decode，适合作为预先选定的首版路径；性能尚未验证，不能当作高效 paged decode 承诺。FP8 cache prefill 若要走它，需要设备上按 chunk 解量化/gather，或者另行验证专用 FP8 sparse prefill。

### 5.2 FP8 sparse decode：使用 GLM NoPE 专用入口

源码：[glm_nope.py](/data/albert/Projects/mate/mate/sparse_mla/glm_nope.py)，`sparse_mla_nope_decode`；cache writer 为 `tilelang/glm_nope_cache.py::write_glm_nope_cache`。

```text
Q             [B,Sq,H,512] BF16
main cache    [P,page_size,1,528] uint8（紧凑格式）
record        512 bytes E4M3 + 4个FP32 scale（每128维一组）
indices       [B,Sq,1,2176] int32，物理 slot，非法位置 -1
topk_length   [B] int32（接口约定），单步 decode 首版 Sq=1
output        [B,Sq,H,512] BF16；LSE [B,H,Sq]
scale         显式0.0625
```

也接受656-byte legacy padded row，但有效载荷仍是前528字节，行 stride 必须保持656。writer 使用 `scale=max(max(abs(group))/448,1e-4)`，再写 E4M3；cache 量化误差需要独立衡量。

调用链为 `sparse_mla_nope_decode` → `get_mla_metadata`（`mla_ops.get_mla_decoding_metadata`）→ `tilelang_flashmla_interface`，checkout 的设备实现位于 `mp31/sparse_mla_v32_decode_fwd_scheduled.py`，以 tail_dim=0 处理 NoPE。这是扩展的 V3.2 scheduled 算法，不是使用 V4 448+64 cache。

`topk_length` 是**候选有效前缀的结束位置**，其中可以有 -1 空洞；不是有效 token 数。例如第2050列有一个有效 tail，中间有空洞时，长度应覆盖到2051，不能填 `count_nonzero`。也可先在设备上稳定紧凑化，再用紧凑长度。2176列剩余 padding 必须初始化为 -1。

不要调用 `flash_mla_with_kvcache(..., indices=...)` 后只把 Q 改为512：其通用 dispatcher 会按512维选 MODEL1 的584/其他专用 record，不能解释 GLM inline528。也不要复用 xLLM DSV4 的 `kCompiledSoftmaxScale=0.044194173824159216`、`kTopk=512`、`kExtraTopk=128` 专用模块。

### 5.3 FP8 index scoring：已有匹配原语，但压池需自己接

checkout `ops/indexer/op.py` 的 `Indexer`/`PagedIndexer` → MP31 MUTLASS implementation → `fp8_mqa_logits`/`fp8_paged_mqa_logits` FFI；安装包可从 `mate.deep_gemm` 调用这两个函数。

MP31 paged 候选支持 index H=32/64、D=32/64/128、next_n=1/2/4、index page size=64。GLM 的 H32/D128/普通 decode next_n=1 匹配。具体 fused index cache 是 **`[P,64,1,132]` uint8**：128个 E4M3 + 一个 FP32 scale，和主 cache 的528-byte record完全不同。Q 为 FP8 `[B,1,32,128]`，weights 为 FP32 `[B,32]`；page table、context lens、scheduler metadata 均是 int32 device tensor。

非分页 prefill 使用 FP8 Q `[T,32,128]`、FP8 pooled K `[Np,128]`、FP32 K scale `[Np]`、FP32 weights 和每 query 的 KV 起止边界，支持 query chunk。已核对模板和 `mp31_fp8_mqa_logits.hpp` 中的逐 head ReLU/加权归约。Q scale 需要按相应 API 折入 head weights；不能重复或漏乘。

pool page table 和长度按**pool单位**构建：`floor((position+1)/4)` 是完整可见 pool 数；不得把原始 token context length 直接交给 index scoring。主 cache page size与index page size分开记录，token页到pool页的映射必须明确。安装包/checkout 的 paged FFI 返回 output layout/padding 需要按实际签名适配。

## 6. 算子库分工与降级决策

| 阶段 | 首选方案 | 选择理由与需要补充的工作 |
| --- | --- | --- |
| Q/KV、index投影、o_proj | xLLM 既有 Linear/ParallelLinear；BF16走现有 muDNN MatMul，量化依 checkpoint 用既有 MUSA 分支 | 复用 weight loader、TP和量化约定；不保证所有 checkpoint 量化格式都可用 |
| Q/KV RMSNorm、index K LayerNorm | 现有 `RMSNorm`/`fused_layernorm` wrapper，核实 MUSA mode；缺口接 muDNN RMSNorm/LayerNorm | index K 是带均值的 LayerNorm，不得替换为 RMSNorm；FP32计算/舍入要对齐 |
| rate4 pool、tail ring | 新增 GLM MUSA device kernel | 所查 MATE sparse/indexer API未提供完整 GLM 压池和tail状态机；muDNN softmax/reduce可作分解参考，融合更适合正式路径 |
| Hadamard128、FP8写index cache、Q量化 | MUSA融合kernel，先对照vLLM和MLU语义 | 不冒充已有稳定MATE KPool API；明确rounding/scale布局 |
| index scoring | MATE FP8 Indexer/PagedIndexer | 已有匹配H32/D128的ReLU多头加权原语，优先于BMM分解 |
| Top512 pools | muDNN `TopK`，或已验证落到设备的 TorchMUSA topk | MATE `hash_topk`/`sparse_topk_select` 名称不等于该行分数精确Top-K契约；先用通用精确TopK，再优化 |
| 展开、追加tail、逻辑到物理slot | MUSA device kernel | 可把展开和映射融合；同一物理token不可重复加入，以免softmax重复计权 |
| Q吸收、V恢复 | muDNN `BatchMatMul`/已有TorchMUSA bmm封装 | MATE已有GEMM/BMM模块可评估，但未确认适合该stride/输出契约的稳定GLM融合入口；首版采用仓库BMM方式 |
| 主sparse attention | MATE BF16 sparse prefill / GLM NoPE FP8 decode | 直接表达选定latent上的QK、softmax、AV；显式主scale |

Attention 决策顺序：**MATE → muDNN → muBLAS**。本模型主 sparse 原语已找到 MATE 候选，因此没有理由首先改写 GEMM+softmax。若固定版本的 MATE 在目标 shape/ABI/设备验证后确实不可用，第二方案才是设备 gather 选定 latent，再评估 muDNN `ScaledDotProductAttention::RunFlash/RunFlashVarlen/RunMath`。

核对的 muDNN SDPA 接口接受 Q/K/V/mask/cu_seqlens，没有任意 sparse physical slot list 参数，不能直接读 paged sparse cache。需分 query chunk gather `[Qchunk,Ksel,512]`、正确处理 padding/空行、`scale=1/16`、dropout=0；D512和对应mask/layout的实际支持仍需设备验证。`RunMath` 还需 `GetMathForwardWorkspaceSize`；Flash通过 `MemoryMaintainer` 管理临时区，不应编造不存在的 forward workspace 查询函数。

muDNN TopK核对到 `SetK/SetDim/SetLargest/SetSorted/GetWorkspaceSize/Run`；dtype、索引输出宽度和并列分数顺序仍需验证。对候选少于512的行先屏蔽不可见项，并在TopK后再次依据有效性写 -1；不能让补齐项变成token0。

只有上述 MATE 和 muDNN 方案均有具体失败证据时，才评估 muBLAS GEMM/BMM + 设备 softmax/mask/AV。当前没有进行这一步，也没有宣称 muBLAS 已提供完整 attention。

## 7. 执行、cache 和 C++ 接入建议

普通长 decode 的主要依赖：主投影/norm → 主cache写入；index投影 → pool/tail更新 → index scoring → Top512 → 展开/slot映射；主Q吸收 → MATE sparse MLA → V恢复/o_proj。投影、归一化、cache写入、metadata、copy还有额外launch；这里不提供未经profile的“整层kernel数”。

prefill必须处理本chunk内pool、上一chunk遗留0～3个tail，以及每个query自己的pool可见边界。短context全部可见位置不超过2048时，可显式选择全因果token，跳过scoring/TopK；主attention仍可用同一个latent sparse基线。dense MHA是后续性能选项，需要显式展开每head K/V并验证MATE接口，不是首版前提。

tail状态按**请求/状态槽/层**隔离，跨chunk拼池，跨请求释放清零或有效位失效；图padding行不写cache；shared-index层按模型schedule复用完整选择结果。初版只开放普通TP、单步decode、BF16主路径，FP8在独立验证后开放。DCP/PCP、MTP/spec、多query可变前缀、prefix复用和graph replay要逐项验证后开启，不允许根据shape相同自动扩大支持。

建议增加以下模块，名字为建议而非当前已实现符号：

| 位置 | 工作内容 |
| --- | --- |
| `core/layers/musa/glm5_next/` | KPoolIndexer和NoPE sparse MLA模块；复用公共Linear、RMSNorm、KVCache、metadata |
| `core/kernels/musa/` | GLM pool/tail/Hadamard/quant/expand/slot-mapping kernel，MATE NoPE attention和indexer C++ wrapper |
| `core/kernels/musa/CMakeLists.txt`、`core/layers/musa/CMakeLists.txt` | 加入源文件及已有TVM-FFI依赖，不在在线进程中触发Python JIT |
| `core/framework/kv_cache/` 相关layout构建 | 明确BF16主cache、528-byte FP8主cache、132-byte indexcache、tail及slot管理；核对分配、恢复、序列结束 |
| `models/llm/musa/`、`models/models.h` 的MUSA分支 | 仅在整模型模块齐备后增加native模型注册；已有args/VL processor注册不能替代模型实现 |
| 本目录编译工具 | 参照已有 `compile_musa_kda.py`，新增离线导出主attention和indexer所需模块、ABI manifest与加载预检 |

MATE C++ 接法：把选定TileLang factory离线导出为TVM-FFI `main`模块；把metadata/scoring的公开FFI函数一起预编译。按实际factory准备output、LSE、partial output、split/调度metadata，不只是转发Python wrapper形参。复用 `get_function`/`to_ffi_tensor`/`TvmffiStreamGuard`，保持device pointer和stream依赖；kernel结果仍在设备端。校验所有status，不以异常后切换实现掩盖错误。

在eager中guard可能涉及独立FFI stream，按现有guard生命周期保证前后依赖和workspace存活；capture时预热/离线加载所有模块，固定buffer容量，device metadata每步更新，workspace至少活到replay完成。首版先eager，随后接现有piecewise runner；不能把DSV4线程局部workspace按shape缓存的行为未经审核直接搬来。

## 8. 验证记录与下一步验收

静态核对已完成：仓库/版本、MLU模型与KPool调用方、MUSA CMake/FFI/cache、MATE源码与安装差异、muDNN头文件和动态链接、设备可见性。没有修改C++，没有执行xLLM增量构建或启动整模型请求；现有二进制的存在不能证明新增方案已编入。

本次执行的MATE算子探测及日志见本目录 `sparse_attention_validation/`。测试结果在文末附录记录。测试基于MATE自己的随机输入和参考，不是实际checkpoint整层比较。

后续验收至少覆盖：

1. pool/tail：长度0/1/3/4/5，页边界，跨chunk形成pool，请求槽复用，多序列，padding行和清理；与vLLM中间结果逐项比较。
2. indexer：FP32 head weight、BF16舍入、Hadamard/FP8字节与scale、逐pool分数、Top512集合/并列分数、短context和空行；FP8不一致可能改变选择，不能只比较最终误差。
3. attention：先固定相同indices，对比显式256维K/V参考与latent吸收实现，scale=1/16；覆盖H=8/16/32/64、2176padding、2051前缀中间有洞、全-1、重复/越界防护、多请求分页。
4. FP8：分别比较FP8 cache解量化后的参考与算子输出，以及原BF16模型到FP8的量化误差；528/656行stride、任意非2幂FP32 scale、主/index两种记录不可混淆。
5. C++：离线模块导出后检查文件/函数名/ABI manifest，`ldd`及FFI加载，再运行有真实MUSA activity的数值测试；Python成功不替代C++验证。
6. 真实模型：包含新代码的二进制，先短请求，再>2048历史的长请求，继续decode并复用cache，再发第二请求检查状态隔离；记录后端选择、响应、日志及设备profile。VL整模型另验图像token路径。
7. 性能：分别测prefill/decode的pool、scoring、TopK、slot-map、Q/V BMM和主attention耗时、峰值临时显存、实际launch数。百万上下文避免一次物化所有query×pool logits；按chunk限制workspace。

本次没有给出性能提升数字；尚需确定checkpoint、目标TP/量化配置及可加载的离线MATE模块后，才能给出完整模型的吞吐和延迟结论。

### 附录：本次实际执行的算子测试

测试命令在xLLM仓库根目录运行。源文件来自同一个MATE checkout，分别以 `PYTHONPATH` 选择checkout实现和已安装实现；不修改系统安装。

```bash
PYTHONPATH=/data/albert/Projects/mate python -m pytest -q \
  '/data/albert/Projects/mate/tests/test_sparse_mla.py::test_sparse_mla_glm_nope[528-glm53_h8_k2176]' \
  '/data/albert/Projects/mate/tests/test_sparse_mla.py::test_sparse_mla_glm_nope[656-glm53_h8_k2176]' \
  '/data/albert/Projects/mate/tests/test_sparse_mla.py::test_sparse_mla_prefill_independent_kpool[2176-8]'

env -u PYTHONPATH python -m pytest -q \
  '/data/albert/Projects/mate/tests/test_sparse_mla.py::test_sparse_mla_glm_nope[528-glm53_h8_k2176]' \
  '/data/albert/Projects/mate/tests/test_sparse_mla.py::test_sparse_mla_glm_nope[656-glm53_h8_k2176]' \
  '/data/albert/Projects/mate/tests/test_sparse_mla.py::test_sparse_mla_prefill_independent_kpool[2176-8]'
```

| 实现 | 结果 | 证据/范围 |
| --- | --- | --- |
| checkout MATE | 1 passed、2 failed，25.15秒 | [source日志](sparse_attention_validation/mate_source_probe.log)；BF16 prefill通过；两个decode都在 `get_mla_metadata → mla_ops` 编译失败，最早编译错误包含 `mutlass/fast_math.h file not found`，另一个编译单元缺 `mutlass/device_kernel.h`。未得到decode数值结果 |
| 已安装 `0.2.8.post1` | 3 passed，33.83秒 | [installed日志](sparse_attention_validation/mate_installed_probe.log)；H8、2176宽候选、528/656-byte cache decode及独立query的BF16 prefill通过设备数值比较 |

测试包含前缀结束位置2051、其间 -1空洞、追加3个tail候选及全无效行；prefill专门设置不同query选择不同token，可排除共享最后一行候选的错误复用。参考输出来自MATE的PyTorch sparse attention，decode参考先解量化同一个FP8 cache。既有断言为output `rtol=0.02, atol=0.01`，LSE `rtol=0.002, atol=0.02`，prefill max-logits同LSE阈值。测试通过表示这些用例通过既定容差，并不表示逐bit相同或FP8相对原BF16模型无损。

所选decode fixture实际为B=4、Sq=3、H=8、KV长度2112、候选宽2176，并额外启用attention sink；prefill为8个query、8条KV、H=8、候选宽2176。这比单query覆盖更广，但仍未验证GLM模型的多query pool可见边界和xLLM调度适配，因此首版建议继续从Sq=1开始。

安装版运行有TileLang弃用警告及 `T.sched_boundary is disabled: MTCC 5.3.0 or newer is required` 警告，但测试完成且数值断言通过。MUSA设备计算由这些测试实际触发；本次没有单独采集设备利用率曲线或kernel timeline。

尚未验证H16/H32/H64、完整IndexPool压缩/scoring/TopK、C++导出ABI、图执行及真实模型；源码checkout缺头文件问题未在本次研究中修复。后续优先固定能复现上述安装版通过结果的依赖，再导出GLM专用FFI模块。不要把checkout失败误判为NoPE算法不支持，也不要把Python测试成功写成xLLM已支持。
