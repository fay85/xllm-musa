# GLM-5.3-Flash / DeepSeek-V4 mHC 与 MUSA kernel 调研

调研日期：2026-11-06。源码基线：当前 checkout `8339dc1a1`（KDA 更新提交）；本地 vLLM 基线：`ebac032b92`；MATE checkout：`9d3260d633597f08fc9d5d4dc73344872241c34d`，实际安装包 `mate==0.2.8.post1`。本文是静态调研，不把算子单测或 Python import 当作完整模型支持。

## 结论

当前仓库已经有一条可复用的 MUSA mHC 基础路径，但它挂在 DeepSeek-V4 命名和模型调用链下，不能直接说明 GLM-5.3-Flash/VL 已支持：

- MATE 有 `mhc_prenorm_gemm_sqrsum`、`mhc_pre_big_fuse`、`mhc_pre`。前者计算 BF16 residual 与 FP32 `hc_fn` 的 TF32 prenorm GEMM/平方和，后者完成 RMS、sigmoid、Sinkhorn、pre/post/comb 输出和 layer-input 融合。
- MATE 没有公开的 mHC post/head API。GLM 所需的 post（`comb @ residual + post * x`）应复用 xLLM 的 MUSA `deepseek_v4_hc_post` 或抽成通用 kernel；不要为 4×4 混合强行使用 attention API。
- MUSA 已有 `deepseek_v4_hc_pre_decode`，优先走 MATE FFI（AOT `.so`），模块缺失时才走本地 MUSA fallback；已有 `deepseek_v4_hc_pre_combine`、`deepseek_v4_hc_post` 和 DeepSeek-V4 的 learned HC head。它们是最直接的设备实现和 stream/workspace 参考。
- MLU 已实现可复用的模型级边界：`MHCPre`、`MHCPost`、`PendingMHC`，以及 GLM5-Next 的“上一层 deferred post + 当前层 pre”融合。MUSA 尚没有对应的 GLM5-Next decoder、MUSA 模型注册或 VLM 组合；`models/models.h` 的 `USE_MUSA` 分支目前只包含 DeepSeek-V4、Qwen3.5/Next 等模型。
- 因此建议先把 mHC 做成 MUSA 通用层接口：MATE 负责 pre 的 GEMM/fuse，现有 MUSA kernel 负责 post/expand/contract；首版先组合执行并保证语义，再增加跨层 fused post-pre。GLM5 的 KDA、sparse MLA/IndexPool、MoE、vision 和权重加载仍是独立接入工作。

## 1. 语义和张量契约

vLLM 的 `vllm/model_executor/kernels/mhc/torch.py::mhc_pre_torch` 是清晰的参考公式。对 `r=hc_mult=4`、hidden `D=4096`：

```text
residual       [T,4,D] BF16
hc_fn          [24,4D] FP32
hc_scale       [3] FP32             # pre/post/comb
hc_base        [24] FP32

x              = residual.float().reshape(T, 4D)
mixes          = x @ hc_fn.T
rms_inv        = rsqrt(sum(x*x)/(4D) + rms_eps)
mixes          = mixes * rms_inv
pre            = sigmoid(mixes[:,:4] * scale[0] + base[:4]) + hc_pre_eps
post           = 2 * sigmoid(mixes[:,4:8] * scale[1] + base[4:8])
comb_logits    = mixes[:,8:].reshape(T,4,4) * scale[2] + base[8:].reshape(4,4)
comb           = softmax(comb_logits, -1) + hc_sinkhorn_eps
               → 列归一化；重复 row/column Sinkhorn 共 sinkhorn_repeat 次
layer_input    = (pre[...,None] * residual.float()).sum(dim=1).to(BF16)
```

post 的公式是 `output[t,j,:] = post[t,j] * x[t,:] + sum_i comb[t,i,j] * residual[t,i,:]`，输出仍为 `[T,4,D]`。GLM 的模型级顺序来自 vLLM `vllm/models/glm5next/common/model.py::Glm5NextDecoderLayer.forward`：

```text
embedding [T,D] → hc_expand → attn mHC-pre → norm → KDA 或 MLA
→ ffn mHC-pre（同时 materialize 上一处 post）→ dense/MoE
→ 非最后层保存 PendingMHC；最后层 hc_post → hc_contract(mean over 4 streams)
→ final RMSNorm → lm_head
```

每层有 `hc_attn_{fn,scale,base}` 和 `hc_ffn_{fn,scale,base}`。GLM 的 MTP 层不走主干 mHC；DeepSeek-V4 则另有 `hc_head_*` learned head，不能把 DeepSeek-V4 的 head 公式直接替换 GLM 的最终 mean contract。`hc_fn` 的 checkpoint dtype、TP 分片和加载时的 BF16→FP32 语义必须以实际 checkpoint 复核；MATE 接口明确要求 `hc_fn` 为 FP32，不能把 BF16 编码值未经语义确认直接当作 FP32 权重。

## 2. vLLM 参考

| 参考 | 关键内容 | xLLM 映射 |
|---|---|---|
| `vllm/models/glm5next/common/model.py::Glm5NextDecoderLayer` | GLM 层类型选择、两处 mHC、deferred post、KDA/MLA 与 dense/MoE 边界 | `xllm/core/layers/mlu/glm5_next/glm5_next_decoder_layer.cpp` 的 MLU 结构；未来新增 MUSA decoder |
| `vllm/model_executor/kernels/mhc/torch.py` | 可信逐元素公式和 dtype/shape | MUSA 数值参考测试 |
| `vllm/model_executor/kernels/mhc/tilelang.py`、`tilelang_kernels.py` | CUDA TileLang 的 pre/post/fused post-pre，split 配置和可选 RMSNorm | 仅借鉴 kernel 分块/融合思路，不能当作 MUSA ABI |
| `vllm/model_executor/layers/mhc.py` | `MHCPreOp`/`MHCPostOp` dispatch；CUDA TileLang、ROCm AITER、CPU fallback | 说明模型需要的公共接口，不代表 xLLM MUSA 已注册这些 CustomOp |
| `vllm/models/deepseek_v4/{cpu,amd,nvidia}/model.py` | DeepSeek-V4 的 mHC pre/post/head 调用和平台差异 | 与 MUSA DSV4 模型交叉核对；NVIDIA/AMD kernel 不能直接移植 |

vLLM CUDA 的 `mhc_fused_post_pre` 把上一处 post 和下一处 pre、norm 合成一次 kernel。GLM 首版可先按 `MHCPost → MATE MHCPre` 组合，确认 logits/greedy 序列一致后再实现跨层融合；融合不能改变 BF16 写回、FP32 coefficient 和 Sinkhorn 顺序。

## 3. MATE 能复用什么

本地 MATE 的 `mate/hyperconnection.py` 是公开入口：

- `mhc_prenorm_gemm_sqrsum(residual_flat, hc_fn, backend="deepgemm", split_k=...)`：输入 BF16 `[T,4D]` 或 `[T,4,D]`、FP32 `[24,4D]`，输出 FP32 mixes 和逐行平方和；`return_partials=True` 返回 `[S,T,24]`、`[S,T]`。
- `mhc_pre_big_fuse(partials, sqrsum, scale, base, residual, backend="tilelang", ...)`：执行 MHC pre 的非 GEMM阶段，可传入 caller-owned output buffer，适合 graph/eager 热路径。
- `mhc_pre(...)`：上述两步的 Python convenience wrapper，输入 residual 必须 BF16、`hc_fn`/scale/base 必须 FP32，返回 `(post[T,4,1], comb[T,4,4], layer_input[T,D])`。

MATE 的 kernel 细节在 `mate/jit/gemm/deep_gemm/hyperconnection.py` 和 `mate/hyperconnection_kernels/tilelang/mhc_pre.py`。MATE 当前 AOT/JIT 只针对 `mp31`/`mp32`，prenorm 有 `tile_m=32/64` staging，MP32 另有 `M=128,K=64,4 stages` 的 prefill box；`split_k`、M、目标架构必须与导出的模块一致。MATE 自带 `tests/test_hyperconnection.py`，覆盖 MP31/MP32、`T={13,137,4096,8192}`、`N=24`、`K=28672/7680/7168`、split/no-split，以及 MHC pre 对 Torch 参考的比较。

这里的“有 MATE 实现”是 Python/JIT/FFI 层面的事实，不等于稳定 C++ ABI。当前 xLLM 已采用正确方向：`deepseek_v4_hc_pre.cu` 通过 `get_module(uri)->GetFunction(...)`、`to_ffi_tensor`、`TvmffiStreamGuard` 调用 AOT `.so`，再以 native MUSA finish kernel 处理需要的 norm/输出。GLM 接入应复用该加载、stream、workspace 和 fallback 模式，而不是在 C++ 中 import Python。

## 4. 当前 xLLM 可复用实现

### MUSA（设备层，最直接）

| 文件 | 可复用点 | 现有限制 |
|---|---|---|
| `xllm/core/kernels/musa/deepseek_v4_hc_pre.cu` | MATE prenorm FFI、预分配 workspace、stream handoff、native fallback、BF16 输出 | 常量 `hc_mult=4`、`mix=24`；HC fn 要 FP32；部分 MATE URI/行数策略按 DSV4 验证；应抽成通用命名和配置 |
| `xllm/core/kernels/musa/deepseek_v4_dsa_cache_ops.cu::deepseek_v4_hc_post` | `[T,D] + [T,4,D] + post/comb` 的 MUSA post，`hc_mult=4` 有 2D 优化 kernel | 当前 API 名称为 DSV4；使用 DSV4 stream helper，GLM 复用前需确认与当前 Torch stream/graph 的一致性 |
| `...::deepseek_v4_hc_pre_combine`、`deepseek_v4_hc_pre_metadata` | pre 的 MUSA combine 和小 metadata 路径 | metadata 路径前置 mixes，不替代 MATE 大矩阵 GEMM |
| `xllm/core/kernels/musa/deepseek_v4_hc_head.cu` | DeepSeek-V4 learned HC head 的 split-K reduction，4096 hidden，BF16/FP16/FP32 dispatch | 这是 DSV4 learned head，不是 GLM 的 mean contract；hidden=4096/`T<=64` 是硬限制 |
| `xllm/core/kernels/musa/CMakeLists.txt` | HC pre/head 已进入 `cuda_kernels` | 新的通用命名/测试需要同步 CMake 和导出产物 |

当前 MUSA DSV4 decoder `xllm/core/layers/musa/deepseek_v4/deepseek_v4_decoder_layer.cpp` 已在 attention/FFN 两处调用 `hc_pre`，并在模型末端调用 HC head；这是“完整 mHC 已接到一个 MUSA 模型”的证据，但不提供 GLM 的 KDA/MLA/MoE/VL 组合。

### MLU（模型级，最直接的 GLM 参考）

`xllm/core/layers/mlu/hyper_connection.{h,cpp}` 提供 `MHCPre`、`MHCPost`、`MHCHead`、`PendingMHC`、权重 alias 解析和 `resolve_mhc_fusion`。`xllm/core/layers/mlu/glm5_next/glm5_next_decoder_layer.cpp` 已实现：

1. 第 0 层 `hc_expand` 后 standalone pre；
2. 后续层消费上一层 `PendingMHC`，执行 fused post-pre-norm（MLU kernel）；
3. attention 后再次执行 FFN pre；
4. 非最后层延期 post，最后层 materialize；
5. MLU 的 `hyper_connection_test.cpp` 覆盖 pre/post/head、Sinkhorn、fused composition、`T={1,2,129,8192}` 和权重跨 shard alias。

MLU 设备 kernel（`fused_mhc`、`fused_mhc_post`）不能直接调用于 MUSA，但其模块边界、pending 生命周期、权重命名和测试公式应复用。当前 `USE_MUSA` 的 `models/models.h` 没有 `glm5_next`/`glm5_next_vl` include；这是 GLM MUSA 不可直接启动的主要缺口。

## 5. 推荐的 MUSA 实现方案

### 阶段 A：公共 mHC 层和组合正确性

新增 `core/layers/musa/hyper_connection.{h,cpp}`（或等价公共模块），API 与 MLU 对齐：`MHCPre::forward`、`MHCPost::forward`、`PendingMHC`、`hc_expand/contract`、`get_hc_state`。实现规则：

1. pre 的大矩阵阶段严格走 MATE：优先 `mhc_prenorm_gemm_sqrsum`/`tf32_hc_prenorm_gemm`，再 `mhc_pre_big_fuse`；C++ 通过已有 TVM-FFI/AOT loader 接入，MATE 模块不可用时明确回退到已验证的 MUSA kernel，不回退到 CPU。
2. post 复用 `deepseek_v4_hc_post`，先将符号改为通用 HC API 或在 MUSA layer 中包一层；传入当前 xLLM provider stream，禁止隐式 default stream。
3. 首版不做融合：`hc_post(attn)` → `hc_pre(ffn)`，并在下一层入口消费 pending。确认数值后再加入 `fused_post_pre_norm`，复用 MLU 的 pending 计划和 vLLM 的输出顺序。
4. GLM 最终使用 `hc_contract = mean(dim=1)`；只有 DeepSeek-V4/其 MTP 使用 learned HC head。

### 阶段 B：MATE AOT、prefill/decode 和 graph

- 增加 GLM 专用但模型无关的 MATE 导出脚本，调用 `gen_hyperconnection_aot()`/相应 MATE builder；至少分别准备 MP31/MP32、decode `T<=64` 和 prefill 的 split 配置。不要复用 DSV4 的 `T<=320/1024` 策略作为 GLM 已验证结论。
- workspace 按 `[S,T,24]`、`[S,T]`、post `[T,4]`、comb `[T,4,4]` 和输出 buffer 预分配；graph capture 不在 kernel 内分配。
- MATE FFI 和 native MUSA kernel 之间保持同一 stream 的 event/dependency；现有 `deepseek_v4_hc_stream_test.cpp` 可扩展为 GLM shape 和 graph replay 测试。

### 阶段 C：接入 GLM5-Next/VL

复用 MLU `Glm5NextDecoderLayerImpl` 的控制流和已有 MUSA `Glm5NextKDA`、`Glm5NextSparseAttention`、MoE、视觉路径，新增 MUSA decoder/model/VLM 注册和 CMake。权重加载需要单独核对 `hc_attn_*`/`hc_ffn_*` 的 checkpoint dtype、prefix、TP shard；FP8 expert scale 不能通过普通 BF16 `copy_` 代替。

## 6. 明确支持边界和拒绝项

首版建议只开放：MUSA MP31（随后验证 MP32）、BF16 residual/activation、FP32 mHC coefficients、`hc_mult=4`、hidden=4096、contiguous `[T,4,4096]`、`sinkhorn_iters=20`、`hc_eps/rms_eps=1e-6`，eager prefill/decode。非 contiguous 输入、其他 `hc_mult`/hidden、FP16 mHC 权重、graph、TP/CP/PP 组合和 GLM FP8 MoE 在没有独立数值/实机验证前显式拒绝或关闭优化路径。

不要：把 MATE Python wrapper 当 C++ ABI；把 DSV4 learned HC head 当 GLM final mean；用 muBLAS/GEMM 拼 attention 代替 MATE；或把数据搬到 host 计算。mHC 不是 attention，MATE 的 attention API 也不是合适替代品。

## 7. 验证清单

静态检查已完成：MATE 版本/入口、vLLM 公式、MLU GLM pending 链、MUSA DSV4 HC kernel/CMake 和 `USE_MUSA` 模型注册均已核对。当前报告未启动完整 GLM-VL 服务，也未声称 checkpoint 输出正确。

落地后必须按以下顺序验收：

1. MATE `tests/test_hyperconnection.py` 在目标设备/架构通过；新增 C++ MUSA pre/post/head 测试，与 Torch/vLLM 参考比较 FP32 coefficients、BF16 layer input、post/comb，覆盖 `T=1,2,16,64,129,8192`、chunked prefill 和 decode。
2. 检查 AOT `.so` 的实际符号、`ldd`、MUSA arch 和 FFI stream；测试 graph capture/replay 与 workspace 复用。
3. 以真实 GLM checkpoint 比较单层、两层 pending 链、完整 prefill 和多步 decode 的 logits/greedy token；另测 KDA/MLA、MoE、视觉 token、TP 和状态槽复用。
4. 最终启动 MUSA GLM-VL 服务并发送纯文本、图片、视频、第二条请求和长 prompt；日志需确认实际 MUSA kernel/activity，而不是只看 HTTP 200。

## 参考文件

- vLLM：`/data/albert/Projects/vllm/vllm/models/glm5next/common/model.py`、`vllm/model_executor/kernels/mhc/{torch.py,tilelang.py,tilelang_kernels.py}`、`vllm/model_executor/layers/mhc.py`。
- MATE：`/data/albert/Projects/mate/mate/hyperconnection.py`、`mate/jit/gemm/deep_gemm/hyperconnection.py`、`mate/hyperconnection_kernels/tilelang/mhc_pre.py`、`tests/test_hyperconnection.py`。
- xLLM MUSA：`xllm/core/kernels/musa/deepseek_v4_hc_pre.cu`、`deepseek_v4_hc_head.cu`、`deepseek_v4_dsa_cache_ops.cu`、`xllm/core/layers/musa/deepseek_v4/deepseek_v4_decoder_layer.cpp`。
- xLLM MLU：`xllm/core/layers/mlu/hyper_connection.{h,cpp}`、`xllm/core/layers/mlu/glm5_next/glm5_next_decoder_layer.cpp`、`tests/core/layers/mlu/hyper_connection_test.cpp`。
