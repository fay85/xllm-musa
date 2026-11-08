<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->
# GLM-5.3-Flash MUSA 完整 sparse attention 层

日期：2026-11-05。完整层入口为 `Glm5NextMLAAttentionImpl::forward`，输入 `[T,hidden_size]`，输出 `[T,hidden_size]`。已接通输入融合投影、RMSNorm、完整 IndexPool/Indexer、NoPE sparse MLA、输出投影、TP reduction、权重加载及 KVCache/metadata。默认真实尺寸为 hidden=4096、Q rank=1536、64 heads。

这里完成的是可由模型调用的 C++ attention 层，并非整个 GLM 的 KDA/MoE/mHC/Vision 模型注册。真实 checkpoint 正在下载，尚未进行真实权重精度或整模型请求验证；下文的数值结果来自合成权重和实际 MUSA 设备计算。

## 文件及调用链

| 部分 | 实现 | 行为 |
| --- | --- | --- |
| 完整层 | [glm5_next_mla_attention.h](../../xllm/core/layers/musa/glm5_next/glm5_next_mla_attention.h)、[实现](../../xllm/core/layers/musa/glm5_next/glm5_next_mla_attention.cpp) | `Glm5NextMLAAttention`、加载/验证权重、完整 forward、cache factory |
| Indexer | [glm5_next_indexer.h](../../xllm/core/kernels/musa/glm5_next_indexer.h)、[实现](../../xllm/core/kernels/musa/glm5_next_indexer.cu) | 压池、tail、Hadamard、FP8、MATE scoring、Top-K、展开 |
| sparse MLA | [flash_mla_sparse.h](../../xllm/core/kernels/musa/flash_mla_sparse.h)、`flash_mla_sparse.cpp`、`flash_mla_sparse_cache.cu` | BF16/FP8 NoPE attention、主 cache 写入、逻辑索引到物理 slot 映射 |
| 离线生成 | [compile_musa_sparse_mla.py](compile_musa_sparse_mla.py) | 导出 attention、Indexer metadata/scoring 模块及 ABI manifest |
| 层级测试 | [glm5_next_mla_attention_test.cpp](../../tests/core/kernels/musa/glm5_next_mla_attention_test.cpp) | 公式参考、分块/状态/加载/TP/实际尺寸测试 |
| 验证工具 | [validate_musa_sparse_attention.py](validate_musa_sparse_attention.py) | 算子及整层测试，两个独立进程、两张卡的真实 MCCL TP 测试 |

```text
hidden_states [T,4096]
  → fused_qkv_a_proj [T,1536+512]
  → Q RMSNorm / KV RMSNorm
  → q_b_proj [T,H,256]
  → Indexer(hidden_states, normalized_q_raw)
      → wq_b / fused wk+weights_proj / FP32 weights_proj / compress_gate
      → index K FP32 LayerNorm → BF16
      → rate-4 softmax pool + BF16 rounding + Hadamard128 + BF16 + FP8
      → MATE paged MQA logits → Top512 pools → 2048 token IDs + 0..3 tail
  → Q absorption [T,H,512]
  → per-token latent cache write + causal logical-to-physical mapping
  → MATE NoPE sparse MLA [T,H,512]
  → V recovery [T,H,256] → o_proj + TP all-reduce → [T,4096]
```

主 scale 显式为 `1/16`，Indexer 的 head/dim 缩放为 `1/64`，Q 的量化 scale 只折入一次 head weights。主 attention 和 Indexer 均不运行 RoPE。Indexer 不按 TP 切 head；主 q_b/kv_b/o 按现有 Column/RowParallelLinear 切分和通信。

## 算子与库

- **MATE**：主 BF16 independent-candidate sparse prefill、GLM NoPE FP8 scheduled decode；Indexer 的 FP8 paged MQA logits 和 scheduling metadata。
- **muDNN / TorchMUSA**：输入/输出和 Indexer 投影、Q/V BMM、Top-K。已核对安装源码 `torch_musa/csrc/aten/ops/Matmul.cpp` 的 `Bmm → BmmOut → BmmCall → musa::dnn::BatchMatMul::Run`，以及 `Topk.cpp` 的 `musa::dnn::TopK`。
- **已有 xLLM norm 封装**：Q/KV RMSNorm、FP32 index K LayerNorm；分别保留 `rms_norm_eps` 和 `1e-6`。
- **MUSA kernel**：GLM rate-4 pooling/tail、Hadamard128、FP8 编码、pool 展开及物理 slot 映射。这部分不由通用 attention API 提供。

主 attention 与 scoring 均找到了匹配的 MATE 原语，故沿 MATE → muDNN → muBLAS 决策顺序停在 MATE，没有另拼 GEMM attention。muDNN 用于已有投影/Top-K 封装，未引入 CPU activation fallback。

FFI 复用 `get_function`、`to_ffi_tensor`、`TvmffiStreamGuard` 和现有 allocator。scoring 返回的 FFI tensor 由 Torch view 持有其 owner，stream guard 保证 Torch Top-K 读取前生产已建立依赖。workspace 随调用存活；scoring 按 query chunk 约束 FP32 logits 临时内存。线上不启动 Python/JIT。

## Cache 和 metadata 契约

主 cache 与 index cache 共享 **256-token 页的 block_table**：

| 内容 | shape / dtype | 物理布局 |
| --- | --- | --- |
| BF16 主 cache | `[P,256,1,512]` BF16 | 每原始 token 一条512维 latent |
| FP8 主 cache | `[P,256,1,528]` uint8 | 每 token 内联512个 E4M3 + 四个 group-128 FP32 scale |
| IndexPool cache | 逻辑 `[P,64,1,132]` uint8 | **每页先放64×128个 E4M3，再放64个 FP32 scale，非逐 pool 内联132字节** |
| tail | `[state_slots,2,4,128]` BF16 | raw index K 与 gate ring；状态槽0保留为padding |

Index cache 的物理布局已对照安装版 `csrc/templates/gemm/deep_gemm/paged_mqa_logits_kern.j2` 中 payload/scale 指针核实。不要混用主 cache 的 inline 格式；之前调研中按逻辑 shape 推断逐条 record 的理解需以上述实际 ABI 为准。

`create_cache` 使用已有 `KVCache(IndexedKVCacheTensors)` 所有权与 tensor role，不建立第二套缓存容器。输入 `AttentionMetadata` 需要 device `block_table`、`slot_mapping`、`linear_state_indices`；packed/chunked prefill 提供 `q_cu_seq_lens`，普通每请求一个 token 的 decode 允许省略该字段。`max_seq_len` 是本批次最大可见 token 数，不是 pool 数。positions 在请求内连续递增，batch 按 query sequence boundaries 分组，写 slot 和目标状态槽须在请求间唯一。

每个 query 的完整 pool 上界为 `floor((position+1)/4)`，因此提前压好当前 chunk 的未来 pool 不会造成未来信息泄漏。未成池的0～3个 tail token 额外追加，2176列 buffer 其余位置填 -1。length 表示候选前缀结束位置，包含中间的 -1 空洞。短序列直接选择全部因果可见 token，仍维护 pool/tail 状态以供后续 decode。

`linear_state_read_indices` 可提供独立来源槽，复制使用更新前快照；position=0 清理复用槽的 tail。被 padding 的请求不写 cache，输出为零。跨请求释放/恢复由调用方现有 scheduler/缓存管理承担；已验证 device tail 恢复，尚未验证完整 host prefix-cache 或 PD 传输流程。

## 权重加载及使用

调用方先去掉真实 checkpoint 的 layer/self_attn 前缀，再逐 safetensors shard 调用 `load_state_dict`。canonical keys：

- `q_a_proj.weight`、`kv_a_proj_with_mqa.weight`、`q_b_proj.weight`、`kv_b_proj.weight`、`o_proj.weight`。
- `q_a_layernorm.weight`、`kv_a_layernorm.weight`。
- `indexer.wq_b.weight`、`indexer.wk.weight`、`indexer.weights_proj.weight`。
- `indexer.k_norm.weight`、`indexer.k_norm.bias`、`indexer.index_kpool_compress_gate`、`indexer.index_kpool_compress_ape`。

支持 BF16/FP32 原始权重和 E4M3 权重及 FP32 `weight_scale_inv`（128×128 block）。FP8 矩阵在设备上反量化为 BF16 后再融合/分片，与 vLLM attention 的 BF16 计算语义对应；Indexer head-weight 副本保持 FP32。允许 weight 与 scale 分不同 shard 到达，未齐全时不会标记加载完成；原始 mmap Tensor 被 clone 保证存活，加载完成后释放暂存副本。不支持加载完成后在线重载，也不把 INT8/W4A8 当作已支持格式。

```cpp
Glm5NextMLAAttention attention(args, parallel_args, options);
// 对每个 checkpoint shard：
attention->load_state_dict(shard.get_dict_with_prefix(attention_prefix));
attention->verify_loaded_weights();
auto cache = Glm5NextMLAAttentionImpl::create_cache(
    num_pages, num_state_slots, options, /*fp8_main_cache=*/false);
auto [output, selection] = attention->forward(
    positions, hidden_states, metadata, cache);
```

共享 Indexer 的层构造时传 `enable_indexer=false`，forward 传同批次来源层的 `selection`；只复用选择，不跳过本层投影、主 cache 写入和 attention。shared selection 的预算及 batch/position 必须一致。o_proj 的原生 Linear 可能复用 scratch，本层返回 clone 后的独立 output，避免后续 decode 改写已返回的 prefill 结果。

## 生成、构建与复验

MATE checkout `9d3260d633597f08fc9d5d4dc73344872241c34d`；实际导出固定安装版 **0.2.8.post1**（与 checkout 路径/ABI 有差异）。详细版本对照见 `musa_sparse_attention.md`。主 attention 的 TileLang 模块按 head 数和 split 上限导出，Indexer 导出 `glm_kpool_metadata_v1` / `glm_kpool_score_v1`。

```bash
python tools/musa_glm5_3_vl_bench/compile_musa_sparse_mla.py --ops-path build/mate_sparse_ops
# 已有主 attention 模块时，仅补 Indexer：
python tools/musa_glm5_3_vl_bench/compile_musa_sparse_mla.py --ops-path build/mate_sparse_ops --indexer-only

export TORCH_MUSA_PYTHONPATH=/usr/local/lib/python3.10/dist-packages/torch_musa/share/cmake
export MUSAMAPPING_PATH=/usr/local/musa/tools/musamapping
export PYTHON_INCLUDE_PATH=/usr/include/python3.10
export PYTORCH_INSTALL_PATH=/usr/local/lib/python3.10/dist-packages/torch
export CPATH=/usr/include/python3.10
cmake --build build/cmake.linux-x86_64-cpython-310 --target glm5_next_mla_attention_test flash_mla_sparse_test -j 32
python tools/musa_glm5_3_vl_bench/validate_musa_sparse_attention.py
```

生成器、层/kernel CMake 和 MUSA 测试目标均已接入。`validate` 工具保留各测试日志，TP worker 通过两个新进程和两张设备建立真实 MCCL ProcessGroup，并与未分片层比较；直接单进程运行 worker 用例会 skip，不能把 skip 作为 TP 已通过的证据。

## 验证与参考

参考语义来自本地 vLLM `models/glm5next/common/attention.py`、`nvidia/ops/kpool_compress.py` 和公共 MLA/sparse MLA；xLLM 接口参考 MLU `deepseek_v2_attention.cpp`、`glm5_next_kpool_indexer.cpp`，cache/stream/Linear 沿用当前仓库公共和 MUSA 实现。MLU 设备代码没有直接当作 MUSA 支持。

测试独立重算投影和 norm、用 CPU 标量公式计算 pooling/Hadamard/scoring，并显式展开256维 K/V 作为整层 attention 参考。CPU BLAS 在本机有 LP64/ILP64 混合链接问题，测试参考避免依赖它；生产激活与缓存计算全在 MUSA。FP8 主 cache writer 的 scale 与逐元素量化结果也单独比较，使用显式 round-to-nearest 除法处理边界。

覆盖：多请求和页边界、512-pool真实预算、2048+3尾部、BF16/FP8完整 forward、整段与分块 prefill/decode、weight/scale 分片加载、共享选择、padding、tail 来源/目标槽和槽复用、TP权重分片与输出归约、64 heads/4096 hidden/1536 Q rank实际尺寸合成权重。

支持范围为 eager MP31、BF16激活、local H=8/16/32/64、NoPE QK/V=256、latent=512、index H32/D128、rate4、token Top-K为4的倍数且≤2048、主 cache页256。DCP/PCP、MTP/spec、graph capture/replay、其他cache格式未开放。prefill 更新目前每请求顺序处理本chunk，性能和长期压力测试尚未完成。导出既有 MATE factory 时的 TileLang 静态 data-race 及当前MTCC的 sched_boundary 警告未关闭；不能用有限数值用例推断所有长期运行配置安全。

真实权重下载完成后，剩余验收是该 checkpoint 的权重/量化配置、真实长请求和输出精度；不应把本层测试通过写成完整 GLM/VL serving 已支持。


## 最终执行结果

xLLM 基线：`4b49ec74d5d37f024aa9bb89e0c2fd1dc870bb6d`，vLLM 参考：`ebac032b92cc55c2232deced9217cd5122648ce8`。在最终源码编译后执行完整 `validate_musa_sparse_attention.py`，返回0：

| 测试 | 结果 | 日志 |
| --- | --- | --- |
| 主 sparse MLA 算子 | 3/3通过，2439 ms | [算子日志](sparse_attention_validation/xllm_cpp_sparse_mla_test.log) |
| 完整层、IndexPool、状态和实际尺寸 | 8/8通过，10296 ms | [整层日志](sparse_attention_validation/glm_sparse_layer_test.log) |
| 两卡MCCL，rank0 | 1/1通过，5317 ms | [rank0日志](sparse_attention_validation/glm_sparse_mccl_rank0.log) |
| 两卡MCCL，rank1 | 1/1通过，5334 ms | [rank1日志](sparse_attention_validation/glm_sparse_mccl_rank1.log) |

整层主参考使用 `rtol=0.04, atol=0.004`，BF16分块结果使用 `0.01/0.001`，FP8分块结果使用 `0.02/0.002`；完整token选择集合与标量评分参考逐行比较。长预算用例覆盖2053个token，实际尺寸用例为hidden4096/Q-rank1536/64 heads/Top-K2048；真实MCCL用例用两张卡、global16/local8 heads的合成层对照未分片层。它们不是checkpoint输出精度或吞吐测试。

`ldd`解析到`/workspace/mudnn_3.4.0/lib/libmudnncxx.so.3`、`tvm_ffi/lib/libtvm_ffi.so`和`/workspace/MCCL_2.3.0/mccl/lib/libmccl.so.2`，未见缺失库。所有修改的C++/CUDA文件经过clang-format **20.1.6**；Python工具语法检查、`git diff --check`及本报告本地链接检查通过。
