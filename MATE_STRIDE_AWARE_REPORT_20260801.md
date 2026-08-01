# MATE FFI stride-aware 改造与 eager TTFT 验证（2026-08-01）

## 结论

本阶段已完成 MATE GDN prefill FFI 的 stride-aware 改造，并在 MUSA GPU 3、Qwen3.5-27B-FP8、C1、ISL=2000、OSL=16 的相同 eager 配置下验证。

最终 xLLM eager 三轮均值：

- TTFT：**269.401 ms**
- sample std：0.852 ms
- 范围：268.630–270.316 ms
- 成功请求：60/60
- model forward：235.854 ms

此前同卡 SGLang eager 两轮均值为 270.976 ms。因此当前 point estimate 为：

- xLLM 比 SGLang 快 **1.575 ms**
- 相对改善约 **0.58%**

这不是大幅领先，但已经从“xLLM 稳定慢于 SGLang”推进到三轮 xLLM 均低于 SGLang 均值。

## Git 快照

开始改造前已按要求冻结全部工作树状态。

主仓库分支：

`codex/mate-stride-aware-20260801`

主仓库快照提交：

- `3e05b26a53383696b7148e2498c9757d732b5a30`
- `4b92bfc1ac06c3af94c155f6d575cb04715cc4b6`

第二个提交用于冻结嵌套的 `.codex_tmp_bucket_src` Git 仓库。五个 dirty dependency submodule 也分别建立了本地 `codex/snapshot-20260801` 分支并提交，主仓库 gitlink 已指向这些 snapshot。

## 根因

容器内已安装的 MATE 0.2.3 源码已经对 Q/K/V 使用：

`T.StridedTensor(..., dynamic leading strides, last_stride=1)`

但 xLLM 部署目录 `/workspace/mate_cached_ops` 中原有的 GDN prefill 和 KKT shared object 是较早导出的 contiguous ABI，运行时会检查：

`q.IsContiguous()`

因此 xLLM 必须在每个 GDN layer 前 materialize Q/K/V。Qwen3.5-27B 包含 48 个 GDN layer，该开销累计成为剩余 TTFT 差距。

此外，xLLM 自己的 fused Q/K L2 normalization kernel 原先把所有 head row 当作连续二维矩阵，也不能直接原地处理 fused-QKV storage 上的 4D split views。

## 实现

### 1. 新的独立 stride-aware MATE URI

新增四个模块，旧模块不覆盖：

- `mate_gdn_prefill_full_strided_hq16_hv48_bf16`
- `mate_gdn_prefill_full_varlen_strided_hq16_hv48_bf16`
- `mate_kkt_solve_strided_hq16_hv48_bf16`
- `mate_kkt_solve_varlen_strided_hq16_hv48_bf16`

新模块使用当前 MATE TileLang `StridedTensor` 定义并导出 `main` TVM-FFI function。旧模块继续使用原 URI 和 `run`/legacy ABI。

xLLM 只有在 GDN 和对应 KKT stride-aware 模块均存在、且 Q/K/V 最后一维 stride 为 1 时才启用新路径；否则自动回退旧 contiguous ABI。

显式回退：

`XLLM_MATE_GDN_DISABLE_STRIDED_ABI=1`

### 2. stride-aware fused Q/K L2 normalization

文件：

`xllm/core/kernels/musa/gdn_decode.cu`

kernel 现在按 `[batch, token, head, dim]` 的三个 leading strides 计算每个 row 的输入和输出地址；只要求最后一维连续。这样 Q/K 可以直接在 fused-QKV storage 的 split view 上原地归一化，无需预先 contiguous。

### 3. 可复现导出工具

新增：

`tools/export_mate_gdn_prefill.py`

工具会：

1. 构建 padded/varlen GDN prefill module。
2. 构建 padded/varlen KKT module。
3. 构造真实 fused-QKV leading-stride views。
4. 对每个 module 做 strided vs contiguous output exact comparison。
5. 导出 shared object 并验证 `main` FFI function。
6. 写入包含环境与 SHA256 的 manifest。

运行命令：

```bash
python3 tools/export_mate_gdn_prefill.py \
  --ops-path /workspace/mate_cached_ops \
  --hq 16 --hv 48 --head-dim 128 --dtype bf16
```

## Module validation

四个 module 均通过 strided-vs-contiguous exact comparison。

| Module | SHA256 |
|---|---|
| GDN padded | `4cd31f08df1faa290ebb0c1a3428eea6113ceb5c2438d88cee1c53f6739a514c` |
| GDN varlen | `bc34bb0d9766f920d803ceb810b6d2b23583adcb1ff43f36f6b9754f5700bf33` |
| KKT padded | `130ed8f585921a9bd40a8db69f53bbf0fc845eb1db2f2c704a943f57a09cf4f6` |
| KKT varlen | `af94be7a1cf4d78fc0a1b5cdb4af7946798b0f26753e645adf2da95114ad54a1` |

MATE：`0.2.3.dev20260602+mu510`

Torch：`2.9.1`

## 端到端路由与正确性

URI smoke 明确命中：

`mate_gdn_prefill_full_varlen_strided_hq16_hv48_bf16`

而不再触发 `q.IsContiguous()`。

fused strided L2 与通用 BF16 reference 的最大差：

- Q：0.0078125
- K：0.00390625

这是 BF16 reduction/rounding 级差异。最终使用生成文本做语义门禁：

- measured requests：5/5
- 与改造前 validated default：逐文本 exact match
- 两侧 generated-text list SHA256：
  `a1a5f2387a80fec521de6974717b0dae8aab8fc0cb8b84947ec8a8252416d012`
- FATAL / NaN / `IsContiguous` error：0

注意：该 SHA256 使用 JSON compact encoding 计算，与上一报告对不同序列化格式得到的 SHA 不应直接比较；本报告中的 old/new 使用完全相同计算方式。

## 同 binary 成对 A/B

两轮采用相反运行顺序；唯一变量为是否设置：

`XLLM_MATE_GDN_DISABLE_STRIDED_ABI=1`

| 轮次 | ABI | TTFT (ms) | model forward (ms) | forward 外部 (ms) | 成功 |
|---|---|---:|---:|---:|---:|
| r1 | legacy contiguous | 272.342 | 239.329 | 33.013 | 20/20 |
| r1 | stride-aware | 270.316 | 236.495 | 33.820 | 20/20 |
| r2 | stride-aware | 268.630 | 235.548 | 33.082 | 20/20 |
| r2 | legacy contiguous | 275.293 | 241.539 | 33.754 | 20/20 |
| **两轮均值** | **legacy** | **273.818** | **240.434** | **33.383** | **40/40** |
| **两轮均值** | **stride-aware** | **269.473** | **236.022** | **33.451** | **40/40** |

成对均值改善：

- endpoint TTFT：**-4.345 ms**
- model forward：**-4.413 ms**
- forward 外部：+0.068 ms

收益完整落在 model forward 内部，符合本次改造目标。

## 默认 stride-aware 三轮

| 轮次 | TTFT (ms) | model forward (ms) | mean server tokens | 成功 |
|---|---:|---:|---:|---:|
| strided r1 | 270.316 | 236.495 | 2058.8 | 20/20 |
| strided r2 | 268.630 | 235.548 | 2063.6 | 20/20 |
| strided r3 | 269.257 | 235.519 | 2055.9 | 20/20 |
| **均值** | **269.401** | **235.854** | **2059.4** | **60/60** |

同卡 SGLang eager 历史配对结果：

- TTFT：270.976 ms
- model forward：231.866 ms
- forward 外部：39.110 ms

因此：

- xLLM model forward 仍慢约 3.988 ms。
- xLLM C++/forward 外部快约 5.563 ms。
- 最终 endpoint TTFT 快约 1.575 ms。

后续若继续追求稳定领先，目标仍应放在剩余 GDN core/model-forward 差距，而不是 C++ 服务框架。

## Breakdown

旧默认和 stride-aware 各取 5 个 measured profiler sample：

| Bucket | 旧默认 (ms) | stride-aware (ms) | 变化 |
|---|---:|---:|---:|
| wall | 242.117 | 239.140 | -2.977 |
| GDN total | 85.308 | 82.156 | -3.152 |
| MATE scope | 19.287 | 16.272 | **-3.014** |
| GDN non-MATE | 66.021 | 65.884 | -0.137 |
| MLP | 126.203 | 126.468 | +0.265 |

profiler 直接证明主要收益来自 MATE wrapper/materialization 路径。

## Build

最终格式化源码 build log：

`build_logs/build_cuda_graph_musa_20260801_173924.log`

未出现 compiler error 或 Ninja failure。

最终 binary：

`build/lib.linux-x86_64-cpython-310/xllm/xllm`

SHA256：

`ff3cf78e7adaac9a032fcebf2a2816b57eafbbe29b5ce9e334cd5a22547edc5c`

最终 binary smoke：1/1 成功，FATAL/NaN/contiguous ABI error 为 0。

## Evidence

主目录：

`/workspace/bench_results/mate_stride_aware_20260801_172500`

有效结果：

- `xllm_strided_smoke`
- `correctness_strided_v2`
- `legacy_r1`
- `legacy_r2`
- `strided_r1`
- `strided_r2`
- `strided_r3`
- `strided_breakdown_v2`
- `final_binary_smoke`

保留的 fail-closed 记录：

- `correctness_strided`：客户端缺少两个官方 benchmark module，请求未执行。
- `strided_breakdown`：机械替换破坏 shell 引号，服务未启动。

两者均未用于任何性能或正确性结论。
