# MUSA MTP 优化状态（2026-08-18）

## 1. 结论摘要

本文件记录 dev92 权威环境中，Qwen3.5-27B-FP8 与
Qwen3.5-35B-A3B-FP8 在 MTP K=2、MUSA graph、FA3、piecewise prefill
路径上的最新性能、正确性和优化归因。

当前结论如下：

- Mate 已从旧的可变 cache/legacy bridge 路径迁移到明确锁定的
  **Mate 0.2.5** 源码、MUBIN 和 cached-op 布局。
- Qwen3.5-27B-FP8 的严格 C1 同协议对比，相对 2026-08-14 legacy
  binary，Total TGS 提升 **19.14%**，TPOT 降低 **16.17%**，TTFT
  降低 **1.70%**。Acceptance 为 **78.69%**，已恢复到 K=2 的
  75% 目标以上。
- Qwen3.5-27B-FP8 的 C4 当前达到 **417.25 Total TGS / 646.82 ms
  TTFT / 17.69 ms TPOT / 77.65% acceptance**。相对 315.92 TGS 的
  历史 corrected target，名义上是 TGS +32.07%、TTFT -32.82%、
  TPOT -24.20%。
- Qwen3.5-35B-A3B-FP8 的 MTP K=2 acceptance 更高：C1 为
  **90.27%**，C4 为 **86.78%**。这说明官方 35B-A3B draft 与 target
  已经在 graph/FA3/GDN/MoE 状态路径上正确对齐，没有再被 stale
  metadata 或错误 recurrent-state commit 人为拉低接受率。
- 35B-A3B 当前 C1 达到 **316.48 Total TGS / 103.46 ms TTFT /
  6.27 ms TPOT**；C4 达到 **719.61 Total TGS / 257.33 ms TTFT /
  10.30 ms TPOT**。
- 四组正式测量都生成完整 2000 output tokens；27B C1/C4 和
  35B-A3B C1/C4 的长输入数学 correctness 检查均通过；最终回归还
  得到正确答案 **45924**，没有 FATAL、graph aborted 或无意义输出。
- 性能提升不是单一改动造成的。Mate 0.2.5 提供了更快、更完整的
  GDN/FA3/GEMM 基础能力；xLLM 侧同时修复了 MTP state/metadata
  correctness，并消除了 graph replay、MoE GEMM、sampling 和稳定
  buffer 路径上的额外开销。

## 2. 权威环境和当前代码状态

- Host checkout：`/data/feihu/xllm-git-master`
- Container checkout：`/workspace/xllm-git-master`
- Container：`xllm_musa_latest_sj2`
- Local branch：`codex/master-head-sync-20260802`
- Functional commit：
  `151a5af7a12a74026057642afe4fe9d2b2217862`
- Style-policy commit：
  `e37bdd598352eef41257e2e469c4b67ecb3444fa`
- Bench binary：
  `/workspace/xllm-git-master/build/lib.linux-x86_64-cpython-310/xllm/xllm`
- Binary SHA256：
  `679ba8baa057e0c273e9094451c545e85e0b7dbc1d13b92e1421f19490a2e560`
- Binary mtime：`2026-08-18 23:28:53 +0800`
- Mate source：`/data/feihu/mate_0.2.5`
- Mate version：`0.2.5`
- Device：MUSA device 2

上述 binary 与 functional commit 的源码内容一致；commit 只固化了已经
完成 rebuild、correctness 和 matched benchmark 的工作树。

## 3. 正式 benchmark 协议

所有 2026-08-18 正式结果使用：

- ISL=2000，OSL=2000
- MTP K=2
- temperature=0，top-k=0，top-p=1
- graph=on
- piecewise prefill graph=on
- packed prefill=off
- FA3 prefill=on
- FA3 decode=on
- MUSA pool compute stream=on
- schedule overlap=off
- C1：4 requests warmup，10 requests measure
- C4：1 wave/4 requests warmup，10 waves/40 requests measure
- Acceptance 只用 measure 前后 BRPC counter delta 计算，不混入 warmup：
  `accepted_delta / draft_delta`

模型与 draft：

- 27B target：
  `/data/nfs_shared/models/Qwen3.5-27B-FP8`
- 27B draft：
  `/data/feihu/bench_results/mtp_27b_fp8_k1_eval_20260807/draft_current_export`
- 35B-A3B target：
  `/data/nfs_shared/models/Qwen3.5-35B-A3B-FP8`
- 35B-A3B official draft：
  `/data/nfs_shared/models/Qwen3.5-35B-A3B-MTP-draft`

## 4. 当前最佳实测

| Model | C | Input TGS | Output TGS | Total TGS | Mean TTFT | Mean TPOT | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-27B-FP8 | 1 | 68.569 | 68.569 | **137.139** | **251.405 ms** | **14.465 ms** | **78.686%** |
| Qwen3.5-27B-FP8 | 4 | 208.624 | 208.624 | **417.248** | **646.821 ms** | **17.689 ms** | **77.650%** |
| Qwen3.5-35B-A3B-FP8 | 1 | 158.241 | 158.241 | **316.481** | **103.462 ms** | **6.271 ms** | **90.266%** |
| Qwen3.5-35B-A3B-FP8 | 4 | 359.803 | 359.803 | **719.607** | **257.333 ms** | **10.304 ms** | **86.782%** |

Acceptance 原始计数：

| Model | C | accepted delta | draft delta | rate |
|---|---:|---:|---:|---:|
| 27B-FP8 | 1 | 12,231 | 15,544 | 78.6863% |
| 27B-FP8 | 4 | 48,665 | 62,672 | 77.6503% |
| 35B-A3B-FP8 | 1 | 12,872 | 14,260 | 90.2665% |
| 35B-A3B-FP8 | 4 | 50,757 | 58,488 | 86.7819% |

35B-A3B 相比 27B 的 acceptance 高约 9.13--11.58 个百分点。这是模型和
draft 质量的差异，不应全部归因于 runtime；但它只有在 target/draft
的 FA3、GDN state 和 MoE metadata 正确对齐后才会稳定体现为端到端收益。

## 5. 与历史代码的对比

### 5.1 27B：严格 C1 同协议对比

最可比 baseline：

`/data/feihu/bench_results/v2_k2_c1_strict_measure_greedy_piecewise1_4w10m_20260814/`

该 baseline 与当前结果使用相同 device、draft、seed、greedy sampling、
ISL/OSL=2k、piecewise=1、schedule overlap=false、4 warmup + 10 measure。

| Metric | 2026-08-14 legacy | 2026-08-18 Mate 0.2.5 + fixes | Delta |
|---|---:|---:|---:|
| Total TGS | 115.105 | 137.139 | **+19.14%** |
| TTFT | 255.752 ms | 251.405 ms | **-1.70%** |
| TPOT | 17.256 ms | 14.465 ms | **-16.17%** |
| Acceptance | 77.434% | 78.686% | **+1.252 pp** |

这组对比最能证明 decode 性能没有回退并有明显提升。Acceptance 的
提升与 TPOT/TGS 收益方向一致；当前稳定高于 75% 目标。

旧 artifact 只记录可变的 `/workspace/mate_cached_ops`，未锁定具体 Mate
版本，因此它是 legacy/0.2.2-era 路径证据，不能被表述为严格
“Mate 0.2.2 vs 0.2.5”单变量 A/B。

### 5.2 27B：C4 corrected target 对比

历史 corrected artifact：

`/data/feihu/bench_results/mtp_k2_c4_3warm_isl2000_osl2048_20260808/k2_c4_mc4_w3_m4_seed44002/result_server_tokens_corrected.json`

| Metric | Historical corrected target | Current | Nominal delta |
|---|---:|---:|---:|
| Total TGS | 315.924 | 417.248 | **+32.07%** |
| TTFT | 962.781 ms | 646.821 ms | **-32.82%** |
| TPOT | 23.337 ms | 17.689 ms | **-24.20%** |
| Acceptance | 76.233% | 77.650% | +1.418 pp |

该结果只能作为方向性状态对比，不是严格 A/B：旧运行 OSL=2048、
piecewise=off、3 warmup + 4x4 measure，并使用 server-token correction；
当前 OSL=2000、piecewise=on、1x4 warmup + 10x4 measure。旧 raw client
JSON 因输出 token 漏计仅报告 219.115 TGS，不能和 corrected/current
口径混用。

### 5.3 27B：merge regression 已恢复

2026-08-13 merge A/B 的 acceptance 是：

- premerge：14,563 / 26,938 = 54.061%
- postmerge：14,130 / 27,800 = 50.827%

这两组使用 temperature=0.9、top-k=20、top-p=0.95，且 acceptance 包含
warmup，不能与当前 greedy measure-only 的 76--78% 做严格数值 A/B。
但当前结果说明当时“acceptance 严重下降、TPOT 变差”的状态已经恢复。

相对 2026-08-09 nominal regression run（286.557 TGS、661.46 ms TTFT、
26.557 ms TPOT、66.560% acceptance），当前 C4 名义上：

- Total TGS +45.61%
- TTFT -2.21%
- TPOT -33.39%
- Acceptance +11.09 pp

同样需要保留 prompt/OSL/provenance 不完全一致的限定。

### 5.4 35B-A3B：最接近的旧 greedy 结果

旧 C1：

`/data/feihu/bench_results/qwen35_35ba3b_fp8_mtp_k2_adaptive_spec_verify_c1_isl2k_osl2k_20260809_0825/result.json`

旧 C4：

`/data/feihu/bench_results/qwen35_35ba3b_fp8_mtp_k2_adaptive_spec_verify_c4_isl2k_osl2k_20260809_0820/result.json`

| C | Metric | 2026-08-09 | 2026-08-18 | Directional delta |
|---:|---|---:|---:|---:|
| 1 | Total TGS | 236.048 | 316.481 | **+34.08%** |
| 1 | TTFT | 127.472 ms | 103.462 ms | **-18.84%** |
| 1 | TPOT | 8.315 ms | 6.271 ms | **-24.59%** |
| 4 | Total TGS | 383.487 | 719.607 | **+87.65%** |
| 4 | TTFT | 282.833 ms | 257.333 ms | **-9.02%** |
| 4 | TPOT | 19.468 ms | 10.304 ms | **-47.07%** |

旧结果使用 OSL=2048、piecewise=off、schedule overlap=on，且 measure
数量少于当前，因此只能用于方向性比较。旧 35B artifact 没有保存可核验的
greedy long-run acceptance counter 前后快照，不能虚构 before/after delta。

有 counter 的旧 temperature=0.9 结果约为 70.99%（C1）和 71.50%
（C4 measure），但 sampling 协议不同，不能和当前 greedy 的 87--90% 直接
比较。旧短 greedy correctness counter 为 249/282 = 88.30%，与当前
87--90% 接近。能够确认的是：当前长上下文、10 个 measure waves 下稳定保持
86--90%，MoE/GDN graph 状态没有随请求漂移；不能声称 Mate 0.2.5 单独将
acceptance 从 71% 提升到 90%。

## 6. Mate 0.2.5 带来的基础能力和收益

Mate `v0.2.2..v0.2.5` 中与本路径直接相关的变化包括：

- `187eacd4 [SW-76391] Optimize gdn decode and mtp`：
  优化 TileLang GDN single-token decode 和 T>1 MTP kernel。
- `00899a91 [SW-83270] FMHA Fix for FA3 100%`：
  修正和增强 MP31 FA3 forward/paged-KV/combine 路径。
- `015b3643 Add fp8 fp8 mega moe mate kernels`：
  增加 FP8 x FP8 MoE 专用 kernel。
- `5d50a67e [SW-81242] ... refactor runtime ... groupgemm`：
  重构 MUBIN runtime/dispatch，并扩展 grouped GEMM。
- `ed767fc9 fix(deep_gemm): robust load fp8 scales`：
  修正 FP8 scale 加载和 accumulation 边界。

xLLM 的 0.2.5 集成不是简单替换目录：

1. `_build_cuda_graph_musa.sh` 和 `run_xllm_musa.sh` 明确校验
   `version.txt == 0.2.5`，统一 `MATE_HOME`、`MATE_WORKSPACE_BASE`、
   `MATE_MUBIN_DIR` 和 `FLASHINFER_OPS_PATH`。
2. 启动前检查 FA3 prefill/decode、1--17 warp metadata、combine kernel、
   MoE launcher 和对应 MUBIN object，缺 artifact 时 fail-fast，避免静默
   fallback 到慢路径或运行时才 FATAL。
3. `attention_decode.cpp` 对齐 0.2.5 的 dense/paged FA3 URI、metadata
   形状和 scheduler ABI，直接复用 persistent scheduler metadata。
4. `musa_moe_gemm.cpp` 直接调用 0.2.5 MUBIN launcher，并针对
   contiguous/ragged/masked、BF16/FP8、token 数和 MP 数选择 kernel，
   减少通用 wrapper、动态编译和不匹配 tile 的开销。
5. GDN decode/MTP 使用 0.2.5 的正式 `main` ABI、state indices、
   checkpoint indices 和 stable output/intermediate buffer，不再依赖
   旧版特制 bridge SO。
6. `run_xllm_musa.sh` 默认关闭 core dump（`ulimit -c 0`），避免
   graph/kernel crash 产生数十 GB 的无用 apport coredump。

## 7. xLLM 侧的主要优化

### 7.1 MTP acceptance 与 correctness

- **完整的 K=2 validate token materialization**：
  `mtp_worker_impl.cpp` 直接在 device 上逐列写入两个 draft token，
  不再让 host/device token row 在异步路径上失配。
- **expanded per-token attention metadata**：
  每个 speculative token 获得正确的 KV length、block table、
  paged-KV indptr/indices/last-page-len；C=3,K=2 等形态可安全 pad 到
  resident graph bucket，而不是把 padding row 当作 live row。
- **双槽 stable MTP graph buffer**：
  `mtp_graph_buffers.cpp/.h` 预分配 pinned host + device tensor，
  用 stream event 管理生命周期。Graph 捕获和 replay 看到的是稳定地址，
  runtime 只更新内容。
- **GDN checkpoint/commit 语义修复**：
  target verify 为每一步写中间 recurrent/conv state，rejection 后只提交
  实际 accepted prefix 对应的 state；不再无条件提交最后一个 draft step。
- **capture warmup state snapshot/restore**：
  graph warmup、FFI record 和真实 capture 都会执行 forward。现在在每次
  preparation pass 前恢复 conv/SSM state，保证第一个 replay 不会重复消费
  当前 token。
- **direct conv superstate / checkpoint path**：
  MTP causal-conv 可直接写最终 superstate，减少大 intermediate buffer，
  同时保留按 accepted token 选择 checkpoint 的正确性。
- **greedy rejection fused MUSA kernel**：
  在 device 上生成 accepted 和 masked token；第一次 mismatch 后严格
  mask 后续 token，并正确处理 bonus token，避免 host round-trip。
- **target-only selected-prob sampling**：
  只对实际 draft token 的 target probability 做 rejection，避免生成
  整个 vocab dense probability tensor。
- **terminal KV capacity guard**：
  request 末尾或 block table 容量不足时不预启动下一次 first draft，
  避免异步越界错误在后续 stream sync 才暴露。
- **安全限制 schedule-overlap shortcut**：
  C1 不启用没有独立工作可重叠的 combined-first-draft；
  device-context/combined-first-draft 在 metadata 未完全覆盖前保持 opt-in。
- **MoE batch verify 适配**：
  Qwen3.5 MoE 的 target/draft graph 使用正确 batch/row layout，
  解决只在 35B-A3B 路径上出现的状态或 token 对齐问题。

这些 correctness 修复对 35B-A3B 特别重要：它同时包含 MoE、GDN 和
full-attention，任何一处 stale row 都会让下一步 draft 与 target 分布偏离。
当前长跑 C1/C4 的 90.27%/86.78% 说明状态已稳定对齐，但不是一个经过
单变量 A/B 隔离的 Mate acceptance 增益。

### 7.2 Decode/TPOT

- 2026-08-07 已完成 GDN 两项优化：
  - Mate MTP 自己计算 gating 时，xLLM 不再重复做 sigmoid/exp/softplus。
  - causal-conv 将 10240 channels 分散到 `grid.y`，不再由一个 block
    串行遍历所有 channel。
- 上述 GDN target T=2 graph body 从 49.682 ms 降到 **33.061 ms**，
  降低 **33.5%**。
- Mate 0.2.5 继续提供正式优化后的 GDN T>1 kernel。
- FA3 scheduler metadata、decode KV metadata 和 expanded spec-verify
  metadata 都使用 persistent device buffer；完整 device metadata 时不再
  每层 D2H 重建。
- decode metadata update 合并 token/position/cache-slot/KV-length/paged-KV
  的更新，并在 device 上 padding，减少每 step host work。
- compact top-k 和 selected-only probability sampling 避免 decode 每步构造
  大 vocab 中间 tensor。
- LM-head 的 C1 BF16 固定形状使用 GEMV dispatch；linear/MoE 输出使用
  persistent output buffer，减少小 batch allocation。
- 大 MTP verify bucket 默认继续走 graph；eager 只保留显式安全回退。

### 7.3 Prefill/TTFT

- piecewise graph 采用可复用 bucket，而不是每个新长度重复 capture。
- Linear/GDN/attention runner 使用稳定输出 buffer 和 persistent metadata；
  replay 时只更新 length、offset、CU-seqlens 和 live endpoint。
- GDN KKT endpoint 按 64 token 对齐，但 live CU metadata 保留真实长度，
  padding 不进入 recurrent state。
- packed/piecewise graph 可复用已存在的同 batch 更大 bucket，减少
  33 graph segments/16 runners 的重复 capture 边界成本。
- dense SwiGLU + FP8 activation quantization 融合，避免先生成 BF16
  activation 再单独量化；当前路径扩展到所有合法 row 数。
- row-parallel FP8 down projection 和 graph matmul 使用持久输出 buffer；
  buffer grow 时保留旧 allocation，防止已捕获 graph 指针失效。
- FA3 dense/paged prefill 复用 scheduler metadata 和 LSE scratch，
  避免 capture 内分配。

### 7.4 MoE/35B-A3B

35B-A3B 的大幅提升主要来自组合收益：

- 0.2.5 FP8/BF16 grouped GEMM MUBIN；
- contiguous/ragged/masked 三类 token layout 的 shape-aware launcher；
- MoE target/draft 的 K=2 batch row 修复；
- expanded FA3 per-token metadata；
- GDN accepted-state checkpoint；
- 当前 86--90% 的稳定 long-run draft acceptance。

Acceptance 越高，每次 target verify 平均产出的有效 token 越多；同时 MoE
GEMM 和 graph replay 更快，因此 TPOT 的改善会相乘，而不是简单相加。

## 8. Correctness 证据

长上下文测试问题为 `12345 * 67 - 89`，正确结果是 `827026`。

已通过：

- 27B-FP8 C1 MTP K=2：长输入 512/1024-token response smoke PASS。
- 27B-FP8 C4 MTP K=2：单请求和四并发均包含 `827026`，PASS。
- 35B-A3B-FP8 C1 MTP K=2：包含 `827026`，PASS。
- 35B-A3B-FP8 C4 MTP K=2：单请求和四并发均包含 `827026`，PASS。
- 四组正式 benchmark 的所有 measure request 都完成完整 2000 output
  tokens。
- 最终 27B/35B-A3B C1 4 warmup + 10 measure 均完成 10 个 2K 输出。
- 最终 35B-A3B 2K 输入数学检查在 1024-token response 中命中
  正确答案 `45924`。
- Qwen3-0.6B 使用同一通用 FA3 predicate，命中正确答案 `391`；
  piecewise bucket1792 和 decode bucket1 capture 成功，未生成 FA2
  PlanInfo。
- Qwen3.5-27B 请求 `BLOCK_SIZE=128` 时先归一为有效 64，保持 FA3
  prefill/decode，数学答案 `45924` 和 graph capture 均 PASS。

Correctness artifact：

- `/data/feihu/bench_results/mate025_c1_k2_correctness_ab_20260818/mtp_k2/`
- `/data/feihu/bench_results/mate025_c4_k2_strict_measure_greedy_piecewise1_1w10m_20260818/correctness_c4_long_math.log`
- `/data/feihu/bench_results/mate025_qwen35_35ba3b_fp8_c1_k2_strict_greedy_piecewise1_4w10m_20260818/correctness_c1_long_math.log`
- `/data/feihu/bench_results/mate025_qwen35_35ba3b_fp8_c4_k2_strict_greedy_piecewise1_1w10m_20260818/correctness_c4_long_math.log`
- `/data/feihu/bench_results/fa3_planinfo_boundary_qwen3_06b_postfix_final_679ba8ba/`
- `/data/feihu/bench_results/mate025_postfix_block128_effective64_correctness_20260818_final_679ba8ba_r2/`
- `/data/feihu/bench_results/mate025_postfix_c1_k2_strict_greedy_4w10m_20260818_final_679ba8ba/`
- `/data/feihu/bench_results/mate025_qwen35_c1_k2_strict_greedy_4w10m_20260818_final_679ba8ba/`

## 9. 本轮最终 gate

1. **FA3/PlanInfo predicate PASS**：FA3 shape 判定由 MUSA layer 提供并被
   graph executor 复用，不再用 Qwen3.5 model 名称代替真实 capability。
   Qwen3-0.6B 这一“同 shape、不同模型”路径未生成 FA2 PlanInfo。
2. **有效 block size PASS**：Qwen3/Qwen3.5 当前实现本来就要求并强制使用
   64-token KV block。launcher 现在在 FA3/PlanInfo preflight 之前把用户
   请求的非 64 值归一成有效值 64，避免“CLI 先判 FA2、binary 后改回 64”
   的错误双重语义。真正保持非 64 page size 的其他模型仍只让 decode
   回落 FA2；普通 prefill 不受 decode page-size 限制。
3. **Fast metadata 风险未扩大**：Qwen3.5 device metadata fast path 继续
   保留 model/block64 资格；通用 FA3 predicate 不会误放宽该优化。
4. **Build PASS**：容器 `xllm_musa_latest_sj2` 完成完整 `xllm` link。
   最终日志：
   `/data/feihu/xllm-git-master/build_logs/build_cuda_graph_musa_20260818_232725.log`。
5. **Correctness PASS**：Qwen3-0.6B、27B-FP8、35B-A3B-FP8，以及
   Qwen3.5 请求 block128 后归一到有效 block64 的路径均输出正确结果；
   无 FATAL、graph aborted 或无意义输出。
6. **Matched performance PASS**：
   - 27B C1：相对 predicate 修正前 TGS **+1.892%**、TTFT **+0.148%**、
     TPOT **-1.874%**、acceptance **+2.561 pp**。
   - 35B-A3B C1：TGS **+0.733%**、TTFT **+0.702%**、
     TPOT **-0.739%**、acceptance **+2.591 pp**。
   两组 TTFT 的小幅变化都低于 1%，同时 TGS/TPOT/acceptance 改善，
   不构成性能回退。
7. **Image gate**：functional/docs commit 完成后制作 sourceful dev image，
   并在无 host source mount 的新容器中重新编译 `xllm`。

## 10. 结果目录

- 27B C1：
  `/data/feihu/bench_results/mate025_postfix_c1_k2_strict_greedy_4w10m_20260818_final_679ba8ba/`
- 27B C4：
  `/data/feihu/bench_results/mate025_c4_k2_strict_measure_greedy_piecewise1_1w10m_20260818/`
- 35B-A3B C1：
  `/data/feihu/bench_results/mate025_qwen35_c1_k2_strict_greedy_4w10m_20260818_final_679ba8ba/`
- 35B-A3B C4：
  `/data/feihu/bench_results/mate025_qwen35_35ba3b_fp8_c4_k2_strict_greedy_piecewise1_1w10m_20260818/`

## 11. 归因边界

- 不能把全部提升都归因于 Mate 0.2.5；xLLM 的 state/metadata/buffer/
  sampler 修复同样关键。
- 不能把旧可变 `mate_cached_ops` artifact 严格标记为 Mate 0.2.2。
- 不能把不同 OSL、warmup、piecewise、schedule-overlap 或 sampling
  协议的结果称为单变量 A/B。
- 35B-A3B 旧 greedy long-run 没保存 acceptance counter delta，因此只能
  报告当前 90.27%/86.78% 的稳定值，不能编造 before/after improvement。
- 性能通过不等于 correctness 通过；本轮以正确答案、完整 token 数、
  acceptance counter、无 graph/runtime error 共同作为 gate。
