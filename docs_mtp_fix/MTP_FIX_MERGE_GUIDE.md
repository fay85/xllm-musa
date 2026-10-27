# Qwen3.5/3.6/3.8 MTP on MUSA: Fix Merge Guide (xllm_musa_jd_master)

- 容器: `xllm_musa_jd_master` (s5000_92, 10.121.38.92), 源码 `/workspace/xllm-jd-master`
- Base: `92dbd809c` (feat: enable fused addrmsnorm on prefill for the qwen3 moe decoder layer. (#500))，全部改动均为 **未提交 (uncommitted)**，未 commit / push
- 验证 binary: `build/lib.linux-x86_64-cpython-310/xllm/xllm` md5 **`90b56d6360c62a7bd3b32772d2fc34e0`**
  - 备份: `xllm.bak_90b56d63`（同 md5）、`xllm.bak_68998edc`、`xllm.bak_257525b6`（早期版本）
  - 源文件备份: `*.bak_pre_mtp_fix`（各修改文件修改前的工作区版本）
- 交付物:
  - `mtp_fix.patch` —— 仅真实修复 (fixes only)，基于 **干净的 92dbd809c** 生成，`git apply --check` 通过
  - `mtp_diag.patch` —— 临时诊断代码 (XLLM_MTP_DIAG / XLLM_MTP_DIAG_FWD / `mtp_diag_tmp.h`)，**不建议合入**，仅供复现/排障

---

## 0. 重要背景：工作区 ≠ 干净 HEAD

`/workspace/xllm-jd-master` 的工作区在本次 MTP 工作开始前，已经带有一层较大的 **MUSA 移植 / 基础改动**（约 44 个文件，例如 `model_input_params.h` 的 paged-KV host 字段、`GdnMtpVerifyCache`、`musa_graph_executor_impl.cpp` 的 FA3 graph decode 判定 `fa3_graph_decode`、mRoPE kernel `mrope.cu/.h`、`qwen3_5_mtp_base.h` 的 MUSA attn_metadata 复用 等）。这些 **不属于** 本次修复，也不在 `mtp_fix.patch` 中。

因此：
- `mtp_fix.patch` = 92dbd809c + 本次 4 项修复 (a)(b)(c)(d) + 修复 (b) 必需的最小依赖（`ModelEmbeddingInput::mtp_token_embedding` 字段、`Qwen3_5MtpModelImplBase::forward` 中的 embedding fallback）。
- 在干净 92dbd809c 上 **apply 成功**，但 **未在干净 worktree 上完整编译**（完整 MUSA 编译需上述移植层，且耗时数小时）；已编译验证的是 "移植层 + 修复" 的工作区（md5 90b56d63）。
- 合入其它分支时，若目标分支已有 MUSA 移植层（如 feihu-1016-dev），请按下文冲突提示逐项比对。

---

## 1. 修复列表总览

| # | 修复 | 文件 | 适用范围 |
|---|---|---|---|
| a | MUSA top_k + top_p 过滤 & `musa::random_sample` | `framework/sampling/logits_utils.cpp`, `framework/sampling/sampler.cpp` | 通用（所有模型，MUSA 平台） |
| b | MTP draft decode 走 graph：持久化 MTP token embedding（graph 外预计算 + clamp），移除强制 eager | `runtime/musa_graph_executor_impl.h/.cpp`, `framework/model/model_input_params.h`, `models/llm/qwen3_5_mtp_base.h` | Qwen3.5/3.6/3.8 MTP（`qwen3_5_mtp` / `qwen3_5_moe_mtp`），MUSA graph |
| c | draft graph 持久化元数据 `max_seqs_per_batch *= 2` + block_tables 行数 CHECK | `runtime/musa_graph_executor_impl.cpp` | Qwen3.5 MTP 专用（×2）；CHECK 通用 |
| d | Block-FP8 `weight_scale_inv` 随 Q/G 行重排同步重排 + accessor | `layers/musa/qwen3_next_attention.cpp`, `layers/common/linear.h` | Qwen3-Next/3.5 系 `attn_output_gate` + block-FP8 (target 与 MTP 层均受影响) |

---

## 2. 逐项说明

### (a) MUSA top_k + top_p 过滤 / random_sample

**现象**: MUSA 上 MTP 接受率极低（~16%），temp>0 时输出质量也异常。

**根因**: `apply_top_k_top_p()` 中 `top_k && top_p` 同时设置的分支只有 NPU/MLU 实现，MUSA 落空 → logits 未被过滤；MTP rejection sampling 复用这些 logits，全词表概率导致接受率崩溃。同时 `Sampler::random_sample` 在 MUSA 下没有走 MUSA kernel。

**改动**:
- `logits_utils.cpp` `apply_top_k_top_p()`：新增 `#if defined(USE_MUSA)` 分支（移植自 feihu-1016-dev）：`topk(k)` 窗口（k = max(top_k)，否则 min(vocab,1000)），窗口内 top_k mask + top_p 累积概率 mask，再 `fill_(-inf)` + `scatter_` 回写，`return`。
- `sampler.cpp` `Sampler::random_sample()`：`#if defined(USE_MUSA) return xllm::kernel::musa::random_sample(probs);`（`musa_ops_api.h` 在 92dbd809c 已声明该函数）。

```cpp
#if defined(USE_MUSA)
  if (top_k.defined() || top_p.defined()) {
    ... auto [sorted, idx] = logits.topk(k, -1, true, true);
    ... sorted.masked_fill_(k_mask, ninf); ... sorted.masked_fill_(p_mask, ninf);
    logits.fill_(ninf);
    logits.scatter_(-1, idx, sorted.to(logits.scalar_type()));
    return;
  }
  return;
#endif
```

**注意**: 当 top_k 未设置时窗口上限为 1000 token（与 feihu 一致），这是近似；`top_k.max().item()` 有一次 host sync。

**冲突提示**: feihu-1016-dev 已有同等 USE_MUSA 分支 → 合入时保留一份即可。

**验证**: temp 0.6 下接受率从 ~0.16 恢复到 0.7+（K=1）；GPQA 答案正确。

---

### (b) MTP draft decode 走 full graph + 持久化 MTP token embedding

**现象**: draft token 全为 0、draft attention 输出 NaN，接受率≈0。

**根因**:
1. `MusaGraphExecutorImpl::run()` 中旧代码（来自 commit `49fa19564` "feat: add MUSA graph executor. (#2117)"）对 `qwen3_5_mtp` decode 强制 eager：
   ```cpp
   if (is_decode && is_qwen35_mtp) { COUNTER_INC(num_model_execution_total_eager); return model_->forward(...); }
   ```
   该 eager 路径与 graph 捕获的 target 使用的持久化 metadata/FA3 scheduler 不一致，draft attention 产生 NaN。
2. 改走 graph 后，draft forward 内的 `embed_tokens_(tokens)`（IndexSelect）在 MUSA graph capture 中不安全，且 tokens 含 -1 占位（padding/被拒 slot）会越界。

**改动**:
- `model_input_params.h` `ModelEmbeddingInput`：新增 `torch::Tensor mtp_token_embedding;`，并在 `to()` 中 `safe_to(..., true)`。
- `musa_graph_executor_impl.h` `MusaGraphPersistentParam`：新增 `persistent_mtp_token_embedding_` 和 accessor `persistent_mtp_token_embedding(actual_tokens)`。
- `musa_graph_executor_impl.cpp`:
  - `maybe_precompute_embedding_for_graph()`：对 qwen3_5_mtp，在 graph 外 `embed_layer(tokens.clamp(0, vocab-1))` 预计算（无 host sync）。
  - capture 时 `params_for_capture->embedding.mtp_token_embedding = persistent_mtp_token_embedding(padded_num_tokens)`。
  - `update()` 时把 `mtp_token_embedding` 拷贝进持久化 buffer（首次按 `max_tokens_per_batch` 分配）。
  - 内存统计加 `bytes(persistent_mtp_token_embedding_)`。
  - **删除** 强制 eager 分支。
- `qwen3_5_mtp_base.h` `forward()`：优先使用 `input_params.embedding.mtp_token_embedding`，未定义时 fallback `embed_tokens_(tokens)`（eager / 非 MUSA 不受影响）。

**适用范围**: Qwen3.5/3.6/3.8 MTP 专用（判定 `args_.model_type()` 为 `qwen3_5_mtp`/`qwen3_5_moe_mtp`）；数据结构字段本身通用。

**冲突提示**:
- 其它分支若仍保留 `49fa19564` 的强制 eager 行，需删除；若已删除（feihu-1016-dev 已走 graph），只需比对持久化 buffer 实现。
- 若目标分支已有 `mtp_token_embedding` 字段（本工作区移植层中已有），跳过 `model_input_params.h` hunk。
- `qwen3_5_mtp_base.h` 在目标分支可能已有 MUSA attn_metadata 复用块，hunk 上下文可能需手工调整。

**验证**: draft token 非 0、无 NaN；graph capture 日志 `Lazy capturing` 覆盖 draft decode bucket；无 eager 计数增长。

---

### (c) draft graph 元数据 `max_seqs_per_batch *= 2` + CHECK

**现象**: C=4 时接受率崩溃（C=1 正常）。

**根因**: Qwen3.5 MTP draft 在完全接受后每个请求会产生 2 行 attention（repair 行 + 当前行），draft 最多 2×max_seqs 行。持久化 per-seq 元数据按 max_seqs=4 分配，graph capture 对 bucket>max_seqs 时 `std::min(actual_batch_size, persistent_block_tables_.size(0))` **静默截断** block_tables，但 forward 仍跑全部 token 行 → 多出的行读错 KV。

**改动** (`musa_graph_executor_impl.cpp`):
```cpp
// MusaGraphPersistentParam ctor, before get_decode_graph_token_bucket
if (args_.model_type() == "qwen3_5_mtp" || args_.model_type() == "qwen3_5_moe_mtp") {
  max_seqs_per_batch *= 2;
}
...
// update(): replace silent min-truncation
CHECK_GE(persistent_block_tables_.size(0), actual_batch_size) << "...";
const int64_t actual_block_table_batch =
    is_rec_multi_round_mode() ? params.attention.device.block_tables.size(0)
                              : actual_batch_size;
```

**适用范围**: ×2 为 Qwen3.5 MTP 专用；CHECK 为通用防御（任何模型行数超出都会显式报错而非静默错误）。

**冲突提示**: **feihu-1016-dev 已有 `max_seqs*=2`** → 合入时不要重复乘 2（会 ×4 浪费显存）；CHECK 可独立保留。

**验证**: C=4 K=1 各请求接受率一致（不再出现 1 个 0.99 + 3 个低值），日志无 CHECK 失败。

---

### (d) Block-FP8 `weight_scale_inv` Q/G 重排

**现象**: FP8 模型接受率偏低（target 与 MTP 层 attention 精度下降）。

**根因**: `Qwen3NextAttentionImpl::load_state_dict()` 在 `attn_output_gate` 时把 `q_proj` 权重行从 per-head 交错 `[q,g]` 重排成分组 `[Q|G]`，但 **未同步重排 block-FP8 的 `weight_scale_inv` 行**（128 行一块）→ Q 与 G 行使用了对方的 scale 反量化。

**改动**:
- `linear.h` `QKVParallelLinearImpl`（`#if defined(USE_MUSA)`）：新增 `weight_scale_inv()` 与 `is_weight_scale_inv_loaded()` accessor。
- `qwen3_next_attention.cpp` `load_state_dict()`，紧随 `qg_rows.copy_(reordered);`：
```cpp
torch::Tensor scale_inv = qkv_proj_->weight_scale_inv();
if (scale_inv.defined() && scale_inv.numel() > 0 && qkv_proj_->is_weight_scale_inv_loaded()) {
  constexpr int64_t kBlock = 128;
  CHECK_EQ(head_dim_ % kBlock, 0);
  const int64_t bph = head_dim_ / kBlock, qg_blocks = q_size_ * 2 / kBlock;
  auto s_rows = scale_inv.slice(0, 0, qg_blocks);
  auto s_3d = s_rows.view({num_heads_, 2 * bph, scale_inv.size(1)});
  auto sq = s_3d.slice(1, 0, bph).reshape({num_heads_ * bph, -1});
  auto sg = s_3d.slice(1, bph, 2 * bph).reshape({num_heads_ * bph, -1});
  s_rows.copy_(torch::cat({sq, sg}, 0));
}
```
（Qwen3.8-27B: head_dim 256，24 q heads，q_proj [12288,5120]，scale [96,40]。）

**适用范围**: Qwen3-Next / Qwen3.5 系 `attn_output_gate` + block-FP8；BF16 不受影响（scale 未定义时跳过）。

**冲突提示**: feihu-1016-dev **不做** Q/G 重排（保持交错布局，`fused_qk_norm_rope` 用 `q_head_stride=2`），因此不需要此修复；若合入目标分支采用 feihu 布局，**整段 (d) 不适用**，且需确认 `qkv_weight_reordered_` 路径不存在。要求 `head_dim % 128 == 0`。

**验证**: FP8 K=1 C=4 temp0.6 接受率 0.68–0.71 → 0.74–0.78；GPQA 0.84 → 0.89。

---

## 3. 测试注意事项

1. **严格 greedy 判定** (`sampling_params.cpp` ~L188)：
   `do_sample = p->do_sample || temperature != 0 || top_p != 1.0 || top_k > 0`
   → 只发 `temperature=0` 但带 `top_p=0.95, top_k=20` 仍走 random 路径。严格 greedy 必须 `temperature=0, top_p=1, top_k=-1`（或不发）。本次之前的 "greedy" 数据均为 random 路径，已作废。
2. **接受率读取**: `curl localhost:<port>/vars/*speculative_num*` 或 `/brpc_metrics`，accept = `speculative_num_accepted_tokens_total / speculative_num_draft_tokens_total`（取 run 前后差值）。jd 另有 `speculative_num_drafts_total`、`speculative_num_committed_tokens_total`；jd 日志 per-request `acceptance_rate` 亦可用。`speculative_*_per_pos` 在三个 binary 中均未导出（无数据）。
3. **客户端 NaN 误报**: 测试脚本 `jd_g3s.py` 的 `nan` flag 只是文本中 "nan" 子串出现 >3 次（如 finance/maintenance），**不是** 服务端 NaN；在 feihu/prod 上同样出现。

---

## 4. 验证结果

模型 Qwen3.8-27B-FP8，draft_model=同模型 (MTP)，GPU3，mem 0.80，piecewise_and_full graph，prefix cache on，prompt 2448 tokens / 输出 2048。
- jd = 本修复 binary `90b56d63`
- prod1012 = `xllm_1012_prod` 只读拷贝 binary `f1edd58e`（branch bugfix/qwen3-beam-topk-local）
- feihu = `xllm_qwen3_8-27b-fp8_g3` `/opt/xllm` 只读拷贝 binary `3817049f`（feihu-1016 部署）

### K=1（accept = accepted/draft）

| Run | jd | prod1012 | feihu |
|---|---|---|---|
| C=1 strict greedy | 0.754 | 0.772 | 0.812 |
| C=4 strict greedy | 0.775/0.766/0.759/0.756 | 0.774 | 0.771 |
| C=1 temp0.6 | 0.729 | 0.750 | 0.763 |
| C=4 temp0.6 | 0.742/0.733/0.727/0.708 | 0.751 | 0.815 |
| GPQA greedy | 0.872 (B) | 0.903 (B) | – |
| GPQA temp0.6 | 0.905 (B) | 0.818 (B) | – |

jd 为日志 per-request 值，prod/feihu 为 /vars 计数器。

### K=4, C=1

| Run | jd | prod1012 | feihu |
|---|---|---|---|
| strict greedy (run 1) | 0.414 (log) | 0.426 | 0.426 |
| strict greedy (run 2, /brpc_metrics) | 0.459 (1330/2896) | 0.415 (1280/3084) | 0.392 (1250/3192) |
| temp0.6 | 0.396 (log) | 0.388 | 0.432 |
| GPQA greedy (run 1) | 0.570 (log) | 0.633 | 0.628 |
| GPQA greedy (run 2) | 0.723 (133/184) | 0.575 (122/212) | 0.569 (123/216) |

结论：修复后 jd 的 MTP 接受率与 prod1012 / feihu 参考处于同一水平（差异在 run-to-run 噪声内）；长 prompt 上 ~0.9 的期望在参考 binary 上同样达不到，~0.9 仅在 GPQA 类短 prompt 上出现。

---

## 5. 启动参数 & 测试命令

```bash
# env 继承自容器 (MATE_*/FLASHINFER_OPS_PATH/XLLM_USE_FA3*/XLLM_GDN_DECODE_BACKEND=mate/XLLM_MATE_GDN_MTP=1 ...)
export LD_PRELOAD=/workspace/artifacts/libmusa_backend_init.so
export MUSA_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 MTHREADS_VISIBLE_DEVICES=3
M=/data/nfs_shared/models/Qwen3.8-27B-FP8
xllm --model=$M --model_id=Qwen3.8-27B-FP8 --draft_model=$M --backend=llm \
  --host=0.0.0.0 --port=19183 --master_node_addr=127.0.0.1:21183 --nnodes=1 --node_rank=0 \
  --random_seed=-1 --block_size=64 --max_memory_utilization=0.80 \
  --max_concurrent_requests=4 --max_seqs_per_batch=4 \
  --enable_prefix_cache=true --enable_chunked_prefill=true --max_tokens_per_chunk_for_prefill=1024 \
  --enable_schedule_overlap=true --enable_graph=true --enable_graph_mode_decode_no_padding=true \
  --enable_graph_vmm_pool=false --enable_prefill_piecewise_graph=true --max_tokens_for_graph_mode=1024 \
  --num_speculative_tokens=1 --max_tokens_per_batch=4096 \
  --speculative_algorithm=MTP --max_linear_state_cache_slots=4
```

容器内脚本：
- 启动: `/tmp/jd_k1_fp8.sh`（K=1）、`/tmp/jd_k4.sh` / `/tmp/prod_k4.sh` / `/tmp/fh_k4.sh`（K=4）
- 客户端: `MID=Qwen3.8-27B-FP8 TEMP=0 TOPP=1 TOPK=-1 python3 /tmp/jd_g3s.py {c1|c4|gpqa}`
- 套件: `/tmp/acc_suite.sh`, `/tmp/k4_suite.sh`, `/tmp/pp_suite.sh`（记录并只 kill 自己启动的 PID）
- 计数器: `bash /tmp/specm.sh` → `accepted draft`

编译：
```bash
export PATH=scripts/ninja_guard:$PATH
./scripts/ninja_safe.sh build/cmake.linux-x86_64-cpython-310 -j8 xllm   # rc 恒为 0，需 grep " error"
```

---

## 6. 合入步骤建议

1. `git apply --check mtp_fix.patch`（基于 92dbd809c 已验证通过）；目标分支有移植层时用 `git apply -3` 并按第 2 节冲突提示处理。
2. 对照 feihu-1016-dev：跳过 (a) 重复分支、(c) 的 ×2；若使用交错 Q/G 布局，跳过 (d)。
3. 不要合入 `mtp_diag.patch`（含 `mtp_diag_tmp.h`、`[MTP_DIAG]`/`[MTP_FWD]`/`[MTP_L]` 日志）。若需复现，在已打 fix 的工作区 `git apply mtp_diag.patch`，用 `XLLM_MTP_DIAG=1` / `XLLM_MTP_DIAG_FWD=1` 开启。
4. 编译后按第 4/5 节复测 K=1 C=1/C=4 strict greedy + temp0.6 + GPQA。
