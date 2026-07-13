# Hybrid TTFT Reduction - Detailed Execution Plan

> Generated: 2026-07-13 (GLM-5.2)
> Source plan: `/workspace/hybrid_ttft_reduction_5a26050f.plan.md`
> Predecessor status: `prefill_ttft_status_0712.md`
> Repo: `/workspace/xllm-git-master` (branch `b2-mate-prefill-fix`, container `xllm-musa2.9.1-sdk5.1-dev`)
> SGLang reference: `/workspace/sglang_qwen35` (plan's `/data/feihu/sglang_qwen35` resolves here inside the container)

## 0. Executive summary

> **Progress 2026-07-13 15:55:** Phase A warp Mate prefill is live. Official
> C=5 mean TTFT **1946 ms** (median 2178; was `_var0` 2132 / `_var2` 3489).
> Smoke PASS. Phase-1 <=1.65 s and SGLang ~1.42 s still open — see §13.

The source plan closes a remaining Qwen3.5-27B MUSA C=5 TTFT gap (xLLM ~2.150 s
mean vs SGLang ~1.422 s) by (1) instrumenting, (2) implementing packed varlen
Mate GDN prefill, (3) batching pure prefills and routing them to eager varlen
while keeping decode on graph, (4) optimizing residual kernel gaps, and
(5)/(6) validating.

Verification of every plan point against the current codebase shows:

- The **prerequisite** (multi-request piecewise-prefill `slot_mapping` fix) is
  **already applied** (`cuda_graph_executor_impl.cpp:645-671, 896-916`); the
  0712 doc's line numbers (596-619 / 836-857) are stale.
- The **largest missing piece** is the packed varlen Mate GDN prefill entry
  point: `MateGatedDeltaRulePrefillParams` has no `cu_seqlens` field, and
  `mate_gated_delta_rule_prefill` only accepts 4D `[B,T,H,D]`. The Mate kernel
  itself supports `cu_seqlens` (SGLang passes it), so this is an xLLM-side
  exposure/caller refactor, not a kernel rewrite.
- The **scheduler** restricts pure prefills to one-per-batch under graph+piecewise
  (`chunked_prefill_scheduler.cpp:99-103, 151-156, 360-367`); there is no
  eager-varlen route and no per-batch execution hint.
- **Stage timing is essentially absent** in the LLM path (only coarse counters
  + decode-only graph timing; no piecewise-prefill replay timing, no GDN/attention
  breakdown). The benchmark script runs xLLM only (no paired SGLang).
- The plan's diagnosis contains one **inaccuracy**: SGLang does NOT disable
  piecewise graphs/chunking for MUSA+Qwen3.5; it keeps piecewise ON but runs
  GDN/attention as eager split-ops inside the graph and packs multiple prefills
  per batch under a 16384 budget.

Critical path: **Step 1 (instrument) -> Step 2 (packed kernel) -> Step 3
(scheduler/routing)**. Step 2 is highest risk/value.

---

## 1. Verification of the source plan (point by point)

### 1.1 Section 1 - instrument-baseline (TODO: pending)

| Plan requirement | Status | Evidence |
|---|---|---|
| Opt-in MUSA-event stage timing (scheduling, graph replay, GDN, attention) | **MISSING** | No xLLM-native `musaEvent` timing in the LLM path; only vendored flashinfer MoE uses `musaEvent_t`. `xllm/core/util/timer.h:23-38` is CPU wall-clock only. |
| Whole-scheduling-loop timing | EXISTS (coarse) | `scheduling_latency_seconds` counter (`metrics.cpp:109`), observed in all schedulers (e.g. `chunked_prefill_scheduler.cpp:890`). No admission/block-alloc/batch-build sub-breakdown. |
| Graph replay timing (decode) | EXISTS (opt-in) | `XLLM_GRAPH_TIMING=1` (`cuda_graph_executor_impl.cpp:71-77`) logs `replay_ms` on the decode branch (`:1934-1957`). |
| Graph replay timing (piecewise prefill) | **MISSING** | The piecewise replay branch (`cuda_graph_executor_impl.cpp:1837-1887`) has no timer. No capture begin/end elapsed either. |
| GDN / attention / forward sub-stage timing | **MISSING** | LLM worker has only `execution_latency_seconds_model` (`metrics.cpp:78-79`). The REC path has `XLLM_DEBUG_ONEREC_XATTN_STAGE_TIMING` but the Qwen3.5 LLM path does not. |
| `XLLM_TTFT_WATERFALL` | **MISSING (proposed only)** | Defined in `tools/prefill_ttft_baseline.md:15,32,41-45` but zero code matches anywhere in the repo. |
| Paired xLLM/SGLang cold+warm TTFT, queue, prefill-forward, logits/sample, HTTP delay | **MISSING** | `.tmp_run_official_c5_bench.sh` runs xLLM only. `conc_eval_sglang.sh` runs in a separate container; no automated paired A/B. |
| Server-side TTFT measurement | EXISTS (end-to-end) | `time_to_first_token_latency_milliseconds` histogram (`metrics.cpp:123-124`), observed at `continuous_scheduler.cpp:1360-1364`. No stage split. |

### 1.2 Section 2 - varlen-mate-prefill (TODO: pending)

| Plan requirement | Status | Evidence |
|---|---|---|
| Mate prefill consumes packed q/k/v/g/beta + `cu_seqlens` + state indices + packed output | **MISSING** | `MateGatedDeltaRulePrefillParams` (`xllm/core/kernels/param.h:1690-1701`) has no `cu_seqlens`, no `state_indices`. `mate_gated_delta_rule_prefill` (`gdn_prefill.cpp:486`) takes 4D `[B,T,H,D]`, enforces it at `:492-493`, derives batch from `query.size(0)`, and the FFI `run()` call (`:592-600`) passes no cu_seqlens. |
| Do not materialize `[batch, max_len]` / do not process padding / no per-seq fallback | **MISSING** | `reshape_qkvz_with_pad` (`qwen3_gated_delta_net_base.cpp:1833-1874`) always pads to `max_len` and stacks. Per-sequence PyTorch loop at `:1502-1572` with comment "MUSA chunk_gated_delta_rule does not honor cu_seqlens yet." |
| Reuse varlen causal-conv metadata + one state gather/scatter per layer | **PARTIAL** | `causal_conv1d.cu:189,241` is already varlen; called with `query_start_loc=q_cu_seq_lens` at `qwen3_gated_delta_net_base.cpp:1175-1183`. State gather/scatter via `index_select`/`index_put_` is already per-request (`:1340,1369,1509,1571`) and varlen-compatible. Only the GDN **core** lacks the packed entry point. |
| Preserve B=1 graph-safe path + k-last state layout + guarded fallback | EXISTS | B=1 gate at `:1325-1334`. k-last `[B,Hv,V,K]` (`gdn_prefill.cpp:586-590`) matches SGLang. |
| SGLang `MusaFlashInferGDNKernel.extend` reference | CONFIRMED | `/workspace/sglang_qwen35/python/sglang/srt/hardware_backend/musa/attention/linear/kernels/gdn_flashinfer.py:149-202`: packed `[total_seq_len,...]` + `cu_seqlens=query_start_loc`, one `chunk_gated_delta_rule` call, `index_copy_` scatter. The Mate kernel supports cu_seqlens (SGLang passes it); xLLM doesn't expose/pass it. |

### 1.3 Section 3 - hybrid-scheduler (TODO: pending)

| Plan requirement | Status | Evidence |
|---|---|---|
| Replace one-prefill restriction with pure-prefill packing | **MISSING (restriction EXISTS)** | `require_homogeneous_graph_batch()` (`chunked_prefill_scheduler.cpp:99-103`) + single-prefill break (`:151-156, 360-367`) when graph+piecewise on. Comment: "packing N x ~2.5k prefills creates unique ~5k/~7k capture buckets that crash MUSA." |
| Batch up to budget; target 1+2+2 / 1+4 | PARTIAL | Budgets exist: `max_tokens_per_batch`=8192 (bench), `max_seqs_per_batch`=256. **No 16384 budget anywhere.** When piecewise is OFF the loop already admits multiple prefills incidentally. SGLang uses `max_prefill_tokens`=16384 + `PrefillAdder` (`schedule_policy.py:375-829`). |
| Per-batch execution hint (don't infer from token count) | **MISSING** | Only `BatchForwardType` (inferred from sequence stages, `framework/batch/batch.cpp:90-116`) is passed. No eager/varlen hint. |
| Route pure-prefill to eager varlen; decode stays graphed; chunked/piecewise retained | **MISSING** | No eager-varlen route distinct from generic `model_->forward()` over-limit fallback (`cuda_graph_executor_impl.cpp:2441-2461, 2601-2606`). No "varlen" executor backend in `executor_impl_factory.cpp`. Capture failure is `LOG(FATAL)` (`:2526-2528, 2594-2597`), not soft fallback. |
| Multi-request slot_mapping fix prerequisite | EXISTS (already applied) | `cuda_graph_executor_impl.cpp:645-671, 896-916`. 0712 doc's line numbers are stale. |

### 1.4 Section 4 - optimize-prefill-kernels (TODO: pending)

| Plan requirement | Status | Evidence |
|---|---|---|
| Replace `torch::normalize` with specialized in-place GDN L2-norm | **MISSING (generic EXISTS)** | `l2norm_last_dim` (`gdn_prefill.cpp:294-300`) uses `torch::nn::functional::normalize` with fp32 cast; called 2x per layer x 48 layers. Mate 0.2.3 ships `gdn_l2norm.py` (`/workspace/mate_0.2.3/mate/gdn_kernels/tilelang/gdn_l2norm.py`); xLLM does not bind it. |
| Reuse/preallocate GDN padding, KKT, output, final-state buffers | **MISSING** | All allocated fresh per call (`gdn_prefill.cpp:440,568,579,584,588`). `kGdnBufMaxRows=128` cap (`qwen3_gated_delta_net_base.cpp:1030`) disables fused-qkvzba reuse for prefill. Decode path has reusable buffers (`fused_gdn_decode_out_buf_` etc.) - pattern to replicate. |
| Compare conv / full-attention with SGLang (TileLang conv, FA3 varlen) | PROFILING-ONLY | xLLM conv is varlen (`causal_conv1d.cu`, `XLLM_USE_CUSTOM_PREFILL_CONV=1`). xLLM attention uses FlashInfer ragged + `XLLM_USE_FA3=1`. SGLang uses FA3 `flash_attn_varlen_func` (`flashattention_backend.py:7,131-138`). Needs Step-1 timing to justify porting. |

### 1.5 Section 5 - demote ladder padding

| Plan requirement | Status | Evidence |
|---|---|---|
| Keep ladder optional; don't use 2560->2816 jump in default C=5 | Conditional | Ladder EXISTS, identical to SGLang (`generate_piecewise_prefill_graph_tokens`, `cuda_graph_executor_impl.cpp:1976-2000`). 2560->2816 jump = 256 step. No 2624 bucket. No padding-ratio cap. But SGLang has the same ladder - this only matters if xLLM keeps capturing pure-prefill into the graph. If pure-prefill goes eager-varlen (Step 3), ladder padding is bypassed entirely. |

### 1.6 Section 6 - validate-ttft

Depends on Steps 1-4. `correctness_check.sh` + `.tmp_run_official_c5_bench.sh` exist as harnesses; no SGLang pairing. Targets: Phase-1 mean TTFT <=1.65 s; Phase-2 <=1.50 s and within 5-10% of paired SGLang.

---

## 2. Corrections to the source plan's diagnosis

1. **"SGLang disables piecewise prefill graphs and chunking for this workload" is INACCURATE (verified 2026-07-13).**
   SGLang keeps piecewise CUDA graph **ON** for Qwen3.5-27B on MUSA. Two checks
   confirm this:
   - **Model-arch condition** (`model_config.py:213` calls
     `is_piecewise_cuda_graph_disabled_model(self.hf_config.architectures)`):
     the disabled-arch list (`model_config.py:1361-1367`) contains
     `Qwen3NextForCausalLM` but **not** `Qwen3_5ForConditionalGeneration`.
     The model's `config.json` declares `architectures=['Qwen3_5ForConditionalGeneration']`,
     `model_type=qwen3_5` (a distinct SGLang EntryClass, `qwen3_5.py:1756`). The
     `Qwen3NextForCausalLM` -> `Qwen3NextForCausalLMMTP` override at
     `model_config.py:340` is draft-model-only and does not apply.
   - **Hardware condition** (`server_args.py:1111`): MUSA is not in
     `is_hip()/is_npu()/is_cpu()/is_mps()/is_xpu()`.
   - `conc_eval_sglang.sh:30-52` passes **no** `--disable-piecewise-cuda-graph`
     flag, so SGLang runs piecewise ON (its default). (It *does* pass
     `--chunked-prefill-size -1`, which disables per-request chunking, and
     `--disable-overlap-schedule`; but piecewise stays on.)

   What SGLang actually does: GDN and full-attention run as **eager split-ops
   inside the piecewise graph** (projections/norms/MLP are graph-captured;
   GDN/attn run eager between pieces), and it **packs multiple prefills per
   batch** under `max_prefill_tokens`=16384. The plan's "eager varlen prefill"
   intent is correct for the GDN core, but the surrounding projections are still
   graphed in SGLang. Step 3's "initially use eager varlen for all pure-prefill
   batches" is a valid simplification but is coarser than SGLang's actual
   split-op design (see Open Question Q2).

   > NOTE: the sibling plan `prefill_ttft_execution_plan_auto_0713.md` asserts
   > the opposite ("SGLang does not use piecewise / must run with piecewise off")
   > and instructs the paired benchmark to add `--disable-piecewise-cuda-graph`
   > to SGLang. That would measure a **non-default** SGLang config (projections
   -> eager, likely slower) and bias the comparison. Do **not** disable
   > piecewise for the SGLang reference; run `conc_eval_sglang.sh` unchanged.
   > The sibling plan conflates `Qwen3NextForCausalLM` (in the disabled list)
   > with `Qwen3_5ForConditionalGeneration` (our model, not in the list).

2. **The 16384 budget does not exist in xLLM today.** Closest is 8192
   (`max_tokens_for_graph_mode`, `max_tokens_per_chunk_for_prefill`,
   `max_tokens_per_batch` in the bench). Introducing 16384 as a batch prefill
   budget is new work.

3. **The plan's TTFT baseline (2.150 s) differs from the 0712 doc's C=5 result
   (6951 ms mean).** The 0712 run used OSL=1500 (long decodes -> heavy queueing
   -> 9.7 s slow TTFTs). The plan's wave (0.60-3.13 s) implies a
   different/shorter measurement. Confirm which workload the 2.150 s / 1.422 s
   figures come from before benchmarking (Step 0).

---

## 3. Detailed implementation steps

### Step 0 - Confirm baseline and workload (prerequisite, no code)

**Objective:** Reproduce the plan's 2.150 s figure and lock the acceptance
workload before any code changes.

**Actions:**
- Run `.tmp_run_official_c5_bench.sh` once unchanged; capture the per-request
  TTFT wave and confirm whether it matches the plan's 0.60-3.13 s wave or the
  0712 doc's 1.70-9.70 s wave (OSL=1500).
- Determine whether the acceptance workload is OSL=1500 (0712) or the plan's
  shorter one. This determines whether the 16384 budget and 1+2+2 packing even
  apply: with OSL=1500, decode queueing dominates, not prefill packing.
- Capture a paired SGLang run on the same workload for the reference 1.422 s.

**Gate:** Documented baseline numbers (xLLM mean TTFT, per-request wave, SGLang
mean TTFT) that all later steps are measured against.

---

### Step 1 - Add opt-in stage timing (plan section 1, TODO instrument-baseline)

**Objective:** Make every later step's gains measurable; produce paired
xLLM/SGLang budgets.

**1.1 Implement `XLLM_TTFT_WATERFALL` (already specified in
`tools/prefill_ttft_baseline.md` but unimplemented).**

- Add a single env gate: `util::get_bool_env("XLLM_TTFT_WATERFALL", false)`
  (pattern mirrors `s_enable_graph_timing` at
  `cuda_graph_executor_impl.cpp:71-77`).
- Instrument with **one `stream->synchronize()` per profiled forward** (not per
  stage), using the existing wall-clock `Timer` (`xllm/core/util/timer.h`) plus
  MUSA events for GPU-side stages.

**1.2 Scheduler sub-breakdown** (`xllm/core/scheduler/chunked_prefill_scheduler.cpp`):
- Split the existing `scheduling_latency_seconds` observation (`:890`) into
  admission / block-alloc / batch-build sub-stages around `:794-873`
  (`handle_running_queue_requests`, `handle_prefill_requests`,
  `handle_remaining_budget`).

**1.3 Executor timing** (`xllm/core/runtime/cuda_graph_executor_impl.cpp`):
- Add timing to the **piecewise prefill replay branch** (`:1837-1887`) mirroring
  the decode branch's `XLLM_GRAPH_TIMING` logging (`:1934-1957`).
- Add capture begin/end elapsed around `:1622, 1684, 1803`.

**1.4 GDN per-layer timing** (`xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp`,
`xllm/core/kernels/musa/gdn_prefill.cpp`):
- Time projection, conv1d, KKT solve, Mate prefill core, output proj, MLP per
  layer. Aggregate across the 48 GDN layers to avoid 64x log spam (e.g. emit one
  summarized line per forward).

**1.5 Attention timing** (`xllm/core/kernels/musa/attention_runner.cpp`):
- Time each graph segment vs eager attention runner (`run_capture` `:29-67`,
  `run_replay` `:122-172`).

**1.6 Paired A/B benchmark runner:**
- Build a wrapper that launches `.tmp_run_official_c5_bench.sh` (xLLM) and
  `conc_eval_sglang.sh` (SGLang, separate container `sglang-wf`) and emits a
  side-by-side report of: cold capture TTFT, fully warm TTFT, scheduler queue
  time, prefill forward time, logits/sample time, HTTP first-byte delay.

**Gate:** With `XLLM_TTFT_WATERFALL=1`, a single prefill forward logs a
complete stage breakdown; the paired runner produces xLLM-vs-SGLang TTFT budgets
matching the plan's required reporting fields. Default (env off) behavior
unchanged.

---

### Step 2 - Packed varlen Mate GDN prefill (plan section 2, TODO varlen-mate-prefill)

**Objective:** Give the executor a correct packed multi-request prefill path so
Step 3 has something to route to. This is the highest-risk, highest-value step.

**2.1 Verify the Mate kernel's varlen signature.**
- Inspect `/workspace/mate_0.2.3/mate/gdn_prefill.py` (and the cached
  `mate_gdn_prefill_hq16_hv48_bf16` `.so`) to confirm `chunk_gated_delta_rule`
  accepts `cu_seqlens`. SGLang's `extend` (`gdn_flashinfer.py:149-202`) passes
  `cu_seqlens=query_start_loc`, so the kernel supports it; confirm the exact
  argument name/order and whether a separate FFI URI is needed for the varlen
  entry point vs the batched one.

**2.2 Extend the params struct** (`xllm/core/kernels/param.h:1690-1701`):
- Add `std::optional<torch::Tensor> cu_seqlens = std::nullopt;`
- Optionally add `std::optional<torch::Tensor> state_indices` (though the caller
  currently gathers state before the call, so this may stay at the caller level).

**2.3 Add the packed path in `mate_gated_delta_rule_prefill`**
(`xllm/core/kernels/musa/gdn_prefill.cpp:486-662`):
- When `params.cu_seqlens` is set:
  - Operate on packed `[total_seq_len, H, D]` tensors (skip the `[B,T]`
    assumptions at `:492-493`); derive `total_seq_len` from `query.size(0)` and
    `num_requests` from `cu_seqlens.size(0)-1`.
  - Pass `cu_seqlens` to the FFI `run()` call (`:592-600`).
  - Size `output`/`final_state` from `total_seq_len` / `num_requests`
    (`:584,588`).
- Keep the existing 4D `[B,T,H,D]` path as the fallback when `cu_seqlens` is
  unset (B=1 graph path unchanged).

**2.4 Add the packed caller branch** in
`xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp`:
- Add a branch that, when `attn_metadata.is_prefill && batch_size > 1` (gated by
  a flag, e.g. reuse `XLLM_MATE_GDN_PREFILL` or a new
  `XLLM_MATE_GDN_PREFILL_VARLEN`), stays in the **packed (unpadded) layout
  produced by `reshape_qkvz_unpad`** (`:1776`) instead of re-padding via
  `reshape_qkvz_with_pad` (`:987`, `:1833-1874`). `reshape_qkvz_unpad` already
  exists and is already used to build the conv1d input at `:1172` - reuse that
  same flat `[total_tokens, H, D]` tensor for the Mate path.
- Pass `attn_metadata.q_cu_seq_lens` as `cu_seqlens` to the new packed Mate
  path. **Eliminate the per-sequence loop** (`:1502-1572`) for this branch.
- Reuse the existing `index_select`/`index_put_` state gather/scatter
  (`:1340,1369,1509,1571`) - already varlen-compatible; no change needed there.
- conv1d is already varlen (`:1175-1183`) - no change.

**2.5 Correctness gate:**
- Eager B=1 vs packed B=2/B=4 with mixed prompt lengths in the same batch (to
  catch state/padding leakage).
- Per-layer recurrent-state parity using the `LayerHiddenDumper` from the 0712
  bisect (slot-by-slot relative-L2 / cosine).
- Require: no MIXED fallback, no OOM, no "invalid argument", no state
  corruption.

**Gate:** Packed B=2/B=4 prefill bit-matches eager B=1 per-layer state within
tolerance; B=1 path unchanged and still graph-safe.

---

### Step 3 - Hybrid scheduler: pack pure prefills + eager routing (plan section 3, TODO hybrid-scheduler)

**Objective:** Admit multiple pure prefills per step and route them to the
Step-2 eager varlen path, while decode stays graphed.

**3.1 Introduce a batch prefill budget and per-batch execution hint.**
- Add a 16384 batch prefill budget (new option, e.g. `max_prefill_tokens`, or
  raise `max_tokens_per_batch` for prefill-only batches). Wire through
  `runtime/options.h` / `framework/config/scheduler_config.h` and the launch
  script `run_xllm_musa.sh`.
- Add a per-batch **execution hint** distinguishing "packed pure-prefill ->
  eager varlen" from "chunked/piecewise -> graph". Either extend
  `BatchForwardType` (`framework/batch/batch_forward_type.h:21-90`) or add a
  field on `Batch` set by the scheduler (not inferred from token count, per the
  plan).

**3.2 Scheduler packing** (`xllm/core/scheduler/chunked_prefill_scheduler.cpp`):
- When the hint says packed-prefill, **bypass `require_homogeneous_graph_batch()`**
  (`:99-103, 151-156, 360-367`) and admit multiple waiting pure prefills up to
  the 16384 budget and `max_seqs_per_batch`, targeting 1+2+2 / 1+4 grouping
  (mirror SGLang's `PrefillAdder` admission loop).
- Keep continuation chunks and decodes homogeneous: never combine continuation
  chunks with unrelated decode work.

**3.3 Executor routing** (`xllm/core/runtime/cuda_graph_executor_impl.cpp`,
`run()` at `:2415-2827`):
- Add a branch that routes the tagged packed-prefill batch to **eager varlen**
  (`model_->forward` with the Step-2 packed path), at a position parallel to the
  existing prefill branches (`:2437-2529` piecewise, `:2601-2606` eager-off).
- Decode stays on the full graph (`:2609-2709`); chunked/piecewise remain
  available policies (`:2537-2598`).
- Make capture failure a **soft fallback to eager** (not `LOG(FATAL)` at
  `:2526-2528, 2594-2597`) for the new packed path.
- Initially: eager varlen for **all** pure-prefill batches; retain piecewise
  only for continuation chunks and controlled B=1 A/B tests.

**Gate:** Under C=5, the scheduler emits batches with >1 pure prefill (visible
in logs); those batches route to eager varlen; decode batches still hit the
graph. No `MIXED`/OOM/invalid-argument. TTFT wave no longer serializes one
prefill at a time.

**Expected batch composition (validation checkpoint, C=5 ISL=2500):** log the
per-step batch shapes and expect roughly `1 x ~2500` -> `2 x ~2500` -> `2 x
~2500` -> decode at concurrency 5 (the 1+2+2 pattern). Per-request TTFTs in a
wave should compress toward the wall time of the largest prefill batch, not the
old 0.6...3.1 s staircase.

---

### Step 4 - Close residual per-request gaps (plan section 4, TODO optimize-prefill-kernels)

**Objective:** Only after Step 3 measures a remaining gap; profile-driven.

**4.1 Specialized in-place GDN L2-norm:**
- Bind Mate's `gdn_l2norm.py` (`/workspace/mate_0.2.3/mate/gdn_kernels/tilelang/gdn_l2norm.py`)
  as an in-place FFI: add a `mate_gdn_l2norm` URI helper analogous to
  `kkt_solve_mate_ffi` (`gdn_prefill.cpp:421-451`).
- Replace `l2norm_last_dim` (`gdn_prefill.cpp:294-300`) when the varlen/URI is
  available; keep the generic path as fallback.

**4.2 Prefill buffer reuse:**
- Lift the `kGdnBufMaxRows=128` cap (`qwen3_gated_delta_net_base.cpp:1030`) with
  a separate prefill-sized buffer pool for KKT `a` / `output` / `final_state` /
  `h0`, mirroring the decode-path grow-only buffers (`fused_gdn_decode_out_buf_`
  etc., header `:120-153`).
- Add `output_buf` / `final_state_buf` / `a_buf` / `h0_buf` fields to
  `MateGatedDeltaRulePrefillParams`, eliminating the fresh `torch::empty`/`zeros`
  at `gdn_prefill.cpp:440,568,579,584,588`.

**4.3 Conv / attention parity (profile-gated):**
- Using Step-1 timing, compare xLLM causal conv (`causal_conv1d.cu`) and
  full-attention (FlashInfer ragged / `XLLM_USE_FA3=1`) against SGLang's
  TileLang conv and FA3 `flash_attn_varlen_func`
  (`flashattention_backend.py:7,131-138,372-397`). Port only if timing shows a
  material difference.

**4.4 Projection / MLP GEMMs (lowest priority):**
- Measure before changing. Both xLLM and SGLang likely share the same TorchMUSA
  GEMM backend, so these are expected to be a wash and should not be touched
  until post-batching Step-1 numbers prove otherwise.

**Gate:** Step-1 timing shows the targeted stage's share of prefill forward
drops measurably; correctness unchanged.

---

### Step 5 - Ladder padding demotion (plan section 5)

**Objective:** Stop paying the 2560->2816 padding cost in the default C=5 path.

- With Step 3 routing pure-prefill to eager varlen, the 2560->2816 jump
  (`cuda_graph_executor_impl.cpp:1976-2000, 2880-2896`) no longer affects the
  default C=5 path (no capture -> no padding). Keep the ladder for B=1 piecewise
  experiments only.
- If piecewise is retained for another workload: add a 2624 bucket around this
  workload, cap allowed padding ratio (no cap exists today), and do not
  pre-capture many shapes until graph memory is bounded and capture frequency is
  proven to affect production TTFT.

**Gate:** Default C=5 path incurs zero ladder padding; B=1 piecewise path still
works unchanged.

---

### Step 6 - Validate (plan section 6, TODO validate-ttft)

**6.1 Correctness matrix:**
- Eager B=1 vs packed B=2/B=4, mixed prompt lengths in one batch (state/padding
  leakage).
- ISL {512, 2500, 20k} x C {1, 5}.
- Require: no MIXED fallback, no OOM, no "invalid argument", no recurrent-state
  corruption.

**6.2 Official benchmark (3 repetitions, median-of-runs, paired with SGLang):**
- Phase-1 target (after Step 3 varlen batching): **mean TTFT <=1.65 s**, 10/10
  successful, TPOT regression <1 ms.
- Phase-2 target (after Step 4 kernel parity): **mean TTFT <=1.50 s** and within
  5-10% of the paired SGLang run.
- Preserve output throughput and decode TPOT; reject TTFT changes that merely
  shift latency into decode.

**Gate:** Median-of-3 meets Phase-1 then Phase-2 targets; correctness matrix
clean.

---

## 4. Execution order, dependencies, and risk

```
Step 0 (baseline)  ── no code, gates the workload definition
   |
   v
Step 1 (instrument) ── enables measurement for all later steps
   |
   v
Step 2 (packed kernel) ── highest risk/value; correctness-gated
   |   (unblocks)
   v
Step 3 (scheduler/routing) ── low risk once Step 2 passes; mostly control flow
   |
   +-> Step 5 (ladder demotion) ── automatic consequence of Step 3
   |
   v
Step 4 (kernel parity) ── incremental, profile-gated; only if Step-1 timing
                          shows a remaining material gap
   |
   v
Step 6 (validate) ── acceptance gates
```

| Step | Risk | Why |
|---|---|---|
| 0 | low | measurement only |
| 1 | low | additive instrumentation, env-gated, default off |
| 2 | **high** | kernel correctness; recurrent-state parity across packed requests; must not regress B=1 graph path |
| 3 | medium | scheduler policy change; must keep continuation/decode homogeneous; soft-fallback wiring |
| 4 | medium | FFI binding; buffer lifetime under graph capture |
| 5 | low | mostly becomes a no-op after Step 3 |
| 6 | low | validation only |

Recommendation: start Step 0 and Step 1 in parallel (independent). Then Step 2
behind a flag with the existing per-request path as the guarded fallback. Do not
begin Step 3 until Step 2's correctness gate passes. Defer Step 4 until Step-1
timing post-Step-3 shows a remaining gap.

### Suggested PR sequence

| PR | Maps to | Scope | Success signal | Depends on |
|---|---|---|---|---|
| **PR0** | Step 0 + Step 1 | Stage timing (`XLLM_TTFT_WATERFALL`) + paired SGLang script | C=5 wave budget table for xLLM and SGLang; >=70% of the gap attributed | - |
| **PR1** | Step 2 | Packed Mate params + layer path + parity tests | Packed B=2/B=4 numerically matches sequential B=1; no NaN/invalid/state corruption | PR0 (for measurement) |
| **PR2** | Step 3 (+ Step 5) | Scheduler packing + executor eager hint + ladder demotion | Wave TTFTs flatten; mean TTFT <=1.65 s; 10/10; TPOT delta <1 ms | PR1 |
| **PR3** | Step 4 | Profile-guided L2norm / buffer reuse (optional conv/attn ports) | mean TTFT <=1.50 s; within 5-10% of paired SGLang | PR2 + Step-1 timing |

PR0 and PR1 can be drafted in parallel; PR1 is the correctness-gated long pole.
Do not ship PR2 without PR1 green.

---

## 5. Key file:line reference index

### Scheduler (`xllm/core/scheduler/`)
- `chunked_prefill_scheduler.cpp:99-103` - `require_homogeneous_graph_batch()`
- `:151-156` - one-prefill break in `handle_running_queue_requests`
- `:360-367` - `require_homogeneous_batch` + break in `handle_prefill_requests`
- `:794-873` - `prepare_batch` steps (running/prefill/remaining)
- `:890` - `scheduling_latency_seconds` observation
- `:758-774` - budget reads (`max_tokens_per_chunk_for_prefill`, `max_tokens_per_batch`, `max_seqs_per_batch`)

### Executor (`xllm/core/runtime/`)
- `cuda_graph_executor_impl.cpp:2415-2827` - `run()` dispatch
- `:2420-2434` - batch type + bucket selection
- `:2437-2529` - piecewise prefill graph (capture/replay), FATAL on failure `:2526-2528`
- `:2441-2461` - eager fallback (over-limit)
- `:2537-2598` - chunked prefill piecewise graph, FATAL on failure `:2594-2597`
- `:2601-2606` - prefill eager (piecewise off)
- `:2609-2709` - decode full CUDA graph
- `:2808-2819` - mixed eager fallback
- `:645-671` - multi-request `q_seq_lens_vec`/`kv_seq_lens_vec` padding fix
- `:896-916` - `q_cu_seq_lens`/`kv_cu_seq_lens` last-element override
- `:1976-2000` - `generate_piecewise_prefill_graph_tokens` (SGLang-identical ladder)
- `:2880-2896` - `get_bucket_num_tokens` (`lower_bound` pad-up)
- `:1837-1887` - piecewise prefill replay branch (NO timing today)
- `:1934-1957` - decode replay timing (`XLLM_GRAPH_TIMING=1`)
- `:71-77` - `s_enable_graph_timing` env gate (pattern to mirror)
- `executor_impl_factory.cpp` - no "varlen" backend registered

### GDN prefill (`xllm/core/layers/musa/`, `xllm/core/kernels/musa/`)
- `qwen3_gated_delta_net_base.cpp:981-991` - `project_padded_inputs` -> `reshape_qkvz_with_pad`
- `:1833-1874` - `reshape_qkvz_with_pad` (pads to `max_len`, stacks)
- `:1502-1572` - per-sequence PyTorch fallback loop (MUSA ignores cu_seqlens)
- `:1325-1334` - `mate_prefill_shape_supported` / `use_mate_gdn_prefill` (B=1 graph-safe gate)
- `:1175-1183` - conv1d varlen call (`query_start_loc=q_cu_seq_lens`)
- `:1340,1369,1509,1571` - state gather/scatter (`index_select`/`index_put_`)
- `:1030` - `kGdnBufMaxRows=128` cap (disables prefill buffer reuse)
- `:1607-1633, 1657-1692` - decode reusable buffers (pattern to replicate)
- `gdn_prefill.cpp:486-662` - `mate_gated_delta_rule_prefill`
- `:492-493` - 4D `[B,T,H,D]` enforcement
- `:592-600` - FFI `run()` call (no cu_seqlens passed)
- `:294-300` - `l2norm_last_dim` (generic `torch::normalize` + fp32 cast)
- `:517-520` - l2norm call sites (q and k)
- `:440,568,579,584,588` - fresh `torch::empty`/`zeros` allocations per call
- `:421-451` - `kkt_solve_mate_ffi` (FFI URI helper pattern)
- `param.h:1690-1701` - `MateGatedDeltaRulePrefillParams` (no cu_seqlens)
- `param.h:1661-1687` - `ChunkGatedDeltaRuleParams` (has cu_seqlens at `:1682`, dead on MUSA)
- `causal_conv1d.cu:189,241` - varlen conv1d (already supports `query_start_loc`)

### Timing / metrics (`xllm/core/util/`, `xllm/core/common/`)
- `timer.h:23-38` - `Timer` (CPU wall-clock)
- `metrics.h:65,79` - `COUNTER_ADD` / `HISTOGRAM_OBSERVE`
- `metrics.cpp:109` - `scheduling_latency_seconds`
- `metrics.cpp:78-79` - `execution_latency_seconds_model`
- `metrics.cpp:123-124` - `time_to_first_token_latency_milliseconds`
- `continuous_scheduler.cpp:1360-1364` - TTFT histogram observation

### Benchmark / SGLang
- `.tmp_run_official_c5_bench.sh` - xLLM-only C=5 bench (no SGLang)
- `conc_eval_sglang.sh:30-52` - SGLang mirror launch: piecewise ON (no `--disable-piecewise-cuda-graph`), `--chunked-prefill-size -1` (chunking OFF), `--disable-overlap-schedule`, decode graph `--cuda-graph-max-bs 32`; separate container `sglang-wf`
- `tools/prefill_ttft_baseline.md:15,32,41-45` - `XLLM_TTFT_WATERFALL` spec (unimplemented)
- `/workspace/sglang_qwen35/python/sglang/srt/hardware_backend/musa/attention/linear/kernels/gdn_flashinfer.py:149-202` - `MusaFlashInferGDNKernel.extend` (packed varlen reference)
- `/workspace/sglang_qwen35/python/sglang/srt/configs/model_config.py:1361-1367,1417-1421` - `piecewise_cuda_graph_disabled_model_archs` (has `Qwen3NextForCausalLM`, NOT `Qwen3_5ForConditionalGeneration`)
- `/workspace/sglang_qwen35/python/sglang/srt/configs/model_config.py:213,340-341` - PCG disable call site + draft-only `Qwen3NextForCausalLM` override
- `/workspace/sglang_qwen35/python/sglang/srt/server_args.py:1111` - MUSA excluded from PCG hardware auto-disable
- `/workspace/sglang_qwen35/python/sglang/srt/server_args.py:1407-1425` - `_generate_piecewise_cuda_graph_tokens` (identical ladder)
- `/workspace/sglang_qwen35/python/sglang/srt/managers/schedule_policy.py:375-829` - `PrefillAdder` (1+2+2/1+4 packing)
- `/workspace/sglang_qwen35/python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py:7,131-138` - FA3 `flash_attn_varlen_func`
- `/workspace/mate_0.2.3/mate/gdn_kernels/tilelang/gdn_l2norm.py` - specialized GDN L2-norm (not bound by xLLM)

---

## 6. Open questions

**Q1 (Step 0):** Which workload produced the plan's 2.150 s / 1.422 s figures?
The 0712 doc's C=5 run (OSL=1500) gave 6951 ms mean TTFT with a 1.70-9.70 s wave.
If the acceptance workload is OSL=1500, decode queueing dominates and prefill
packing (Step 3) may not move mean TTFT much; the plan's wave (0.60-3.13 s)
implies a shorter OSL. This must be settled before benchmarking. NOTE
(verified 2026-07-13): the cited baseline log `20260712_221224` does **not
exist** in `logs/` (0712 logs end at `conc_eval_20260712_205738.log`), and the
staircase `0.60/1.24/1.87/2.51/3.13` averages to **1.87 s**, not the stated
2.15 s - the baseline figures are internally inconsistent and unverified. Step 0
must re-derive the baseline from a fresh run.

**Q2 (Step 3):** The plan says "initially use eager varlen for all pure-prefill
batches". SGLang actually keeps projections/norms/MLP graph-captured and only
runs GDN+attention as eager split-ops inside the piecewise graph. Should xLLM
match SGLang's split-op design (graph the projections, eager only GDN+attn), or
is full-eager-prefill simpler and sufficient for Phase-1 (<=1.65 s)? Full-eager
is lower implementation risk but may leave projection-graph-capture gains on the
table for Phase-2 (<=1.50 s).

**Q3 (Step 2):** Does the cached Mate `.so` (`mate_gdn_prefill_hq16_hv48_bf16`)
expose the varlen `cu_seqlens` signature that SGLang uses, or does xLLM's cached
copy predate it? Verify before assuming the existing FFI URI works for varlen; a
re-cache against Mate 0.2.3 may be needed.

> **UPDATE (2026-07-13 later):** Varlen C ABI `.so` is built and **bit-matches**
> JIT once `g` is preprocessed with `chunk_local_cumsum`. The prior "C ABI vs JIT
> mismatch" was a test bug (raw alpha vs cumsummed log-decay). Q3 is closed.
> Remaining: finish xLLM caller wiring + B=2/B=4 layer parity, then PR2.

**Q4 (Step 4.2):** Persistent prefill buffers conflict with graph capture
(captured graphs pin tensor addresses). If Step 3 routes pure-prefill to eager
(not captured), buffer reuse is safe. If any prefill stays piecewise-captured,
buffer reuse must be capture-aware (like the decode path's grow-only model).
Confirm the routing decision before implementing 4.2.

---

## 7. Cross-check against `prefill_ttft_execution_plan_auto_0713.md`

Compared the sibling auto plan point-by-point. Summary:

| Area | Auto plan | This plan | Resolution |
|---|---|---|---|
| SGLang piecewise for Qwen3.5 | asserts OFF ("must run with piecewise off") | verified ON (arch `Qwen3_5ForConditionalGeneration` not in disabled list; `conc_eval_sglang.sh` passes no disable flag) | **This plan is correct.** Do not disable piecewise for the SGLang reference (would measure a non-default, slower config). See Correction #1. |
| Baseline 2.15 s / staircase | accepts at face value | flags as unverified (cited log `20260712_221224` missing; staircase mean 1.87 s != 2.15 s) | **This plan is correct.** Re-derive baseline in Step 0. |
| Priority order | instrument -> packed Mate -> scheduler -> kernels | identical | agree |
| Critical path / risk | Phase 1 (packed Mate) highest risk | identical (Step 2) | agree |
| PR packaging | explicit PR0-PR4 table | adopted | **improved this plan** (added PR sequence table) |
| Non-goals / anti-patterns | explicit section | adopted | **improved this plan** (added non-goals below) |
| Expected batch composition | `1x~2500 / 2x~2500 / 2x~2500 / decode@5` | adopted | **improved this plan** (added to Step 3 gate) |
| Packed-tensor reference | `reshape_qkvz_unpad` (precise) | adopted | **improved this plan** (Step 2.4 now references `:1776`) |
| GEMM priority | lowest priority (shared TorchMUSA) | adopted | **improved this plan** (added Step 4.4) |
| Env var name | `XLLM_TTFT_STAGE_TIMING` (invented) | `XLLM_TTFT_WATERFALL` (already specified in `tools/prefill_ttft_baseline.md:15,32`) | keep WATERFALL (grounded in existing spec); avoid two parallel mechanisms |
| Full-eager vs split-op (Q2) | chooses full-eager for all pure prefill | frames as open question | keep as Q2; full-eager is lower-risk for Phase-1, revisit split-op if Phase-2 (<=1.50 s) needs the projection-graph gains |

---

## 8. PR1 progress log (Step 2 - packed Mate GDN prefill)

> Updated: 2026-07-13 (host-cu fix re-bench: mean TTFT 3489 ms, residual gap vs padded)

### 8.1 Summary

PR1 Half A is **functionally green** (FFI parity exact; `correctness_check`
PASS with packed on; capture-safe B=1 path). Phase-1 target (<=1.65 s)
**not met**.

| Run | Config | Mean TTFT | Median | Notes |
|---|---|---|---|---|
| `_var0` | packed + padded Mate | **2132 ms** | 2728 | baseline |
| `_var1` | packed + varlen Mate (D2H) | 3847 ms | 3178 | +1715 ms regression |
| `_var2` | packed + varlen + host-cu fix | **3489 ms** | 4527 | recovered ~360 ms; residual −1.36 s vs `_var0` |

**Root cause / fix**: §8.9 identified 384 D2H syncs/forward; §8.10 landed host
`cu_seqlens` + reshape pack + vectorized cumsum. Re-bench (§8.11) shows D2H
was real but not the whole gap — warmup C=2 (~2253 ms) is near `_var0`, so
residual is multi-req packed forward + queue. Next: Option A A/B (padded Mate
under packed scheduling).

### 8.2 What was done

1. **Mate kernel choice**: Selected the **simple kernel**
   (`gdn_prefill_simple.py`) over the warp-specialized kernel. Rationale:
   - Exact varlen parity in JIT (max_diff=0.0 across all test cases).
   - Compatible `T.alloc_barrier` API (warp-specialized crashes on
     `tilelang.language.create_list_of_mbarrier`).
   - No NaN for partial chunks; computes WY inverse in-kernel (eliminates KKT
     solve).
   - The `a` argument remains in the `call()` signature but is unused by the
     device kernel.

2. **`.mu` generation** (`scripts/gen_gdn_prefill_mu.py`): Generates the varlen
   simple kernel `.mu` via `tilelang.compile(execution_backend="cython")` with
   monkeypatched `subprocess.run` to capture the source without compiling
   (avoids mcc crashes on aggressive `-misched` flags). Output: 13-arg `call()`
   signature with `cu_seqlens`, no stride parameters.

3. **Two-stage build** (`scripts/build_gdn_prefill_varlen_op.py`): Patches the
   `.mu` to fix the `__mt_bfloat16_raw` union member-order issue (see 8.4),
   compiles with mcc 5.1 (`-O3`, no crashing `-misched` flags), builds the FFI
   bridge `.so`, deploys both to cached ops dir.

4. **FFI bridge update** (`csrc/integrations/gdn/mate_gdn_prefill_ffi.cpp`):
   Updated `CallFn` to 13-arg signature (no strides, with `cu_seqlens`), added
   `cu_seqlens` parameter to `MateGdnPrefillRun`, `raw_batch_size =
   cu_seqlens.size(0)-1`.

5. **Deployed artifacts** to
   `/workspace/mate_cached_ops/mate_gdn_prefill_hq16_hv48_bf16/`:
   - `kernel_lib_cabi.so` (155784 bytes, varlen C ABI, 13-arg `call()`)
   - `mate_gdn_prefill_hq16_hv48_bf16.so` (122904 bytes, FFI bridge)

6. **JIT parity test** (`test_varlen_parity.py`): PASSED - simple kernel varlen
   JIT matches non-varlen JIT exactly (max_diff=0.0).

7. **FFI parity test** (`test_ffi_varlen_parity.py`): **PASS** after cumsum
   precondition fix (was false FAIL).

8. **xLLM wiring** (`param.h`, `gdn_prefill.cpp`, `qwen3_gated_delta_net_base.cpp`):
   varlen ABI + log-space chunk cumsum + pack/unpack. Binary rebuilt
   2026-07-13 12:39.

9. **`correctness_check.sh`**: **PASSED** with `ENABLE_PACKED_PREFILL=1` (all
   4 checks green: `run1_http`, `run1_not_garbage`, `run1_has_tokens`,
   `expected_substring`; answer: 17×23=391). Binary rebuilt 2026-07-13 ~12:57.

### 8.3 The numerical mismatch — RESOLVED (false diagnosis)

**Symptom (was)**: C ABI vs JIT diffs grew with T (7e-3 at T=64 → 1e12 at T=2500).

**Actual root cause**: `test_ffi_varlen_parity.py` passed **raw alpha** to the
FFI/`call()`, but the simple kernel requires `g = chunk_local_cumsum(alpha)`
(log then per-chunk cumsum). The Python `mate.gdn_prefill.chunk_gated_delta_rule`
wrapper always applies that preprocess before JIT; the FFI test did not.
Missing cumsum → recurrent state diverges exponentially — looks like a
compile-backend bug, but is not.

**Fix**: apply `chunk_local_cumsum` (or log-space cumsum when g is already
`log(alpha)`, as in xLLM `fused_gdn_gating`) before the FFI call.

**Verified**: with cumsum, `test_ffi_varlen_parity.py` → **ALL PASS**, including
`[2500]×5`, every `o_diff=ht_diff=0.0`.

The `tvm_ffi` vs `cython` codegen difference below is **no longer the blocker**
(kept for archaeology):

| Aspect | `tvm_ffi` (JIT) | `cython` (C ABI) |
|--------|-----------------|-------------------|
| `enable_device_compile` | `True` | `False` |
| Device codegen | `device_codegen()` -> `target.build.tilelang_musa` (generates source + compiles to mubin) | `device_codegen_without_compile()` -> `target.build.tilelang_musa_without_compile` (generates source only) |
| mcc invocation | `mcc --cuda-device-only --cuda-gpu-arch=mp_31` (device-only, produces mubin) | `mcc --shared -fPIC --offload-arch=mp_31` (host+device, produces .so) |
| Host wrapper | TVM host codegen (`LowerTVMBuiltin`, `LowerCustomDatatypes`, etc.) | `TLWrapper.wrap()` (standalone C++ with `call/init/get_last_error`) |

The device kernel source may differ between `tilelang_musa` and
`tilelang_musa_without_compile` codegen passes. Even if the source is identical,
the mcc compilation mode differs (`--cuda-device-only` vs `--shared
--offload-arch`), which could produce different GPU code.

**What was ruled out**:
- **Compile flags** (`-O3` vs `-Od3`): produces identical results for C ABI.
- **`use_initial_state` flag**: JIT with True vs False produces identical
  results (h0=zeros makes the flag a no-op).
- **`real_batch_size=0` vs `1`**: both are `T.dynamic("raw_batch_size")`,
  always dynamic.
- **Tensor strides**: 4D `[1,T,H,D]` unsqueeze is contiguous and shares the same
  data pointer as 3D `[T,H,D]`.
- **bfloat16 `.mu` patching**: The designated-initializer patch
  (`__mt_bfloat16_raw{.data=...}`) is semantically equivalent to the original
  aggregate init; it only fixes the host-side union member order.

**Attempted to verify**: Direct comparison of device kernel source between
`device_codegen()` and `device_codegen_without_compile()` was blocked by TVM
target context setup issues (`Target context required` error when calling
`tilelang.lower()` directly from Python).

### 8.4 The bfloat16 type mismatch in mcc

`__mt_bfloat16_raw` (in `/usr/local/musa/include/musa_bf16.hpp:100-112`) is a
union with different member order depending on compilation context:

```cpp
// When __MUSA_ARCH__ defined (device compilation):
struct __mt_bfloat16_raw { __bf16 data; unsigned short x; };

// When __MUSA_ARCH__ not defined (host compilation):
struct __mt_bfloat16_raw { unsigned short x; __bf16 data; };
```

The tilelang codegen emits aggregate initialization like
`__mt_bfloat16_raw{v__16[0]}` which works on device (first member is `data`)
but fails on host (first member is `x`, an `unsigned short`). Fix: patch `.mu`
with designated initializers: `__mt_bfloat16_raw{.data=v__16[0]}` (4
occurrences).

### 8.5 Key files

| File | Description |
|------|-------------|
| `scripts/gen_gdn_prefill_mu.py` | Generates varlen simple kernel `.mu` |
| `scripts/build_gdn_prefill_varlen_op.py` | Two-stage build: patch `.mu` + mcc compile + FFI bridge + deploy |
| `csrc/integrations/gdn/mate_gdn_prefill_ffi.cpp` | Updated FFI bridge (13-arg CallFn, cu_seqlens) |
| `test_ffi_varlen_parity.py` | FFI parity test (PASS w/ cumsum) |
| `test_varlen_parity.py` | JIT parity test (PASSED) |
| `bench_varlen_speedup.py` | Varlen vs padded benchmark |
| `mate/gdn_kernels/tilelang/gdn_prefill_simple.py` | Simple kernel source |

Cached ops dir: `/workspace/mate_cached_ops/mate_gdn_prefill_hq16_hv48_bf16/`
(4 files: `kernel_lib.so`, `kernel_lib_cabi.so` [NEW varlen],
`kernel_lib_full.so`, `mate_gdn_prefill_hq16_hv48_bf16.so` [NEW FFI bridge]).

### 8.6 Next steps (updated)

1. ~~C ABI / JIT numerical mismatch~~ **DONE**.
2. ~~xLLM wiring (eager path)~~ **DONE**.
3. ~~Capture safety (B=1)~~ **DONE**.
4. ~~`correctness_check.sh`~~ **PASSED**.
5. ~~Official C=5 re-bench~~ **DONE** - TTFT regression (see §8.8).
6. ~~Root-cause the TTFT regression~~ **DONE** (see §8.9).
7. ~~Host-side cu_seqlens + pack/cumsum fix~~ **DONE** (see §8.10).
8. ~~Re-bench after host-cu fix~~ **DONE** — mean **3489 ms** (see §8.11);
   recovered ~360 ms vs broken varlen, still −1.36 s vs padded baseline.
9. **Option A A/B**: run packed scheduling with padded Mate (no varlen pack/
   cumsum) to isolate remaining gap vs `_var0`.
10. **Correctness matrix** B=2/B=4 mixed lengths (still open).
11. **PR0 instrumentation** if Phase-1 (<=1.65 s) still missed after Option A.

### 8.7 correctness_check crash log and resolution (2026-07-13)

#### Crash #1 (binary 12:26) - `.item()` D2H during capture

```
MUSA error: operation not permitted when stream is capturing
  at::musa::LocalScalarDense_ / at::Tensor::item<int>()
  in mate_gated_delta_rule_prefill (cu_seqlens.index({-1}).item())
```

**Fix**: removed `.item()` check; single-seq cumsum via torch `view+cumsum`
(no D2H).

#### Crash #2 (binary 12:39) - `torch::tensor({0,T}, device)` H2D copy

```
MUSA error: operation not permitted when stream is capturing
  at::musa::MUSACopyFrom / at::native::_to_copy / at::Tensor::to(...)
  in mate_gated_delta_rule_prefill during piecewise capture
```

Root cause: `torch::tensor({0, num_tokens}, options.device(query.device()))`
at `gdn_prefill.cpp:653` performs an internal H2D copy (host `int` data ->
device tensor), which is capture-unsafe.

**Fix** (`gdn_prefill.cpp:655-657`): replaced with capture-safe pure kernel
launches:

```cpp
cu_seqlens = torch::zeros(
    {2}, torch::TensorOptions().dtype(torch::kInt32).device(query.device()));
cu_seqlens.select(0, 1).fill_(static_cast<int64_t>(num_tokens));
```

`torch::zeros` allocates + zero-fills on device (kernel launch);
`.fill_()` is a fill kernel. No H2D/D2H sync. Capture-safe.

#### Result (binary ~12:57) - correctness_check PASSES

```
ENABLE_PACKED_PREFILL=1 START_SERVER=1 bash correctness_check.sh
==> server ready after 60s
[run1] ok=True pt=27 ct=256 lat=12.6s
  answer: ... 391 ...
######## CHECKS ########
  [PASS] run1_http
  [PASS] run1_not_garbage
  [PASS] run1_has_tokens
  [PASS] expected_substring
CORRECTNESS: PASS
```

**Capture-safety note**: the multi-seq branch (`num_seqs > 1 &&
pad_size > 0`, `gdn_prefill.cpp:658-662`) uses `.to(torch::kCPU)` (D2H
sync). This is **safe** because when `enable_packed_prefill=true`, all
prefill batches route to **eager** `model_->forward()`
(`cuda_graph_executor_impl.cpp:2439-2440`: `!enable_packed_prefill_`
skips the piecewise graph capture/replay block). D2H sync is allowed in
eager mode. Graph capture only hits the B=1 path (`num_seqs == 1`,
lines 654-657), which is already capture-safe (`torch::zeros + fill_`).


### 8.8 Official C=5 re-bench (2026-07-13 13:22)

**Harness**: `.tmp_run_official_c5_bench.sh` — C=5, ISL=2500, OSL=1500,
warmup=2, measure=10; `ENABLE_PACKED_PREFILL=1`, `MAX_TOKENS_PER_BATCH=16384`,
graph/chunk=8192. Binary stamped `2026-07-13 12:59:11`.

**Results dir**:
`/workspace/bench_results/official_c5_isl2500_osl1500_20260713_132219`

| Metric | Prior packed (10:26, padded Mate `_var0`) | This run (13:26, varlen Mate `_var1`) |
|---|---|---|
| Mean TTFT | **2132 ms** | **3847 ms** |
| Median TTFT | 2728 ms | 3178 ms |
| P90 TTFT | — | 5169 ms |
| Mean TPOT | 55.1 ms | 54.3 ms |
| Successful | 10/10 | 10/10 |

**Server-side TTFT wave (ms, measure)**:
`2939, 2940, 2940, 4733, 4741, 2964, 2964, 2964, 4937, 4938`
→ clear **3+2 / 3+2** packing groups; later pair in each wave pays extra
queue behind long OSL=1500 decodes.

**Health checks**: Falling back to eager=0; MIXED=0; no piecewise prefill
captures (decode graphs only: buckets 1/2/3/4/5/8); SERVER_STILL_UP;
OFFICIAL_BENCH_DONE.

**Verdict**: Functional packing + varlen Mate path works, but **Phase-1
TTFT target (<=1.65 s) is not met**; mean TTFT **regressed ~1.7 s** vs the
padded-Mate packed baseline. Optimization focus shifts from "wire varlen"
to "why is packed varlen prefill slower / more queued than padded packed".

### 8.9 Root-cause analysis: varlen Mate TTFT regression

> Analyzed: 2026-07-13

#### Problem

`_var1` (varlen Mate, 3847 ms) regresses +1715 ms vs `_var0` (padded Mate,
2132 ms). Both runs use `ENABLE_PACKED_PREFILL=1` (eager prefill, packed
scheduling, decode on graph). TPOT is unchanged (54.3 vs 55.1 ms), so the
regression is entirely in the prefill forward path.

The `_var0` baseline (padded Mate) passes `[B, max_len, H, D]` tensors
directly to the Mate FFI kernel with no cu_seqlens, no packing, and no
per-layer host-side processing. The `_var1` path (varlen Mate) introduces
cu_seqlens, pack/unpack, chunk-padding, and log-space chunk cumsum - all
executed **per GDN layer** (48 layers).

#### Root cause: 384 D2H syncs per forward

The multi-seq varlen path in `gdn_prefill.cpp` performs **8 D2H
synchronizations per layer**. With 48 GDN layers, this is **384 D2H syncs
per forward pass**. Each `tensor.to(torch::kCPU)` forces the CPU to wait
for all queued GPU kernels to complete (stream drain), copies data D2H,
then the CPU resumes launching kernels. This serializes CPU and GPU
execution, preventing kernel pipelining and leaving the GPU idle during
each sync.

Per-layer D2H sync inventory (all in `gdn_prefill.cpp`, multi-seq path
where `num_seqs > 1`):

| # | Call site | Function | D2H location | Why |
|---|-----------|----------|--------------|-----|
| 1 | `:630` | `pack_time_dim_4d(query, cu_seqlens)` | `:352` | Read cu_seqlens to narrow each seq |
| 2 | `:631` | `pack_time_dim_4d(key, cu_seqlens)` | `:352` | Same |
| 3 | `:632` | `pack_time_dim_4d(value, cu_seqlens)` | `:352` | Same |
| 4 | `:659` | padding adjust | `:659` | Read cu_seqlens to bump last element by pad_size |
| 5 | `:678` | `pack_time_dim_3d(beta, pack_cu)` | `:370` | Read cu_seqlens to narrow each seq |
| 6 | `:679` | `pack_time_dim_3d(g_log, pack_cu)` | `:370` | Same |
| 7 | `:687` | `chunk_local_cumsum_log_space` | `:330` | Read cu_seqlens to iterate seq/chunk boundaries |
| 8 | `:763` | `unpack_time_dim_4d(output, unpack_cu)` | `:390` | Read cu_seqlens to scatter back to padded layout |

All 8 syncs copy the **same cu_seqlens tensor** (identical content every
time - it doesn't change across layers). The syncs are pure overhead: the
cu_seqlens is known at batch construction time and is invariant across the
48 GDN layers within a single forward.

#### Secondary: CPU for-loops prevent kernel pipelining

Even without D2H syncs, several functions use CPU for-loops that iterate
over sequences or chunks, launching small kernels per iteration. The CPU
cannot run ahead of the GPU, so kernels cannot overlap:

| Function | Location | Loop iterations | Per-iteration cost |
|----------|----------|-----------------|-------------------|
| `pack_time_dim_4d` | `:358-364` | num_seqs (5) | `select` + `narrow` (view, no copy) |
| `pack_time_dim_3d` | `:376-382` | num_seqs (5) | Same |
| `unpack_time_dim_4d` | `:393+` | num_seqs (5) | `slice` + `copy_` |
| `chunk_local_cumsum_log_space` | `:334-345` | ~200 (5 seqs x ~40 chunks) | `slice` + `cumsum` kernel launch per chunk |

The `chunk_local_cumsum_log_space` multi-seq path is the worst: ~200
individual `cumsum` kernel launches per layer x 48 layers = **~9600 small
kernel launches**. Each launch has ~5-10 us CPU overhead. The single-seq
path (`:318-325`) avoids this with a vectorized `view+cumsum+reshape` -
but the multi-seq path cannot use it because chunk boundaries may not
align with sequence boundaries.

#### Why _var0 (padded Mate) doesn't have this overhead

The `_var0` path passes `[B, max_len, H, D]` directly to the Mate FFI
kernel. No cu_seqlens is set, so `gdn_prefill.cpp` takes the `else` branch
at `:635-638` (requires `input_batch == 1` for the non-varlen path). The
kernel processes padding tokens (wasted FLOPs), but:

- **Zero D2H syncs** (no pack/unpack/cumsum host-side processing)
- **Zero CPU for-loops** (no per-seq or per-chunk iteration)
- **Full kernel pipelining** (CPU launches one large kernel, GPU runs it
  asynchronously while CPU proceeds to the next layer's projections)

With ISL=2500 and 5 sequences of ~2500 tokens each, max_len ~= 2500, so
padding waste is minimal (~0-5%). The padded path processes ~12500 tokens;
the varlen path processes ~12500 tokens. The token count is nearly
identical - the regression is **entirely host-side overhead**, not kernel
compute.

#### Estimated impact breakdown

| Source | Per-layer cost | x48 layers | Estimated total |
|--------|---------------|------------|-----------------|
| 8 D2H stream syncs (pipeline stalls) | ~2-5 ms each x 8 = 16-40 ms | x48 | ~770-1920 ms |
| `chunk_local_cumsum` ~200 small kernel launches | ~1-2 ms | x48 | ~48-96 ms |
| pack/unpack CPU loops + `cat`/`copy_` | ~0.5-1 ms | x48 | ~24-48 ms |
| **Total estimated overhead** | | | **~840-2060 ms** |

Observed regression: **1715 ms** - within the estimated range.

Note: D2H sync cost is conservative. Each sync doesn't just cost the copy
time (~10-50 us); it costs the **full pipeline bubble**: all pending GPU
kernels must drain, then the CPU processes the data and re-launches
kernels. With 48 layers of projections/norms/MLP queued between GDN calls,
each sync can stall the pipeline for 5-20 ms.

#### Fix options (ranked by impact, lowest risk first)

**Option A: Revert to padded Mate (recover 2132 ms baseline immediately)**

- Keep `ENABLE_PACKED_PREFILL=1` (packed scheduling, eager prefill).
- Don't set `mate_params.cu_seqlens` (leave it nullopt).
- `gdn_prefill.cpp` takes the non-varlen path (no pack/unpack/cumsum).
- Mate kernel processes `[B, max_len, H, D]` with minor padding waste.
- **Pro**: zero per-layer overhead, recovers _var0 baseline immediately.
- **Con**: processes padding tokens; doesn't use the varlen kernel we
  built. But padding is ~0-5% with ISL=2500 x 5 seqs.
- **Risk**: very low (code change is one line: don't set cu_seqlens).

**Option B: Cache cu_seqlens on CPU once per forward**

- In `qwen3_gated_delta_net_base.cpp`, before the 48-layer loop, do ONE
  D2H copy: `cu_seqlens_cpu = attn_metadata.q_cu_seq_lens.to(kCPU)`.
- Pass `cu_seqlens_cpu` (CPU tensor) down to each layer's
  `mate_gated_delta_rule_prefill`.
- In `gdn_prefill.cpp`, accept a CPU cu_seqlens and use it directly in
  pack/unpack/cumsum (no `.to(kCPU)` per call).
- **Pro**: eliminates 383 of 384 D2H syncs.
- **Con**: still has CPU for-loops and ~200 small cumsum kernel launches.
- **Risk**: low (plumbing change, no kernel change).

**Option C: Vectorize chunk_local_cumsum on GPU**

- Replace the multi-seq CPU for-loop (`:334-345`) with a single GPU op:
  1. Create a `[1, T, 1]` "reset mask" marking the first token of each
     chunk within each sequence (1 = reset cumsum, 0 = accumulate).
  2. Use the mask to zero out cumsum contributions at chunk boundaries.
  3. Single `view({1, num_chunks, chunk_size, H}).cumsum(2).reshape(...)` 
     like the single-seq path, plus mask application.
- **Pro**: eliminates ~9600 small kernel launches.
- **Con**: complex; must handle partial chunks at sequence boundaries.
- **Risk**: medium (correctness-sensitive cumsum logic).

**Option D: Precompute packed tensors + cumsummed g once per forward**

- Since cu_seqlens is invariant across layers, pack q/k/v/g/beta once at
  the batch level (before the layer loop) and reuse across all 48 layers.
- Each layer just does: reshape projections -> pre-packed view -> Mate FFI
  call -> unpack output.
- **Pro**: eliminates all per-layer pack/unpack/cumsum overhead.
- **Con**: large refactor; projections produce different q/k/v per layer,
  so only the pack indices/cumsum can be precomputed, not the actual
  tensors.
- **Risk**: medium-high (changes the layer interface).

**Recommendation (superseded by §8.10 / §8.11)**: Implemented **Option B +
C-lite**; re-bench showed residual −1.36 s vs `_var0`. Next gate is
**Option A A/B** (padded Mate under packed scheduling).


### 8.10 Fix: eliminate per-layer D2H + reduce launch storm (2026-07-13)

**Landed in**:
- `xllm/core/kernels/param.h` — `MateGatedDeltaRulePrefillParams::cu_seqlens_host`
- `xllm/core/kernels/musa/gdn_prefill.cpp` — host-length pack/unpack/cumsum
- `xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp` — build host cu from
  `q_seq_lens_vec` each Mate prefill call

**What changed**:

1. **Option B (primary)**: pack / unpack / chunk-pad / cumsum use host
   `cu_seqlens` (`std::vector<int32_t>`). Device `cu_seqlens` is uploaded
   once per layer for the Mate FFI only (H2D, no stream drain). B=1 capture
   path still uses `zeros+fill_` (no D2H).
2. **Equal-length fast path**: when every seq length == `max_T`, pack/unpack
   are `reshape` views (no `cat` / `copy_`). Official C=5 ISL is homogeneous,
   so this removes the pack memcpy storm.
3. **Option C (lite)**: multi-seq `chunk_local_cumsum` does one vectorized
   `view+cumsum` per sequence (plus a remainder slice) instead of
   ~ceil(L/64) launches per seq.

**Binary**: `build/lib.linux-x86_64-cpython-310/xllm/xllm` stamped
`2026-07-13 13:54:38`.

**Result** (§8.11): recovered **~360 ms** of +1715 ms; mean still
**3489 ms** (−1357 ms vs `_var0`). D2H was necessary but not sufficient.


### 8.11 Official C=5 after host-cu fix (2026-07-13 13:57)

**Results dir**:
`/workspace/bench_results/official_c5_isl2500_osl1500_20260713_135720`
Binary stamped `2026-07-13 13:54:38`. Health: Falling back=0, MIXED=0,
no piecewise prefill, SERVER_STILL_UP, 10/10 OK.

| Metric | `_var0` padded | `_var1` broken varlen | **This (`_var2` host-cu)** |
|---|---|---|---|
| Mean TTFT | **2132 ms** | 3847 ms | **3489 ms** |
| Median TTFT | 2728 ms | 3178 ms | **4527 ms** |
| Mean TPOT | 55.1 ms | 54.3 ms | 54.5 ms |
| Warmup mean TTFT | — | — | 2253 ms |

**Server-side measure TTFT wave (ms)**:
`2707, 4103, 1805, 1806, 4095, 915, 4509, 4509, 4497, 4509`
— early requests ~1.8–2.7 s (near `_var0`); later / queued requests ~4.1–4.5 s
pull mean/median up. Warmup pair was `918, 1819` (client mean 2253).

**Verdict**: D2H fix recovered **~360 ms** of the +1715 ms regression but
did **not** restore the padded baseline (−1357 ms still vs `_var0`). Warmup
(C=2) is near baseline, so residual cost is in multi-req packed forward +
queue (median regresses to 4527). Next: Option A A/B (padded Mate under
packed scheduling) to isolate remaining varlen overhead.

---

## 9. Non-goals / anti-patterns

- Do **not** spend more effort refining the 58-bucket ladder for the default
  C=5 path (it is bypassed once pure-prefill goes eager varlen).
- Do **not** enable piecewise capture for packed multi-request prefills until
  packed Mate is proven (PR1 green); the homogeneous-batch gate exists for a
  real MUSA capture-crash reason.
- Do **not** remove or weaken decode graphs; decode stays on the full CUDA graph.
- Do **not** optimize GEMMs before post-batching Step-1 timing exists; both
  stacks share TorchMUSA and it is expected to be a wash.
- Do **not** ship PR2 (scheduler/routing) without PR1 (packed Mate) correctness
  gates green.
- Do **not** disable piecewise for the SGLang reference run; run
  `conc_eval_sglang.sh` unchanged (see Correction #1).
- Do **not** trade TTFT for worse TPOT; reject any change that merely shifts
  latency into decode.

---

## 10. Confirmed root cause: wrong Mate prefill kernel variant (2026-07-13 14:14)

The residual `_var2` regression is now localized. It is **not primarily**
remaining D2H synchronization, varlen packing, or scheduler overhead. xLLM's
deployed C ABI uses the non-warp-specialized **simple** Mate GDN kernel, while
SGLang's installed Mate path uses the warp-specialized
`fused_chunk_gdn_prefill` kernel.

### 10.1 Dispatch evidence

- xLLM's current varlen artifact was generated from
  `mate/gdn_kernels/tilelang/gdn_prefill_simple.py` by
  `scripts/build_gdn_prefill_varlen_op.py`.
- The active xLLM bridge loads:
  `/workspace/mate_cached_ops/mate_gdn_prefill_hq16_hv48_bf16/kernel_lib_cabi.so`
  (78,536 bytes, rebuilt Jul 13).
- SGLang's installed wrapper
  `/usr/local/lib/python3.10/dist-packages/mate/gdn_prefill.py` imports:
  - `mate.gdn_kernels.tilelang.gdn_kkt_solve.kkt_solve`
  - `mate.gdn_kernels.tilelang.gdn_prefill.fused_chunk_gdn_prefill`
- The installed warp-specialized Mate source compiles successfully with the
  current TileLang installation. The older `mate_feihu` warp-specialized source
  fails during TIR construction because it calls the removed
  `T.create_list_of_mbarrier` API. Therefore, the full C ABI must be built from
  the installed/new Mate source, not the stale local implementation.

### 10.2 Target-shape microbenchmarks

Hardware and shape:

```text
MTT S5000, BF16
C=5, ISL=2500
Hqk=16, Hv=48, D=128, chunk=64
total tokens=12,500
```

Raw kernel-family benchmark:

| Kernel family | Padded batch | Packed varlen |
|---|---:|---:|
| xLLM deployed simple kernel | 29.346 ms/layer | 28.384 ms/layer |
| SGLang/installed warp-specialized kernel | 1.182 ms/layer | 1.363 ms/layer |

The deployed simple kernel is approximately:

- **24.8x slower** for padded B=5.
- **20.8x slower** for packed varlen B=5.

Across 48 GDN layers:

```text
(28.384 - 1.363) ms/layer * 48 layers = 1,297 ms
```

Observed `_var2` regression relative to `_var0`:

```text
3,489 ms - 2,132 ms = 1,357 ms
```

The predicted kernel-variant gap (1,297 ms) nearly equals the measured TTFT
regression (1,357 ms). This is the strongest evidence so far and supersedes the
earlier hypothesis that remaining pack/cumsum orchestration is the primary
bottleneck.

### 10.3 Wrapper/layout overhead is secondary

An additional unequal-length benchmark used lengths:

```text
2556, 2585, 2580, 2570, 2568
```

with the installed full kernel:

| Path | Time per GDN layer |
|---|---:|
| Direct packed varlen kernel | 1.361 ms |
| Pack q/k/v/g/beta + kernel + unpack output | 2.143 ms |

Wrapper/layout cost is therefore approximately:

```text
0.782 ms/layer * 48 = 37.5 ms/forward
```

This is worth optimizing after kernel parity, but it cannot explain the
remaining ~1.36-second regression.

### 10.4 Revised diagnosis

The current TTFT stack is:

1. Packed scheduler and varlen correctness: functional.
2. Per-layer D2H storm: substantially fixed.
3. Remaining wrapper/packing cost: tens of milliseconds.
4. **Dominant remaining cost: simple serial GDN device kernel, ~1.30 seconds
   per prefill forward versus SGLang's warp-specialized kernel.**
5. Queueing amplifies this device-kernel gap for the second packed prefill
   group, producing the ~4.1–4.5 second TTFT tail.

TPOT remains ~54.5 ms because the regression is isolated to prefill.

---

## 11. Revised implementation plan

### Phase A - Immediate recovery with full padded Mate

Objective: recover the `_var0` baseline quickly and validate the kernel-choice
diagnosis before another varlen integration cycle.

1. Keep packed scheduling and eager pure-prefill routing enabled.
2. Add a separate, explicitly named full-kernel URI; do not overwrite or
   symlink the known-good simple artifact blindly.
3. Build a non-varlen warp-specialized C ABI from the installed Mate source.
4. For near-equal prompt lengths (padding ratio below a configured threshold),
   call the full padded kernel without `cu_seqlens`.
5. Keep the current simple-varlen kernel as a guarded fallback.

Expected result:

- Raw GDN kernel: ~29.3 -> ~1.2 ms/layer.
- Official mean TTFT: recover at least the 2.13-second `_var0` baseline.
- Likely additional queue compression because the first packed group completes
  ~1.3 seconds sooner.

Gate:

- B=1/B=2/B=5 correctness and recurrent-state parity.
- 10/10 official requests.
- No TPOT regression above 1 ms.

### Phase B - Build the warp-specialized varlen C ABI

Objective: match the actual SGLang kernel and remove padding dependence.

1. Use the installed Mate
   `gdn_kernels/tilelang/gdn_prefill.py`, which compiles with the current
   TileLang API.
2. Generate a varlen C ABI with:
   - packed q/k/v/g/beta;
   - `cu_seqlens`;
   - initial state and final state;
   - `is_log_space=true`;
   - output-final-state enabled.
3. Preserve the separate KKT solve required by the full kernel.
4. Deploy under a new URI and retain the simple kernel fallback.
5. Do not replace production artifacts until parity and latency gates pass.

Required tests:

- JIT full-varlen vs C ABI full-varlen.
- Full-varlen vs full-padded for equal lengths.
- Mixed lengths, partial 64-token chunks, B=1/2/4/5.
- Initial-state and final-state parity.
- Q/K L2 normalization parity.
- Long-context continuation state handoff.

Performance gate:

```text
full-varlen C ABI <= 1.6 ms/layer at C=5, ISL=2500
```

### Phase C - Remove secondary wrapper overhead

Only after Phase A/B is green:

1. Precompute host and device `cu_seqlens` once per forward, not per layer.
2. Precompute reusable pack/unpack index maps once per batch.
3. Replace per-layer `torch::cat` and output scatter with one custom packed
   gather/scatter kernel or produce packed projections directly.
4. Replace generic fp32 `torch::normalize` with Mate's specialized GDN
   L2-normalization kernel.
5. Reuse output, final-state, h0, and KKT buffers.

Expected remaining opportunity: approximately 38–100 ms per forward.

### Phase D - Scheduler and queue validation

After device-kernel parity:

1. Re-run the official C=5 benchmark three times.
2. Log packed group composition and per-forward device time.
3. Confirm the second group no longer produces a 4.5-second TTFT tail.
4. Compare directly with SGLang using the same prompt seeds and sampling.

Acceptance targets:

| Milestone | Mean TTFT target |
|---|---:|
| Phase A full padded | <=2.2 s |
| Phase B full varlen | <=1.65 s |
| Phase C wrapper cleanup | <=1.50 s |
| Final parity | within 5–10% of paired SGLang |

### Updated priority

```text
P0: full warp-specialized Mate C ABI
P1: full varlen ABI parity and deployment
P2: pack/unpack/cumsum and buffer reuse
P3: scheduler tuning after kernel parity
P4: ladder or unrelated GEMM optimization
```

Do not spend another iteration tuning ladder buckets or host-side cumsum before
the simple device kernel is replaced; those changes cannot recover the
measured ~1.30-second kernel deficit.

---

## 12. Code-level verification of §10 diagnosis (2026-07-13)

> Verified by reading all relevant source files in the container
> `xllm-musa2.9.1-sdk5.1-dev`.

### 12.1 Kernel variant claim — confirmed

The deployed artifact and its source differ from SGLang's installed kernel
at the TileLang pass-config level:

| Attribute | xLLM deployed (simple) | SGLang installed (warp-spec) |
|---|---|---|
| Source file | `mate_feihu/.../gdn_prefill_simple.py` | `mate/.../gdn_prefill.py` (installed pip pkg) |
| `TL_DISABLE_WARP_SPECIALIZED` | `True` | `False` |
| Warp groups | 0 (serial, single CTA) | 8 (`T.ws(0)`–`T.ws(7)`, producer/consumer pipeline) |
| Build script docstring | "simple (non-warp-specialized) GDN prefill kernel" | — |
| Deployed .so size | 78,536 bytes (`kernel_lib_cabi.so`, Jul 13 11:49) | — |
| SGLang `mate/gdn_prefill.py:8` | — | imports `fused_chunk_gdn_prefill` from installed `gdn_prefill.py` |

The warp-specialized kernel uses 8 warp groups with explicit mbarrier
synchronization, TMA loads (`T.copy` with barrier), and a producer/consumer
pipeline (`T.ws(6)`/`T.ws(7)` load q/k/v/a/b/g; `T.ws(0)` computes Vd;
`T.ws(1)` computes output; `T.ws(2,3,4,5)` compute P/Ag/state-update). The
simple kernel has none of this — it is a single-CTA serial implementation.

### 12.2 KKT solve requirement — confirmed

The warp-specialized kernel reads `a` (the KKT solve output, stored in
`pa_shared`) in its main loop for both the `Ag = G * Ar * b` and
`Pg = s * G * P` computations. The `a` tensor must be pre-computed.

The simple kernel computes the WY representation internally and does not use
`a`. xLLM's current prefill path at `gdn_prefill.cpp:793–795`:

```cpp
// Simple kernel computes WY inverse in-kernel; `a` is unused but required
// by the FFI signature.
a = torch::empty({1, num_tokens, num_v_heads, kGdnChunkSize}, ...);
```

xLLM already has the KKT solve infrastructure (`gdn_prefill.cpp:579–631`,
both torch fallback and Mate FFI paths) but **does not call it** in the
prefill forward. The installed SGLang `mate/gdn_prefill.py:165` calls
`kkt_solve` before `fused_chunk_gdn_prefill`:

```python
o, _, final_state = fused_chunk_gdn_prefill(
    q, k, v, a, g, b, ...)  # `a` is the kkt_solve output
```

### 12.3 ABI compatibility analysis

Current C ABI (13 args, from `mate_gdn_prefill_ffi.cpp`):
```
call(q, k, v, a, g, b, h0, cu_seqlens, o, ht,
     int num_tokens, int raw_batch_size, musaStream_t stream)
```

The warp-specialized kernel's Python wrapper `fused_chunk_gdn_prefill` takes
`(q, k, v, a, g, b, output, output_state, scale, initial_state, ...)`. The
tensor argument order matches the simple kernel's C ABI — the same 13-arg
signature is reusable. The only difference is that `a` must be **filled**
(by KKT solve) rather than left empty.

The `is_log_space` parameter is `True` in SGLang's usage. xLLM's current
code already does `chunk_local_cumsum_log_space` (`gdn_prefill.cpp:773`),
which produces log-space cumsummed g — matching the warp-specialized
kernel's expectation.

### 12.4 Microbench correlation — confirmed

```
Predicted gap: (28.384 − 1.363) × 48 layers = 1,297 ms
Measured gap:  3,489 − 2,132               = 1,357 ms   (within 5%)
```

The 60 ms residual (4.4%) is consistent with the measured wrapper/layout
overhead of ~37.5 ms (§10.3) plus measurement noise.

### 12.5 Phase A implementation requirements

To switch from the simple kernel to the warp-specialized kernel:

1. **New C ABI .so**: JIT-compile the installed `gdn_prefill.py`
   (`fused_chunk_gdn_prefill` with `is_varlen=False`) instead of
   `gdn_prefill_simple.py`. Modify `build_gdn_prefill_op.py` to import
   from `gdn_prefill` instead of `gdn_prefill_simple`.

2. **KKT solve integration**: Call `kkt_solve(key, beta, chunk_size)`
   before the kernel call in `gdn_prefill.cpp:795`. The
   `kkt_solve_mate_ffi` function (`:579`) already exists and produces
   `a` in the correct `[B, T, Hv, chunk_size]` shape. The KKT solve
   kernel must also be built from the installed `gdn_kkt_solve.py`
   (`is_varlen=False`).

3. **New URI**: Deploy as `mate_gdn_prefill_full_hq16_hv48_bf16` to
   keep the simple kernel (`mate_gdn_prefill_hq16_hv48_bf16`) as a
   guarded fallback. The URI is resolved at `gdn_prefill.cpp:775–776`.

4. **Log-space g (corrected in §13)**: Warp full path uses raw log-g and
   lets the kernel cumsum internally (`is_log_space=true`). Do **not** apply
   host `chunk_local_cumsum` on the full path (that is simple-kernel only;
   double-cumsum → NaN).

5. **L2 norm**: Host `l2norm_last_dim` before KKT is fine for Phase A;
   Mate `gdn_l2norm_` can replace it in Phase C.

6. **State layout (corrected in §13)**: Keep mate k-last `[B,Hv,V,K]` at the
   cache edge; do not FLA-transpose around the warp C ABI.

### 12.6 Risk: mate_feihu vs installed mate source divergence

The `mate_feihu` package at `/workspace/xllm_qwen3.5/mate_feihu` also has a
`gdn_prefill.py` with `TL_DISABLE_WARP_SPECIALIZED: False`, but it uses the
removed `T.create_list_of_mbarrier` API and fails to compile with the
current TileLang installation. The **installed** pip package
(`/usr/local/lib/python3.10/dist-packages/mate/`) compiles successfully.
Phase A/B must build from the installed source, not the stale local copy.

---

## 13. Phase A landed: warp-specialized Mate prefill (2026-07-13 15:55)

> Status: **DONE** for Phase A (padded warp C ABI + xLLM wiring + official
> C=5 re-bench). Phase-1 target (<=1.65 s) **not yet met**.

### 13.1 What landed

1. **Built warp C ABI from installed Mate** (not stale `mate_feihu` warp):
   - Scripts: `mate_feihu/scripts/gen_gdn_prefill_full_mu.py`,
     `build_gdn_prefill_full_op.py`
   - FFI: `mate_feihu/csrc/integrations/gdn/mate_gdn_prefill_full_ffi.cpp`
   - URI: `mate_gdn_prefill_full_hq16_hv48_bf16`
   - Deployed under `/workspace/mate_cached_ops/` (`kernel_lib_cabi.so` ~72 KB
     + bridge `.so`)
   - ABI differs from simple: needs `batch_size` + Q/K/V strides; `a` must be
     KKT-filled (not empty)

2. **xLLM wiring** (`gdn_prefill.cpp`, `qwen3_gated_delta_net_base.cpp`):
   - Prefer full URI when present; fallback simple via
     `XLLM_MATE_GDN_PREFILL_SIMPLE=1`
   - Full path: padded `[B,T]`, **no pack**, **no host cumsum** (warp
     `is_log_space=true` cumsums log-g internally — pre-cumsum causes NaN)
   - Call `kkt_solve` before the kernel
   - State stays mate **k-last** `[B, Hv, V, K]` (do **not** FLA-transpose;
     kernel indexes `h0[..., V, K]`; wrong transpose broke decode → empty
     "Thinking" answers)
   - `use_qk_l2norm_in_kernel = true` (host L2norm before KKT)

3. **TileLang**: ported installed Mate sources into local tree; simple is no
   longer the primary production path (kept as env fallback).

### 13.2 Correctness

| Check | Result |
|---|---|
| C ABI vs JIT (matching SGLang inputs) | maxdiff **0.0** |
| `XLLM_MATE_GDN_PREFILL_SIMPLE=1` A/B | PASS (391) |
| Warp path after k-last fix | **PASS** (391) |
| Official C=5 10/10 | Successful |

### 13.3 Official C=5 results (warp live)

Workload: C=5, ISL=2500, OSL=1500, packed_prefill + piecewise graph.
Results dir:
`/workspace/bench_results/official_c5_isl2500_osl1500_20260713_155119`

| Run | Config | Mean TTFT | Median TTFT | Mean TPOT |
|---|---|---:|---:|---:|
| `_var0` | packed + simple padded | 2132 ms | 2728 | ~54 |
| `_var2` | packed + simple varlen + host-cu | 3489 ms | 4527 | ~54 |
| **`_warp` (this)** | packed + **warp full** padded | **1946 ms** | **2178 ms** | 53.9 |
| SGLang ref | warp Mate | ~1420 ms | — | — |
| Phase-1 target | — | <=1650 ms | — | — |

Warmup (cold capture) mean TTFT was 3592 ms; measure numbers above are the
10-prompt result.

### 13.4 Interpretation / caveat

- Phase A gate (<=2.2 s) **met** (1946 ms).
- Beat simple `_var0` by ~186 ms; still **~300 ms above Phase-1** and
  **~500 ms above SGLang**.
- **Do not equate `_var2`−`_var0` to simple−warp.** `_var0` also used the
  simple C ABI; that 1.36 s gap was orchestration/queue under broken varlen,
  not the kernel-family delta alone. The right comparison is warp xLLM vs
  SGLang (~1.42 s).
- Microbench still predicts ~1.3 s device-kernel headroom vs simple; most of
  that is now closed in e2e. Residual vs SGLang is likely remaining
  orchestration (pack groups, piecewise, KKT host/FFI, state/layout edges)
  plus Phase B varlen warp.

### 13.5 Next

1. **Phase B**: warp **varlen** C ABI (packed lengths, real `cu_seqlens`).
2. Profile residual ~500 ms vs SGLang (KKT+kernel waterfall, packed group
   composition, queue).
3. Phase C wrapper cleanup only after B is green.
4. Keep `XLLM_MATE_GDN_PREFILL_SIMPLE=1` as rollback.
