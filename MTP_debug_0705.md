# MTP-on-MUSA Graph-Mode Bring-up Handoff (Qwen3.5-27B)

Last updated: 2026-07-05. Continues `handoff.md` / `handoff_0703.md`.

## 0. TL;DR for the next agent

- **Eager MTP is GREEN** (both `XLLM_MATE_GDN_MTP=1` and `=0`) -> 391 on `17x23`.
- **Graph MTP (`ENABLE_GRAPH=1`) still FAILS** (garbage text) -- this is the open bug.
- The failure is **isolated to the spec-verify (target validate) CUDA-graph capture/replay path**, NOT the GDN kernel and NOT the attention kernel choice:
  - Graph fails with mate GDN **and** host-loop GDN -> not a GDN-kernel bug.
  - Non-MTP decode graph PASSES -> the generic decode graph machinery is fine.
  - **`XLLM_SPEC_VERIFY_EAGER=1` + `ENABLE_GRAPH=1` PASSES** (draft+decode graphed, only the 2-token target verify runs eager). Key bisection: the bug lives in the spec-verify graph capture/replay.
- A **refactor is in-flight** (see section 4) to make the GDN MTP verify-cache a shared, layer-id-keyed registry. Source edits are complete; **the build was interrupted and has NOT been verified to compile or run yet.** Finish the build first.
- The **most likely remaining root cause** after the refactor: `input_params.num_accepted_tokens` (and the derived `checkpoint_indices` gather) are **captured with stale/dummy values** and never refreshed on replay (see section 5). Fix that next.

## 1. Environment / build

- Container: `xllm-musa2.9.1-sdk5.1-dev`; host `/data/feihu` -> `/workspace`. Repo = `/workspace/xllm-git-master`.
- Repo root is root-owned: create new files from inside the container (as here) or edit existing files; the host-side Write tool is permission-denied at repo root.
- Binary: `build/lib.linux-x86_64-cpython-310/xllm/xllm`.
- Incremental rebuild (needed after every edit):
  ```bash
  docker exec xllm-musa2.9.1-sdk5.1-dev bash -lc '
    ln -sf /usr/local/bin/ninja /ninja   # CMake cache has a broken CMAKE_MAKE_PROGRAM=/ninja
    cd /workspace/xllm-git-master/build/cmake.linux-x86_64-cpython-310
    /usr/local/bin/ninja -j4 xllm 2>&1 | tail -8'
  ```
  - Full-ish rebuild ~85s (`-j4`). Editing `model_input_params.h` (widely included) triggers a bigger rebuild.
  - The MUSA `mcc` wrapper prints a `checking .cpp` clang pre-pass whose diagnostics can be benign, but treat any `error:` as real and re-verify.
- Models: target `/workspace/model_weights/Qwen3.5-27B`, draft `/workspace/model_weights/Qwen3.5-27B-mtp`.

## 2. How to test

```bash
docker exec xllm-musa2.9.1-sdk5.1-dev bash -lc '
cd /workspace/xllm-git-master
source kill_zombie_xllm.sh && kill_zombie_xllm 8092 9748

# (a) EAGER MTP -- expected PASS (391)
START_SERVER=1 STOP_SERVER=1 PORT=8092 MASTER_PORT=9748 \
  XLLM_MATE_GDN_MTP=1 ENABLE_GRAPH=0 \
  NUM_SPECULATIVE_TOKENS=1 SPECULATIVE_ALGORITHM=MTP \
  DRAFT_MODEL_PATH=/workspace/model_weights/Qwen3.5-27B-mtp \
  bash correctness_check.sh 2>&1 | tail -8

# (b) GRAPH MTP -- currently FAIL (garbage). This is what to fix.
START_SERVER=1 STOP_SERVER=1 PORT=8093 MASTER_PORT=9749 \
  XLLM_MATE_GDN_MTP=1 ENABLE_GRAPH=1 ENABLE_GRAPH_VMM_POOL=0 \
  NUM_SPECULATIVE_TOKENS=1 SPECULATIVE_ALGORITHM=MTP \
  DRAFT_MODEL_PATH=/workspace/model_weights/Qwen3.5-27B-mtp \
  bash correctness_check.sh 2>&1 | tail -12

# (c) BISECTION: graph on, verify forced eager -- expected PASS
START_SERVER=1 STOP_SERVER=1 PORT=8094 MASTER_PORT=9750 \
  XLLM_MATE_GDN_MTP=1 ENABLE_GRAPH=1 ENABLE_GRAPH_VMM_POOL=0 XLLM_SPEC_VERIFY_EAGER=1 \
  NUM_SPECULATIVE_TOKENS=1 SPECULATIVE_ALGORITHM=MTP \
  DRAFT_MODEL_PATH=/workspace/model_weights/Qwen3.5-27B-mtp \
  bash correctness_check.sh 2>&1 | tail -8
'
# Server log: /workspace/xllm-git-master/log/xllm_Qwen3.5-27B.log  (binary-ish)
# strings <log> | grep -E "FATAL|Lazy capturing|Failed to capture|spec-verify|TVM function not found"
```

Env flags added this session:
- `XLLM_SPEC_VERIFY_EAGER=1` -- routes the spec-verify validate forward to eager even when `ENABLE_GRAPH=1` (decode/draft still graphed). Bisection + temporary correct-but-slower fallback.
- `XLLM_MATE_GDN_MTP=1` (default on) -- fused mate GDN MTP kernel; `=0` uses host-loop recurrent GDN.
- `XLLM_DEBUG_QWEN35_MTP=1` -- inserts `cudaDeviceSynchronize` at many GDN stages (masks stream races) and logs `[MTP fused-vs-host]` diff.

## 3. FIXED this session (keep these)

### 3a. Eager mate GDN MTP regression -> FIXED (missing FFI-stream sync)
- Symptom: `XLLM_MATE_GDN_MTP=1 ENABLE_GRAPH=0` gave garbage ("Thinking!!!!..."); `XLLM_DEBUG_QWEN35_MTP=1` made it pass. Classic stream race.
- Root cause: on MUSA, `bind_tvmffi_stream_to_current_torch_stream()` (`xllm/core/kernels/musa/musa_tvmffi_stream.cpp`) binds the TVM-FFI kernel to a **dedicated pool stream** when NOT capturing (and pre-syncs the compute stream). The mate MTP FFI kernel runs on a *different* stream than the caller's compute stream, but there was **no post-kernel sync**, so the caller consumed `output`/`intermediate` before the kernel finished. (Committed `0b8863e2` had a blunt `cudaDeviceSynchronize` here; earlier edits this session removed it.)
- Fix: `sync_musa_ffi_stream(q.device())` right after the FFI `run(...)` in `mate_gated_delta_rule_mtp` (`xllm/core/kernels/musa/gdn_decode.cu`, ~line 663). No-op under capture (kernel then runs on the capture stream) and during replay -> graph-safe.
- Also removed now-redundant `sync_spec_verify_cross_streams()` (`cudaDeviceSynchronize`) calls in `qwen3_gated_delta_net_base.cpp` that would break capture.

### 3b. Attention routing under graph (URI mismatch) -> mitigated
- Original graph crash: `F ... TVM function not found. uri=batch_decode_..._paged_run ... batch_chunked_prefill() ... chunked_prefill_forward()`. Planner built a **decode** plan (uri `batch_decode...`) for spec verify, but the runtime called `batch_chunked_prefill` (looks up `paged_run` on that .so -> missing -> FATAL).
- Fixes in `xllm/core/layers/{musa,cuda}/flashinfer_attention.cpp`:
  - `forward(...)` routes spec-verify chunked-prefill to `decoder_forward` when `is_spec_verify` (plus the expanded-decode flags).
  - `chunked_prefill_forward(...)` early-returns into `decoder_forward` if `plan_info->uri` contains `batch_decode` OR `is_spec_verify`.
  - Rewrote a `*std::optional::emplace(...)` one-liner into `if`/`has_value()` (fixed a real "indirection requires pointer operand" compile error).
- Result: FATAL gone; capture logs `Lazy capturing CUDA spec-verify graph ... done`. But output still garbage -> remaining bug is data/state correctness, not dispatch.

### 3c. Graph executor plumbing (keep)
- `cuda_graph_executor_impl.cpp`: `use_expanded_spec_decode_attention` at the persistent-param layer keys off `graph.use_expanded_decode_for_spec_verify_attention` + chunked-prefill (no longer requires `params.is_spec_verify`, which capture/replay copies may drop); `params_for_capture->is_spec_verify` propagated; eager fallback for generic chunked prefill without a spec-verify graph path.
- `qwen3_next_hybrid_base.h`: when graph supplies prebuilt `attn_metadata`, re-sync expanded-decode fields (flag, expanded paged-KV, block tables) from `input_params.graph.*` or the prebuilt `attn_metadata`.

## 4. REFACTOR IN-FLIGHT (source complete, build/run NOT yet verified)

**Goal:** make the GDN MTP verify-cache correct under both eager and graph. Old design stashed per-layer intermediates into `std::vector`s inside `input_params.gdn_mtp_verify_cache` (a plain `std::optional<GdnMtpVerifyCache>`), then the post-verify scatter read them back. Broken because:

- `ModelInputParams::to(device)` **deep-copies** the optional, so the model forward pushes intermediates into the **device copy** (`target_prepared.input_params`), while `run_validate` scatter read the **original** (`validate_input.input_params`) -> scatter saw an **empty** cache. Confirmed with debug logging: pushes to cache `0x...ad8` size 48, scatter on cache `0x...5c8` size 0.
- Under CUDA graph, the C++ `push_back` runs only during **capture**; on **replay** nothing is pushed, so a per-step vector is always stale.

**Changes made:**
- `model_input_params.h`: `GdnMtpVerifyCache` is now `{ bool enabled; std::map<int32_t, LayerState> layer_states; }`, `LayerState { torch::Tensor ssm_intermediate; torch::Tensor conv_intermediate; }`. Keyed by layer id, **overwritten (not appended)** each forward -> stable set across eager steps and graph replays. Added `#include <map>`.
- `ModelInputParams::gdn_mtp_verify_cache` is now a **`std::shared_ptr<GdnMtpVerifyCache>`**; `ModelInputParams::to()` **shares** it (no deep copy). Device copy, graph capture copy, and original all point at one registry. Under graph, entries alias the **per-layer persistent grow-only buffers** (`mate_gdn_mtp_intermediate_buf_`), whose contents the replayed kernel refreshes in place.
- `qwen3_gated_delta_net_base.cpp`:
  - `run_spec_verify_gated_delta_rule_mate(...)` writes `verify_cache->layer_states[layer_id].ssm_intermediate = std::move(intermediate);`
  - `run_spec_verify_conv(...)` gained a `layer_id` param and writes `verify_cache->layer_states[layer_id].conv_intermediate = ...`.
  - Single `gdn_layer_id` (from `attn_metadata.plan_info->layer_id`) computed once, passed to both.
  - `scatter_gdn_mtp_verify_ssm_states(...)` rewritten to iterate `cache.layer_states` map.
- `mtp_worker_impl.cpp`: `input_params.gdn_mtp_verify_cache = std::make_shared<GdnMtpVerifyCache>()`; scatter call passes `*validate_input.input_params.gdn_mtp_verify_cache`.

**WARNING: next step is to finish the incremental build and re-run tests (a) and (b) in section 2.** `model_input_params.h` changed -> expect a larger rebuild. If it compiles, eager should still PASS; then check whether graph improves.

**Subtle correctness note:** eager mate PASSED *even while the scatter was a no-op* (empty cache). So either (i) the uncommitted SSM state is "good enough" for `num_spec=1` greedy short generations, or (ii) state is committed by another path. After the shared-cache refactor makes the scatter actually run, **re-verify eager still PASSES** and ideally confirm the scatter changes generated text at longer `OUTPUT_LEN`, so we know the commit is wired AND correct (right slot, right transpose -- `fla_ssm_state_layout=true`).

## 5. MOST LIKELY REMAINING GRAPH BUG (attack next)

The verify forward depends on **per-step runtime values not staged into persistent graph buffers**, so capture bakes stale/dummy values and replay is wrong:

1. **`input_params.num_accepted_tokens`** (device tensor). The mate wrapper computes
   `gather_idx = (num_accepted_tokens - 1); init_slots = checkpoint_indices.gather(1, gather_idx)`
   to pick the **initial recurrent state slot**. Capture uses the warmup/first-step value (often all-1s -> slot 0); replay with `accepted=2` needs slot 1. `num_accepted_tokens` is NOT in the persistent graph metadata -> captured gather reads stale data.
   - Fix idea A: add `persistent_num_accepted_tokens_` in `CudaGraphPersistentParam` (like `persistent_linear_state_indices_`), refresh in `update()`, feed the persistent version to the GDN layer during capture/replay.
   - Fix idea B (sglang-style, preferred): make verify compute **all** `T` steps unconditionally (accepted-count-independent), select the accepted step only in the (eager) post-verify scatter.
2. **`checkpoint_indices`** derives from `linear_state_base_indices` (from `linear_state_indices`, staged as `persistent_linear_state_indices_`) + `arange`. Confirm the layer actually consumes the persistent indices during capture (hybrid model copies `attn_metadata`/`input_params`).
3. **`run_spec_verify_conv`** is a host-loop with `.to(fp32)`/`torch::empty`/`index_copy_` and per-seq `narrow`s driven by `num_accepted_host` (CPU vector). Under capture it may allocate mid-capture and bakes step-0 accepted lengths. For a fully-captured verify, port to a device-indexed kernel (NPU reference: `CausalConv1dGraphBranch::kSpecVerify` / `run_causal_conv1d_graph_update` in `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp`) or keep conv eager.

sglang reference (design, not copy): target-verify = **ragged paged prefill with a causal/tree mask over `draft_token_num` tokens/seq**; GDN kernel checkpoints **all** per-step states with `disable_state_update=True` into a **persistent pool tensor** indexed by layer+req-slot; accepted step scattered back **after** verify by `update_mamba_state_after_mtp_verify` (`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `.../linear/gdn_backend.py`, `.../mamba/mamba_state_scatter_triton.py`). Verify forward is **independent of accepted count**; only the post-verify scatter uses it. Our mate wrapper consuming `num_accepted_tokens` inside the forward is what makes it graph-unsafe.

## 6. Pragmatic fallback

`XLLM_SPEC_VERIFY_EAGER=1` (in `cuda_graph_executor_impl.cpp::run` spec-verify phase) makes MTP correct with decode/draft graphed and only the 2-token target verify eager. Already PASSES. Not the full TPOT win (target verify is the expensive forward) but a safe checkpoint / A-B baseline.

## 7. Files changed this session (uncommitted; do NOT commit unless asked)

- `xllm/core/kernels/musa/gdn_decode.cu` -- 3a: `sync_musa_ffi_stream` after mate MTP FFI `run`.
- `xllm/core/layers/musa/flashinfer_attention.cpp`, `xllm/core/layers/cuda/flashinfer_attention.cpp` -- 3b: spec-verify -> `decoder_forward` routing + chunked-prefill->decode redirect; optional emplace compile fix.
- `xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp` / `.h` -- 4 refactor: layer-keyed verify cache writes; `run_spec_verify_conv` gets `layer_id`; scatter iterates `layer_states`; removed dead `sync_spec_verify_cross_streams` uses. (`.h` keeps `mate_gdn_mtp_intermediate_buf_` / `mate_gdn_mtp_output_buf_`.)
- `xllm/core/framework/model/model_input_params.h` -- 4 refactor: `GdnMtpVerifyCache` map form + `shared_ptr` + shared in `to()`; `#include <map>`.
- `xllm/core/runtime/mtp_worker_impl.cpp` -- 4: `make_shared` verify cache; scatter reads shared cache.
- `xllm/core/runtime/cuda_graph_executor_impl.cpp` / `.h` -- 3c + `XLLM_SPEC_VERIFY_EAGER` gate; expanded-decode persistent buffers.
- (Earlier-session, still present) `attention_metadata.h` / `attention_metadata_builder.cpp` (shared `is_spec_verify`); `spec_input_builder.{h,cpp}` (expanded paged-KV builder); `qwen3_next_hybrid_base.h`.

Debug `[mtp-dbg]` LOGs were removed. If re-added, gate on `std::getenv("XLLM_MTP_DBG")`.

## 8. Bisection results (do not re-run)

| Config | Result |
|---|---|
| Eager, `XLLM_MATE_GDN_MTP=1` | PASS (391) -- after 3a fix |
| Eager, `XLLM_MATE_GDN_MTP=0` (host loop) | PASS (391) |
| Graph, `XLLM_MATE_GDN_MTP=1` | FAIL (garbage) |
| Graph, `XLLM_MATE_GDN_MTP=0` (host loop) | FAIL (garbage) |
| Graph, `XLLM_SPEC_VERIFY_EAGER=1` | PASS (391) |
| Non-MTP decode graph (no draft) | PASS |

Interpretation: bug is in the **spec-verify CUDA-graph capture/replay of the validate forward**, independent of GDN kernel and generic decode-graph machinery -> focus on section 5 (per-step runtime inputs not staged into persistent graph buffers), starting with `num_accepted_tokens`.
