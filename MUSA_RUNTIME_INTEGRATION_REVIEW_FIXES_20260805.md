# MUSA runtime integration review fixes

Date: 2026-08-05
Branch: `musa-runtime-integration`
Base: `ed930eb4`

## Scope

This note records the ordered fixes applied after reviewing
`musa-runtime-integration-REVIEW-P1-P2.md`. Existing staged PR5 changes were
preserved; no commit was created in this pass.

The cumulative sequence-length fix adds
`xllm/core/framework/batch/batch_input_builder.h` to the original PR5 source
file list because the `.cpp` behavior is not correct without the matching
state initialization.

## Changes

### 1. Decode identity-selection guard

`xllm/core/runtime/llm_worker_impl.cpp` now uses an explicit structural
invariant for the MUSA decode fast path: the batch must be `DECODE`, the
selection must be a contiguous one-dimensional integer tensor, and the
selected-row count must equal both the hidden-state row count and the recorded
sequence count. Padded decode and multi-row speculative inputs therefore keep
the indexed path.

The same decision is reused for speculative embeddings, so MUSA does not
reintroduce an identity `IndexSelect` after the logits optimization.

### 2. Preserve non-MUSA `qwen3_next` behavior

`xllm/models/llm/qwen3_next.h` selects the MUSA decoder header only for
`USE_MUSA`; all other builds retain the previous NPU header path. This keeps
the runtime-integration PR scoped to MUSA and avoids changing unrelated device
paths.

### 3. Remove unconditional MUSA host event waits

The MUSA-only `StreamEvent::synchronize()` calls were removed from
`LLMWorkerImpl`. Both metadata readiness and schedule-overlap dependencies now
use the existing `Stream::wait_event()` device-side dependency. Null events
remain safe because `Stream::wait_event()` treats them as already satisfied.

This avoids blocking the host at every schedule-overlap boundary. The
TorchMUSA event implementation maps `block()` to `musaStreamWaitEvent`; a
future platform-level workaround should be added there if a reproducible
cross-thread wait failure is found, rather than exposing an environment switch
that can silently trade correctness for latency.

### 4. Make paged-KV mirror ownership explicit

MUSA batch construction now assigns the device-side CPU handles from the
retained host mirrors instead of moving temporary handles. The mirrors remain
read-only until `AttentionInput::to()` creates the actual device tensors for
Mate/FlashInfer.

### 5. Initialize cumulative MUSA sequence lengths

`BatchInputBuilder::BuilderState` now initializes MUSA `seq_lens` and
`q_seq_lens` with a leading zero, matching the cumulative updates in
`process_single_sequence()` and multithreaded state merging. This prevents the
first MUSA sequence from calling `back()` on an empty vector.

### 6. Formatting

The changed C++/header hunks were run through `clang-format 20.1.6` and the
previously reported `qwen3_next_hybrid_base.h` formatting violation was fixed.

## Validation performed

- Target local style and upstream canonical style were both read and matched.
- `git diff --cached --check` is clean after these fixes.
- No changed file contains CRLF or a BOM.
- `git-clang-format` was run with `/home/mccxadmin/.local/bin/clang-format`
  version 20.1.6; the repository CheckFormat-equivalent diff reports no files
  requiring changes.
- A TorchMUSA two-thread/two-stream probe on a free MUSA device passed: the
  consumer `wait_event()` call returned in 0.149 ms, and work queued after it
  observed both ends of the producer tensor at the expected value `32.0`.
  This verifies device ordering without a host-side event synchronize.

## Follow-up gates

- Build and MUSA correctness must be run after the dependent kernels/layers/
  graph-executor PRs are present in the same build graph.
- Run a matched schedule-overlap correctness and latency A/B before declaring
  the event-wait change performance-neutral.
- Keep common `fused_moe` excluded only while the MUSA model registry remains
  limited to the MUSA implementations; add a MUSA implementation or an
  explicit CMake source gate before enabling generic `qwen3_moe` on MUSA.
