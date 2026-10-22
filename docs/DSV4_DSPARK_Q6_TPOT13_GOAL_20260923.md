# DSV4 DSpark q6 TPOT goal regression (2026-09-23)

**2026-09-25 latest result:** one current-Attention q6 Graph+draft candidate
passes fixed GSM8K100 **96/100**, hot TTFT **215.340 ms**, TPOT **11.9828 ms**,
strict counting **6/6**. See [joint acceptance and reproducible launch](DSV4_DSPARK_MERGED_ACCEPTANCE_20260925.md).
Statements below about an unmet goal describe the earlier experiments, not this
latest candidate. General rejected-prefix robustness remains separately qualified.

Latest measured state-lookup optimization, canonical speed reproduction, and
C128 Graph correctness work: [2026-09-25 investigation](DSV4_DSPARK_GRAPH_STATE_C128_20260925.md).

## Evidence correction (2026-09-25; supersedes causal claims below)

- The `86.6 ms` run in `repro_historical_hybrid_b1024_20260924_b`
  contains **489 forced-Eager route messages** and no successful Graph capture.
  It did not set `XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY=1`. It is NOT evidence
  that a 1024-token prefill budget causes a slow q6 Graph or that 10240 is
  required for 8 ms. Both budgets still select a six-token decode bucket.
- The 47.9 ms value is server-only diagnostic timing: its client artifact
  contains no visible content. Neither throughput nor correctness passed.
- Two startup FPEs (`repro_capturekernel_basearchive_20260924` and
  `repro_capture_safe_archive_graph_20260924`) were preceded by **master port
  bind failures**, not a demonstrated kernel ABI fault. OOM in
  `repro_merge_legacyq6_basearchive_20260924` happened while our 70 GiB GPU
  reservation was still active. These are experiment setup failures.
- `XLLM_GRAPH_TIMING=1` alone does not time replay until three captures exist;
  this service has one target capture. `XLLM_GRAPH_TIMING_ALL=1` is required
  for direct replay timing. Log-to-log step intervals also include draft and
  metadata work and cannot be called kernel timings.
- Comparing 33 ms (short-prefix capture/legacy Attention) with 155 ms
  (long-prefix capture/another archive) changes more than one variable. No
  claim that the archive, ABI, or batch budget is the sole cause is proven.

The actual historical binary remains available at
`build/gsm8k_candidates/repro_q6_legacy_20260923/xllm`, SHA256
`ca820538d9500e2c7a33eca24f791026b214db82dacaff14661bb1d14955e08f`.
A fresh same-hash run (`logs/goal_samehash_shortfirst_20260925`) explicitly
enables compact Graph and legacy q6, captures with 64/64 first, and records
the routing flags. Its subsequent plain-padding 2500/1500 probe reports
TPOT **8.259 ms**, replay approximately **33.6 ms**, but strict counting
**fails** (485 parsed numbers, not 485 correct). This is a speed reproduction,
not accuracy/performance acceptance. The cold/warm setup, capture request,
prompt contents, active path and cache hits must all be retained for each A/B.

The active merge goal remains unmet. Do not use the fixed-100 short Eager
97/100 result as proof of pure q6 Graph accuracy or as proof for another
binary. A final acceptance run must use one candidate, fixed launch flags,
and both the fixed GSM8K set and complete declared performance window.

## Scope

This record covers the `fix/dspark-accuracy93-perf-20260922` checkout, DeepSeek-V4-Flash-0731-FP8-mt, MUSA TP8/EP1/DP1/CP1, and the fixed GSM8K 100-question set. The default checkout remains the accuracy-protected q2 Eager route.

## Candidate design

The current Attention implementation is retained for q1/q2 and all Eager paths. A historical `d787` Attention object is linked only as a private `forward_legacy_graph` symbol. Its constructors, loader, and helper members are renamed before linking, so they cannot override current q2 state or weight loading. Runtime selection additionally requires:

```text
XLLM_DSV4_ENABLE_LEGACY_GRAPH_Q6=1
```

and only applies when `attn_metadata.is_acl_graph`, decode is active, and the target token bucket is larger than two (DSpark q6). A normal build without the historical object has no legacy symbol and keeps the current implementation.

Reproducible builder:

```bash
python3 tools/build_dsv4_q6_legacy_candidate.py \
  build/gsm8k_candidates/repro_q6_legacy_20260923
```

The builder reuses the existing `build/dspark-musa-local` compile database and libraries, records source/object/archive hashes, renames every old Attention member, and writes a private executable. The independently rebuilt candidate is:

```text
build/gsm8k_candidates/repro_q6_legacy_20260923/xllm
sha256 ca820538d9500e2c7a33eca24f791026b214db82dacaff14661bb1d14955e08f
```

## q2 accuracy acceptance

Use the accuracy baseline configuration, not the q6 performance configuration:

```text
num_speculative_tokens=1
enable_prefix_cache=false
max_tokens_per_batch=2048
max_tokens_per_chunk_for_prefill=2048
Graph=true, target compact verify defaults to Eager
```

The script-rebuilt candidate was evaluated on the fixed 100 GSM8K indices with `temperature=0`, `max_tokens=2048`, reasoning prompt, and the same dataset as the previous 95% baseline:

```text
logs/goal_repro_q6_legacy_q2_accuracy_20260923/eval100/summary.json
```

Result: **95/100 (95%)**, failed indices `235, 255, 333, 952, 1038`. The service log contains `[DSV4_GRAPH_ROUTE] target compact verify forced eager`, confirming that this is q2 Eager precision protection rather than Graph q2 accuracy.

## q6 Graph/DSpark performance acceptance

Configuration:

```text
N=5, Graph=true, Prefix Cache=true (rate=0.65)
TP8/EP1/DP1/CP1, B1, input/output=2500/1500
formal variants 2--7, count-to=750, strict integer output
XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY=1
XLLM_DSV4_ENABLE_GRAPH_BATCHED_MATE_VERIFY=1
XLLM_DSV4_ENABLE_LEGACY_GRAPH_Q6=1
```

Run directory:

```text
logs/goal_repro_q6_legacy_perf_20260923/
```

Formal window result:

| Metric | Result |
|---|---:|
| TTFT_AVG | 217.83 ms |
| TPOT_AVG | **8.127 ms** |
| Strict formal requests | 6/6 |
| Output correctness | 6/6 |
| Gate | PASS |

Per-variant q6 TPOT was `8.167, 8.185, 8.187, 8.145, 8.173, 8.158 ms`. All variants had 1536 cached tokens and passed the strict full-text checks. This satisfies TPOT <13 ms and TTFT <250 ms for the requested q6 hot-state configuration.

## Important boundary

The legacy Attention object is an experimental q6 performance candidate, not a default source replacement. Do not enable it for q2 or Eager validation without the fixed accuracy configuration above. The q6 path and q2 accuracy path must be reported separately because they intentionally use different scheduler settings.

## Follow-up q6 GSM8K semantic check

The first 20 indices of the fixed 100-question set were subsequently run with
the same q6 Graph+draft settings. This exposed the gap that counting-only output
tests cannot detect:

| q6 route | Correct | Failed indices |
|---|---:|---|
| Legacy q6 fast Attention | 13/20 (65%) | 64, 125, 184, 201, 216, 235, 244 |
| Current Attention Graph q6 | 14/20 (70%) | 3, 23, 128, 235, 244, 255 |
| Current Attention + target-only acceptance A/B | 17/20 (85%) | 206, 235, 244 |

The target-only A/B greatly improves semantic stability but increases per-request
latency and is not yet a production merge. Clearing persistent state, disabling
Prefix Cache, and disabling the batched-MATE bypass did not change the 14/20
current-Graph result. Therefore q6 Graph target Attention/DSA state equivalence
and accepted-token rollback remain open; the q2 95/100 result must not be used as
q6 evidence.

## q2/q6 merge A/B (goal continuation)

To test whether q6's accuracy gap was caused by the frozen persistent state, the
following opt-in controls were evaluated on the first 20 fixed GSM8K indices:

| A/B | Accuracy | q6 effect |
|---|---:|---|
| q6 Graph, frozen state | 14/20 | baseline Graph path |
| q6 Graph, full Eager-style compressor/indexer state update | 18/20 | semantic improvement, but formal TPOT about 50 ms |
| q6 Graph, batched C4 state kernel update | 18/20 | same semantic result, TPOT about 50 ms |
| q6 Graph, only first state row update | 14/20 | insufficient |
| q6 Graph, target-only accepted-token commit | 17/20 | slower and still incomplete |

The batched state kernel reduces per-token compression launches but does not yet
recover the 8 ms q6 path. The remaining overhead is broader than one kernel:
state gather/scatter, Graph persistent metadata, MATE/indexer metadata, and cache
publication still occur on the q6 verification step. The next design should
precompute or incrementally update accepted-row state outside the replay critical
path, then copy only persistent buffers before replay; simply enabling all q2
state updates inside Graph forward is not a viable performance merge.

Logs:

```text
logs/repro_q6_legacy_gsm8k20_20260923/eval20/
logs/repro_current_q6_gsm8k20_20260923/eval20/
logs/repro_q6_unfrozen_state_20260923/eval20/
logs/repro_q6_batched_state_v2_20260923/eval20/
```

The latest batched-state v3 A/B also confirms that the remaining GSM8K error is
localized: indices 3, 23, 128, and 206 pass while index 244 still fails. A
target-Eager mixed q6 run is not equivalent to q2 Eager: keeping N=5 draft and
accepted-token processing changes the request lifecycle, and its first 20 result
was 16/20. Therefore the final merged design must preserve q2's accepted-row
state semantics, not merely route the target model forward through Eager.

Current status of the merge goal: **not complete**. The verified compatible
points are q2 Eager 95/100 and q6 Graph performance 8.127 ms; their direct
combination still requires an out-of-replay accepted-state update path.

## Goal continuation: state-update optimization A/B

An opt-in `XLLM_DSV4_DISABLE_GRAPH_Q6_FREEZE=1` path restores the q2-style
compressor/indexer persistent state updates inside q6 Graph. The first 20 GSM8K
result improved to **18/20**, but formal q6 TPOT was approximately **50 ms**.
Replacing the per-token C4 updates with the existing
`deepseek_v4_compress_decode_sequence_c4` batched kernel kept **18/20**, but did
not remove the approximately 50 ms overhead. A first-state-row-only variant
returned to 14/20, proving that all proposal rows cannot simply be discarded.

This isolates the next required architecture change: compute the complete
proposal state with stable batched work, but publish only accepted-row state in a
persistent buffer outside the Graph replay critical path. The current code still
updates state, MATE/indexer metadata, cache slots, and attention output in the
same captured forward, so it cannot satisfy both accuracy and the 8 ms path yet.

Additional checks confirm that forcing target q2 Eager while retaining the N=5
draft/accepted-token lifecycle is not equivalent to the q2 baseline (16/20 on
the first 20). A single failing q6 request (index 244) logged accepted rows
`[0]` followed by `[0,1,2,3]`, input KV lengths 113 then 114, and stopped after
six generated tokens. This points to speculative context/state lifecycle
handling, not only arithmetic evaluation.

## Goal continuation: device-position C4 state A/B

The next candidate adds an opt-in graph-safe C4 state kernel. It reads the
absolute start position from a persistent device scalar at replay time instead
of capturing a host position, writes only valid C4 boundary rows, and uses a
graph shape key that includes the boundary count so output shapes stay stable.
The scalar and graph kernels were compared in a MUSA unit test over absolute
starts `0, 1, 2, 3, 127`; state, score, and boundary outputs all had zero
measured difference:

```text
build/gsm8k_candidates/mate_sparse_unit_state_v3_20260923/mate_sparse_test
GraphC4StateMatchesScalarAcrossAbsoluteBoundaries: PASS
```

With `XLLM_DSV4_DYNAMIC_GRAPH_STATE=1`, keyed graph buckets, and q2-style
compressor/indexer state enabled, the q6 Graph+draft first-20 result improved
from 14/20 to **17/20**. Remaining failures were `122, 235, 244`; index 122 is
correct in q2 Eager but wrong with the q6 N=5 draft lifecycle, while 235 is a
known GSM8K semantic convention error (the model includes John among the 20
friends). The device-position kernel therefore removes a numerical state-kernel
difference but does not close accepted-token semantic drift.

Candidate and logs:

```text
build/gsm8k_candidates/dynamic_graph_state_final_candidate_v2_20260923/xllm
sha256 07c7057fc9591e7e06b709d1dc43ed95c10ad4ac84c9fb94efeeb3279d875171
logs/repro_q6_dynamic_state_keyed_v3_20260923/eval20/summary.json
```

This remains an A/B candidate: it has not passed the fixed-100 >=95% gate or
the formal <=13 ms q6 gate, so the overall merge goal remains open.

## Goal continuation: accepted-state and compact-indexer A/B

The first target-only commit was made an explicit opt-in
`XLLM_DSV4_FIRST_TARGET_ONLY=1`. It fixes request bootstrap state while leaving
steady-state q6 draft verification unchanged. On the first 20 fixed indices,
the dynamic-state candidate with first-target protection reached **18/20**;
index 122 moved from 50 to 48. It is intentionally not enabled by default
because target-only verification has approximately 250 ms/token behavior when
forced for every step.

The remaining q6 errors were then isolated to the Graph compact indexer route.
With `XLLM_DSV4_DISABLE_GRAPH_COMPACT_INDEXER=1`, the same candidate answered
indices 122, 235, and 244 correctly in isolated requests. However, a partial
fixed-100 run exposed additional long-context failures (the run was stopped
before completion), so this A/B is not an accuracy gate and the indexer cannot
simply be disabled as the production default without a performance/accuracy
tradeoff measurement.

The standalone C4 state unit remains green:

```text
build/gsm8k_candidates/mate_sparse_unit_state_v3_20260923/mate_sparse_test
GraphC4StateMatchesScalarAcrossAbsoluteBoundaries: PASS
```

The latest source retains all experiments behind explicit environment switches;
the verified default q2 95/100 and q6 8.127 ms results are unchanged. The
requested unified >=95/100 plus <=13 ms q6 result is still **not complete**.

## Goal continuation: request-boundary reset and current Graph fixed-100

The main accuracy issue was request-to-request compressor/indexer/decode
workspace contamination. With current Attention, Graph compact verify,
batched causal MATE, dynamic Graph state, and request-boundary reset enabled:

```text
XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY=1
XLLM_DSV4_DISABLE_GRAPH_Q6_FREEZE=1
XLLM_DSV4_DYNAMIC_GRAPH_STATE=1
XLLM_DSV4_ENABLE_GRAPH_BATCHED_MATE_VERIFY=1
XLLM_DSV4_CLEAR_STATE_CACHE_ON_REQUEST=1
XLLM_DSV4_CLEAR_DECODE_WORKSPACES_ON_REQUEST=1
```

the fixed GSM8K 100-question selection completed with **97/100 (97%)**, no
transport errors, and failures only at `273, 952, 1038`. The full artifact is
`logs/repro_q6_batched_mate_clearstate_fixed100_20260924/results.jsonl`.

The same configuration was measured with the retained `2500/1500` long
performance protocol (prefix-cache rate 0.65, count-to 750). The strict
counting request passed, but measured **TTFT 603.017 ms / TPOT 50.729 ms**.
Thus the accuracy gate is above the q2 Eager 95/100 baseline, while the
requested current Graph gate (`TTFT <250 ms`, `TPOT <13 ms`) is still not met.
The historical isolated legacy-Attention result (about 217.8 ms / 8.13 ms)
remains a performance reference and is not the merged current-Attention result.

The true Graph compact fixed-100 run was stopped after 67 completed requests
once it had already accumulated at least eight failures (`235, 244, 333,
570, 585, 690, 692, 882`). This confirms that the 97/100 result above is an
Eager target-verify result, not a Graph accuracy result. The Graph precision
gate therefore remains open.

The current Graph runtime also clears persistent accepted-token, linear-state,
KV-length, block-table, and paged-KV metadata carriers at DeepSeek request
prefill boundaries. The candidate was built as
`build/gsm8k_candidates/current_attention_graph_persistent_reset_20260924_b/xllm`.
A focused `122,235,244` sequence still produced `50,11,20`, so persistent
metadata residue is only one contributing factor; the Graph accuracy gate is
still open.

Graph capture warmup/FFI-recording forwards are now prevented from mutating
DSV4 compressor/indexer state; only the actual captured forward may advance it.
The targeted sequence remained `48,11,17`, so warmup mutation was not the
sole cause, but the change removes a confirmed state hazard.

The Graph executor now also recognizes the non-block-parallel DSpark cumulative
KV layout `[0, seq0, seq0+seq1, ...]` when compacting multi-sequence metadata,
instead of silently assigning `kv_max_seq_len` to every sequence. The change
was compiled and linked in `build/gsm8k_candidates/runtime_graph_seq_cumulative_20260924/xllm`.

Two performance A/Bs were rejected as unsafe: current-Attention Graph capture
aborts with the historical `max_tokens_per_batch=10240` configuration even
when q6 state freezing is enabled, and the experimental cross-layer MATE
metadata sharing switch (`XLLM_DSV4_MATE_SHARE_METADATA=1`) aborts during the
2500/1500 request. Both switches remain non-default; no unsafe result is used
for the performance gate.

## Goal continuation: current-Attention fixed-set regression

The newest current-Attention candidate was rebuilt with the verified stream
kernel archive and tested with:

```text
XLLM_DSV4_FIRST_TARGET_ONLY=1
XLLM_DSV4_DISABLE_GRAPH_COMPACT_INDEXER=1
XLLM_DSV4_DYNAMIC_GRAPH_STATE=1
```

The fixed 100-question selection was stopped after 90 completed requests due to
several very long generations. It scored **81/90 (90.0%)**, with failures:
`144, 163, 331, 590, 611, 672, 782, 894, 1047`. A reference-free arithmetic
audit flagged only `611` (plus a correct response at `702`), so most failures
are speculative semantic/state drift rather than simple arithmetic kernel
rounding. This is diagnostic evidence only and does not satisfy the 95/100
gate.

The same candidate's short Graph replay showed approximately 80--100 ms per
six-row target verify, versus approximately 33 ms for the legacy q6 Attention
A/B. Thus the current precision-protection path still needs a lower-cost
accepted-state/indexer design before the 13 ms TPOT requirement can be met.

The latest 90-request slice was `81/90` correct (90.0%); it was stopped before
the planned 100 requests after several 145--163 second long generations. The
corresponding reference-free arithmetic audit flagged only two records, which
confirms that most misses are state/speculative semantic drift rather than a
single arithmetic kernel defect. No fixed-100 acceptance claim is made from
this incomplete run.

## Goal continuation: Graph capture dependency fix

The q6 performance replay initially failed at capture with
`dependency created on uncaptured work in another stream`. The source was the
MUSA MODEL1 FP8 cache-pack completion event: an eager event was recorded or
waited from the capture stream. Both paths are now capture-aware: pack keeps
its kernel in the graph but skips the eager completion event during capture,
and the consumer skips a stale eager wait while capture is active.

The cache kernel stream selection was also corrected: during capture it now
uses the native MUSA current stream, while eager execution retains the
CUDA-compatible mapped stream. The new unit tests all pass:

```text
GraphC4StateMatchesScalarAcrossAbsoluteBoundaries: PASS
CapturedCompressionReplaysChangedPositionsAndClearsPadding: PASS
CapturedPackIgnoresStaleEagerEvent: PASS
```

An important qualification was found during profiling. The current Attention
q6 path replays at roughly 80--100 ms for a six-row verify bucket, whereas the
legacy q6 Attention A/B with the same current runtime/kernel archive replays at
roughly 33 ms and reports HTTP TPOT around 8.4 ms. The legacy path remains
semantically below the required GSM8K gate, so this is diagnostic evidence, not
a valid merged result.

The rebuilt candidate reached `MUSA graph capture end` without that dependency
failure. The subsequent six-request replay produced 1500 server tokens per
request, but the strict visible-output checker did not pass (the response text
was truncated/duplicated), and measured generation wall time was about 262
seconds per request. This is a capture-stability improvement, not a valid
TPOT result; the 8.127 ms historical number remains the only qualified q6
performance evidence.

## GPU recovery

After the tests, all eight GPUs were returned to the reservation program:

```text
PID 45280, 70 GiB reserved per GPU, devices 0--7
```

No GPU reset was performed.

## Goal continuation: Graph request-order and dynamic Indexer investigation

The Graph-key trace showed that the original current-Attention path reused only
two target Graphs for a six-row verify (`key=97/98`, split only by C4 boundary
count). A single request at index 244 could be correct, while a fresh
122->235->244 sequence could fail at 244; this demonstrates request-order
dependence rather than a fixed arithmetic answer error. Enabling synchronous
Graph-input waits and disabling Prefix Cache did not change the failure.

The first suspected static input was the fused Indexer TopK valid-row count:
`build_indexer_topk()` passed host `q_abs_start` into the captured fused radix
kernel. A later replay therefore kept the first request's C4 valid count and
could omit newly compressed history. The candidate now provides a separate
`deepseek_v4_indexer_topk_graph()` ABI with a device `dynamic_kv_len`, uses a
bounded capacity bucket in the Graph key, and updates MATE SWA physical mapping
from the same replay-time device length.

Focused unit coverage passed in
`build/gsm8k_candidates/dynamic_indexer_unit_v2_20260924/mate_sparse_test`:

```text
GraphC4StateMatchesScalarAcrossAbsoluteBoundaries: PASS
CapturedCompressionReplaysChangedPositionsAndClearsPadding: PASS
CapturedIndexerUpdatesLengthsAndBlockTables: PASS
```

However, the fresh current-Graph DSpark sequence still did not pass the
accuracy gate (`122/235/244 = 1/3` in the latest v3 candidate; index 235 alone
still produced 11 instead of 10). MATE dense verification changed the output
but did not correct this case, so the remaining error is not solely the MATE
FlashMLA arithmetic path. No Graph accuracy or TPOT acceptance claim is made
from this candidate; the q2 Eager fixed-100 result remains 97/100 and the
historical legacy q6 217.8 ms/8.13 ms result remains reference-only.

The latest focused candidate is:

```text
build/gsm8k_candidates/current_attention_graph_dynamic_indexer_v3_20260924/xllm
```

After diagnostics, GPUs 0--7 were returned to the reservation program; no GPU
reset was performed.

## Goal continuation: merged candidate performance recheck (2026-09-24)

The current source was linked with the isolated legacy q6 Attention object so
that q2/Eager keeps the current accuracy-protection implementation while Graph
target-q6 uses the historical fast Attention implementation. The candidate was
rebuilt as:

```text
build/gsm8k_candidates/current_graph_short_eager_legacyq6_basearchive_20260924_v2/xllm
```

The fixed GSM8K 100-question run for the short full-KV protection configuration
completed with **97/100**, failures `273, 952, 1038`, and no transport errors:

```text
logs/repro_merge_legacyq6_short_eager_20260924/fixed100.jsonl
```

This confirms short-KV Eager precision for
`current_graph_short_eager_legacyq6_20260924_v3/xllm` (the binary recorded by
that service), NOT the later `basearchive_20260924_v2` candidate.

For the long q6 performance recheck, the current candidate with
`max_tokens_per_batch=1024`, Graph enabled, and legacy q6 enabled reached the
server request completion for 2500/1500. The server-side request log measured:

```text
TTFT = 798 ms
TPOT = 47.9 ms
generated tokens = 1500
```

The corresponding server log is
`logs/repro_merge_legacyq6_basearchive_20260924_b/rank0.log`. The client-side
probe did not receive a valid visible SSE completion (`probe2500.json` records
"stream completed without a content token"), so this is diagnostic server
timing rather than a formal performance gate. It still does not satisfy the
TPOT goal.

Using the historical `max_tokens_per_batch=10240` setting does not produce a
valid result on the current Graph executor: capture aborts with
`dependency created on uncaptured work in another stream` before the request
can be evaluated. Artifact:
`logs/repro_merge_legacyq6_basearchive_20260924_10240/rank0.log`.

The old hybrid executable from 2026-09-23 still has the reference hot-window
result `TTFT 217.83 ms / TPOT 8.127 ms`, but it was built from an earlier
Attention/runtime snapshot and is not evidence for the current merged
candidate. The unified goal remains open; 47.9 ms is server-only diagnostic
timing, not a valid current q6 performance gate.

## Goal continuation: batch-size and capture isolation (2026-09-24)

To separate batch configuration from source changes, the old hybrid executable
was rerun with the same model and draft settings but
`max_tokens_per_batch=1024`. Its 2500/1500 server request measured
`TTFT=8248 ms`, `TPOT=86.6 ms`, and generated 1500 tokens. **Correction:**
this launch omitted compact Graph opt-in and ran target verify Eager. It
does not isolate the effect of 1024 versus 10240 (see the evidence correction
at the top of this document).

The current basearchive hybrid candidate with `max_tokens_per_batch=10240`
begins capture but aborts with
`dependency created on uncaptured work in another stream`. Replacing only
`deepseek_v4_dsa_cache_ops.cu` in the kernel archive removes the original
link-time mismatch but still fails during capture with a MUSA runtime error.
These A/Bs are retained as diagnostics and are not performance gates.

An archive containing capture-safe cache-pack/event changes and an additional
device-position compressor kernel (not solely an event-only patch) was then
linked with the historical q6 hybrid Attention. With Graph compact verify
enabled and `max_tokens_per_batch=10240`, capture completed, but the formal
2500/1500 probe measured `TTFT=8336 ms`, `TPOT=30.3 ms`, and only `724/750`
strict counting tokens. This proves that skipping the cross-stream event is
necessary for capture, but is not sufficient to recover the historical 8 ms
replay or its output stability. Artifact:
`logs/repro_capture_safe_archive_graph2_20260924/probe2500.json`.
