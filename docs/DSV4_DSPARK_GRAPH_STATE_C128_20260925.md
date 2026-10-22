# DSpark q2 accuracy / q6 Graph merge: state and C128 investigation

Date: 2026-09-25. Checkout: `/data/nfs_shared/jiaming.wang/xllm0914/xllm-musa`,
branch `fix/dspark-accuracy93-perf-20260922`, base HEAD `c7d07f5a`.
Goal: one candidate/configuration must pass fixed GSM8K 100 >=95,
and B1 2500/1500 Prefix Cache 0.65 hot-state TTFT <250 ms / TPOT <13 ms.
No arithmetic/reference-answer correction is used in the experiments below.

**Latest follow-up:** the shared-address candidate passes the joint fixed-100
and performance gates: **96/100, TTFT 215.340 ms, TPOT 11.9828 ms, strict 6/6**.
See [merged acceptance and launch instructions](DSV4_DSPARK_MERGED_ACCEPTANCE_20260925.md).
The sections below preserve intermediate candidates and their limitations.

## Earlier causal-C128 candidate acceptance status

| Requirement | Measured result | Status |
|---|---:|---|
| Fixed GSM8K 100 >=95 | 96/100, zero request errors | met |
| B1 2500/1500 hot TTFT <250 ms | 213.762 ms | met |
| Same configuration hot TPOT <13 ms | 14.385 ms | **not met** |
| Strict count 1--750, exact usage 2500/1500 | 6/6 formal requests | met |
| Hot Prefix Cache | 1536 tokens for every formal request | confirmed |

These are one binary (`621c7b8d...`) and one live service/configuration,
q6 Graph+draft, TP8/EP1, short-Eager protection disabled. Source/header hashes
match the build manifest. All 100 questions, reference answers and prompt-token
counts match the historical q2 Eager 95/100 records. The current `baseline`
prompt string equals the old evaluator's `reasoning` string despite the label
difference; there are no few-shot examples or reference-based retries.

Formal performance: `logs/goal_causal_c128_graph_20260925/formal_2_7/summary.json`.
Output throughput is 68.878 tok/s; input+output throughput is 183.674 tok/s,
whole-service aggregate including cached input tokens. The `gate: PASS` in
the replay utility checks counting/usage/completion, **not** the goal's latency
thresholds. At that stage the combined goal remained ACTIVE because TPOT needed a further
reduction of more than 1.385 ms (about 9.6%).

## 1. Historical speed is reproducible, not current accuracy acceptance

`logs/goal_samehash_shortfirst_20260925/canonical_0_7/` contains eight exact
canonical requests, all strict counting passes and usage 2500/1500. Hot variants
2--7 give TTFT **217.323 ms**, TPOT **7.853 ms**, each with 1536 cached tokens.
Binary: `build/gsm8k_candidates/repro_q6_legacy_20260923/xllm`, SHA256
`ca820538d9500e2c7a33eca24f791026b214db82dacaff14661bb1d14955e08f`.
It actually captured/replayed Graph, with compact verify and legacy q6 enabled.
This does NOT establish q6 GSM8K accuracy (historical first-20 was 13/20).

## 2. Serial state-block lookup is a measured performance bug

`deepseek_v4_sp_pad.cu` previously scanned all 8193 block-table capacity columns
in thread zero on each compressor/indexer state gather. A parallel 256-thread
last-valid-logical-column reduction preserves holes, nonmonotonic physical IDs,
empty tables, and previous-page source selection at a page boundary.

| GPU microbenchmark, 8193 columns / 8 state rows | Mean per call |
|---|---:|
| Old serial lookup | 1449.05 us |
| Parallel lookup | 18.314 us |

The old/new tests are under `build/gsm8k_candidates/state_lookup_{old,new}_unit_20260925`.
The new suite passed 8/8, including changed positions/table contents under Graph.
This approximately 79x gain is for one kernel, NOT model TPOT.

End-to-end comparison (same launch flags and GSM8K request order 122,235,244):

| Dataset index | Old latency | Parallel lookup latency | Text identical |
|---|---:|---:|---|
| 122 | 19.414 s | 15.575 s | yes |
| 235 | 6.624 s | 3.796 s | yes |
| 244 | 121.162 s | 69.219 s | yes |

Artifacts: `logs/goal_swa_publish_graph_20260925/probe3.jsonl` and
`logs/goal_parallel_state_graph_20260925/probe3.jsonl`. Both score 1/3.
The lookup-only candidate is
`build/gsm8k_candidates/current_graph_parallel_state_20260925/xllm`, SHA256
`06f6994bd241c553eff95d14a2a4d659901f4111618cf63732793545a0ace354`.
Its canonical 0--2 counting probe passed 3/3 with TPOT 31.186 ms, TTFT 744.627 ms,
but **zero cached tokens**, so it is NOT a hot-window acceptance result.

## 3. Two additional captured-state correctness gaps

1. The old Graph key partitioned C4 output shape and Indexer capacity, but not
   the full-KV versus compressed/Indexer branch at the 128-token SWA boundary.
   Those host branches cannot change during replay. The new common policy
   header keeps Attention and Graph-key decisions identical.
2. C128 still ran scalar compression with capture-time absolute positions,
   unlike C4's dynamic device-position kernel. It now uses the already tested
   C128 dynamic sequence kernel when dynamic Graph state is enabled. C128
   output count (zero/one for q6) also enters the Graph key.

The policy unit passes, and captured compression replay matches scalar C4/C128
updates across changed positions (including 127/128 and 439/440), FP32/BF16,
and int32/int64 device positions.

Candidate `build/gsm8k_candidates/current_graph_dynamic_c128_key_20260925/xllm`,
SHA256 `98bea425ff8adf2d657d27780fa827a992b5298b192fcceab07201f3271453cd`,
scores **2/3** on the same targeted requests: 122=48 correct, 235=11 incorrect,
244=20 correct. Index 244 now completes in 7.535 s / 177 tokens, compared with
69.219 s / 1714 tokens before this fix. This is a diagnostic improvement only.

Artifacts: `logs/goal_dynamic_c128_key_graph_20260925/`. This run uses the
historical performance prefill budget 10240 instead of the previous 1024;
these three short GSM8K prompts (97--112 tokens) do not require chunking under
either budget. This configuration difference is retained in the manifests.

## 4. C128 per-row causal cache visibility: candidate awaiting validation

The scalar paged fallback also passes the same new compressed tail to all six
queries, and captures a host compressed-history length. Early queries can see
a later compression boundary, while later replays can omit newly completed
C128 rows. A new private MUSA API publishes the current compressed output first,
then masks SWA/C128 history using each query's device KV length. The existing
FP32-reduction warp kernel is reused, with a query grid dimension so six rows
execute in one launch. Its old scalar entry points retain their behavior.

The added capture/replay unit compares against independent FP32 dense Attention
with sink, changed logical/physical block maps, per-row lengths spanning C128
boundaries, FP32/BF16 values and int32/int64 tables. **10/10 tests passed**, no
skips: `build/gsm8k_candidates/dynamic_paged_c128_unit_20260925/all_tests.log`.
The new C128 Attention test covers 144 query rows under captured replay; its
absolute-error limits are 2e-5 (FP32) and 2e-3 (BF16), against the independent
dense reference, not the scalar implementation being modified.

Candidate: `build/gsm8k_candidates/current_graph_causal_c128_20260925/xllm`,
SHA256 `621c7b8d80654fd4f77fc1cc157c2d3542608ad2f60cfeaeb899280a39c4948e`.
The source-complete build records the new policy-header snapshot/hash as well
as the selected kernel archive and four rebuilt source files. E2E validation
is recorded under `logs/goal_causal_c128_graph_20260925/`; see the latest status
table above for the completed accuracy and performance windows.

### Fixed-100 result on the causal C128 candidate

Completed: **96/100 (96%), zero request errors**, failed indices
`235, 333, 952, 1038`. Artifacts: `fixed100`, `fixed100.jsonl`, and
`fixed100.config.json` under `logs/goal_causal_c128_graph_20260925/`.
Same selected indices as the q2 Eager 95/100 baseline; N=5/q6, Graph enabled,
short-Eager protection explicitly disabled, no legacy Attention object.
The fixed-set accuracy gate is met on this candidate. The formal TPOT is
14.385 ms, so the performance gate is not yet met.

Strict-scoring misses are retained without changing references: 235 includes
John in the pizza count; 333 generates `1350 + 2100 = 3350`; 952 interprets
8 AM--11 PM literally as 15 hours whereas the dataset answer uses 3 hours;
1038 divides opportunity cost by incremental rather than total post-college
income. These observations describe the generated answers, not proof that
every upstream operator is correct. The kernel-unit evidence has its own
explicit scope and tolerances.

### Rejected-prefix negative control (not a fixed integration)

`accepted_prefix_negative_unit_20260925/negative_control.log` demonstrates a
separate state-lifecycle risk: process six proposed rows, accept only one, then
continue without restoring the accepted recurrent state. Against processing
only the accepted prefix, compressed output differs by **0.187227 (C4)** and
**0.00613724 (C128)**. This test intentionally expects that difference; its
PASS means the negative control reproduced a hazard, NOT that rollback is
fixed. It does not establish that any particular GSM8K failure is caused by
this condition. Accepted-prefix checkpoint/restore needs a separate worker
integration investigation if the full regression or robustness checks expose it.
The diagnostic unit ran briefly during the accuracy run, not during the
formal performance window; GSM8K latencies are not used as performance gates.

## Work identified before the shared-address follow-up

- Reduce q6 verify cost further without changing the validated state/causal
  behavior. Long-window target replay is approximately 72 ms; the residual
  gap should be profiled across C4 per-row Indexer preparation, MATE metadata
  and draft/host overhead before attributing it to MoE or collective bandwidth.
- Any metadata-sharing optimization must refresh metadata inside each capture
  and on each replay, not rely on capture-time host KV length. The earlier
  `MATE_SHARE_METADATA` experiment is not enabled in the accepted-accuracy run.
- Verify the rejected-prefix recurrent-state hazard in the actual worker loop
  and add a safe accepted-prefix restoration strategy if confirmed there.
- Re-run fixed-100 and the full six-request hot window on each final candidate;
  do not mix this candidate's accuracy with another candidate's speed.

## Follow-up: reuse paged row addresses inside C128 Attention

The fused paged Attention context loop resolved the same logical token's page
address for every output dimension, including repeated 64-bit block/page
division. The candidate stores each valid row address once in shared memory
during score computation and reuses it during value accumulation. Addresses
are recomputed on device on every replay; no host position or block mapping
is frozen. Softmax and accumulation order are unchanged. Shared pointer storage
is 8-byte aligned even for odd-length scalar histories; the existing 512-row
fused-capacity bound is retained.

GPU microbenchmark: q6, 8 heads, head dimension 512, 128 SWA + 64 compressed
capacity, BF16/int32, 320 calls in captured Graph, runtime KV length 2500:

| Implementation | Mean call time |
|---|---:|
| Repeated address calculation | 968.674 us |
| Shared row addresses | 260.929 us |

Artifacts: `build/gsm8k_candidates/c128_address_old_unit_20260925/run.log` and
`build/gsm8k_candidates/c128_address_new_unit_20260925/all_tests.log`.
The new suite passes 10 positive tests plus the separately identified
rejected-prefix negative control (11 reported PASS, no skips). This 3.71x
kernel speedup is not a model TPOT result. Source-complete model candidate
`current_graph_shared_address_20260925` subsequently passed fixed-100 **96/100**
and the full six-request hot window **215.340 ms / 11.9828 ms**, with no Eager
compact-verify fallback and all count/usage checks passing. Run directory:
`logs/goal_shared_address_graph_20260925/`; binary SHA256
`5785ab8228debdd5e5e9a9b24be91a53ba650f913d947d225cd0b11052a43f52`.

## End-of-run resource state

The `goal_causal_c128_graph_20260925` service and all eight ranks were stopped
after the completed formal window. GPU reservation was restarted using the
existing reservation program (PID 34941, devices 0--7, 70 GiB requested per
device). No GPU reset was performed. PID and live GPU ownership must be
revalidated before the next experiment; this entry is historical run state.

## Reproduction/provenance

All builds are private under `build/gsm8k_candidates/`, reusing existing Torch,
MUSA, MATE AOT and other dependencies read-only. No Mooncake rebuild, no GPU reset.
The launcher records the binary hash, AOT hashes, topology and routing flags.
Current source-complete candidates rebuild Attention, DSA metadata, MUSA Graph
executor and DFlash together; no historical Attention object or fake kernel
ABI shim is used in the candidates in sections 2--4.

Do not remove the short-Eager protection default based on these targeted tests.
The current diagnostic services explicitly use `GRAPH_SHORT_FULLKV_EAGER=0` to
test pure q6 Graph; a complete fixed-100 run and hot performance window remain
required on one final candidate/configuration.

Rebuild the section-4 candidate from the current source without changing donor
archives (choose new output names if these directories already exist):

```bash
python3 tools/build_gsm8k_isolated_candidate.py \
  build/gsm8k_candidates/dynamic_paged_c128_kernel_20260925 \
  xllm/core/kernels/musa/deepseek_v4_dsa_attention.cu \
  --kernel-base-archive build/gsm8k_candidates/parallel_state_lookup_kernel_20260925/libcuda_kernels.a
python3 tools/build_gsm8k_isolated_candidate.py \
  build/gsm8k_candidates/current_graph_causal_c128_20260925 \
  xllm/core/layers/musa/deepseek_v4/deepseek_v4_attention.cpp \
  --kernel-archive build/gsm8k_candidates/dynamic_paged_c128_kernel_20260925/libcuda_kernels.a
```

Diagnostic launch after releasing the reservation and verifying ports/devices:

```bash
env \
 DSPARK_BIN="$PWD/build/gsm8k_candidates/current_graph_causal_c128_20260925/xllm" \
 MODEL_PATH=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt \
 XLLM2_EXTERNAL_TP8_HTTP_PORT=19250 \
 XLLM2_EXTERNAL_TP8_MASTER_ADDR=127.0.0.1:21450 \
 XLLM_DSV4_ENABLE_GRAPH_COMPACT_VERIFY=1 \
 XLLM_DSV4_ENABLE_GRAPH_BATCHED_MATE_VERIFY=1 \
 XLLM_DSV4_GRAPH_SHORT_FULLKV_EAGER=0 \
 XLLM_DSV4_DISABLE_GRAPH_Q6_FREEZE=1 \
 XLLM_DSV4_DYNAMIC_GRAPH_STATE=1 \
 XLLM_DSV4_GRAPH_DYNAMIC_INDEXER=1 \
 XLLM_DSV4_CLEAR_STATE_CACHE_ON_REQUEST=1 \
 XLLM_DSV4_CLEAR_DECODE_WORKSPACES_ON_REQUEST=1 \
 XLLM_DSV4_GRAPH_KEY_TRACE=1 \
 XLLM_GRAPH_TIMING=1 XLLM_GRAPH_TIMING_ALL=1 \
 bash tools/start_dspark_custom.sh logs/goal_causal_c128_graph_20260925 10240 0.8
```

The launcher refuses an existing run directory or a live reservation. The
same service is used for accuracy and performance, with no flag changes:

```bash
python3 tools/eval_gsm8k_goal.py \
 --dataset logs/gsm8k_eval_20260914_1925/gsm8k_test.jsonl \
 --url http://127.0.0.1:19250/v1/chat/completions \
 --model DeepSeek-V4-Flash-FP8 \
 --output logs/goal_causal_c128_graph_20260925/fixed100 \
 --samples 100 --seed 20260914 --max-tokens 2048 \
 --reasoning-effort high --prompt-style baseline --timeout 180
python3 tools/replay_dspark_perf.py \
 --url http://127.0.0.1:19250/v1/chat/completions \
 --input-dir logs/goal_repro_q6_legacy_perf_20260923 \
 --output-dir logs/goal_causal_c128_graph_20260925/warmup_0_1 --first 0 --last 1
python3 tools/replay_dspark_perf.py \
 --url http://127.0.0.1:19250/v1/chat/completions \
 --input-dir logs/goal_repro_q6_legacy_perf_20260923 \
 --output-dir logs/goal_causal_c128_graph_20260925/formal_2_7 --first 2 --last 7
```
