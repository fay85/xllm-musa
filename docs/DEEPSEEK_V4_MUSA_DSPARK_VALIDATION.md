# DeepSeek-V4-0731 MUSA DSpark: validated hot-prefix configuration

## Scope and acceptance

- Independent worktree: `xllm0814/xllm-dsv4-musa-dspark`.
- Branch: `dsv4/feat/musa-dspark-20260910`, based on perf checkout
  `0dacf20781d43a8f2c05fea47be4d76b87ebc434`. The original perf checkout,
  including its uncommitted MoE workspace and tensor-helper source changes,
  has not been edited in this continuation. See the original-index warning below.
- Model: `/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`.
  Config SHA256: `9a387b757dc12751447ec0f450a30929b3bfab17f2ee4b9a9a3372300c373fa4`.
- Required comparison: 8 cards TP8/EP1, 2500 input / 1500 output tokens,
  batch 1, prefix cache 0.65 with measured 1536-token hits, target Graph,
  hot TTFT <250 ms and TPOT <14 ms. Retain the original documented
  <=13.758500 ms TPOT non-regression gate as well.
- Retain complete six-request hot windows, all warmups/failures and strict
  1..750 checks. The old model's Run539/Run540 numbers are reference only;
  they are not evidence for the 0731 model or this branch.
- No GPU reset or remote push. Use host mthreads-gmi for occupancy.

## Final measured result

Code revision `a9cd485c`, binary SHA256
`78f4a5acff37b5353510bdf1b4d3a20aa8cb455c6b2e683f584b10d322eceee5`.
Both independent services used the exact same binary/configuration:
TP8/EP1/DP1/CP1, batch 1, static DSpark N=5, target six-token Graph plus
draft eager block, prefill SP, Prefix Cache 0.65, overlap OFF, 2500/1500.

| Independent service | Complete hot window | Mean client TTFT ms | Mean client TPOT ms | Strict full-text accuracy | Cache |
| --- | --- | ---: | ---: | --- | --- |
| dspark_sp_prefix_07 | variants 2–7 | 217.370500 | 8.212000 | 6/6, each 750/750 | 1536 on all six |
| dspark_sp_prefix_08_fresh | variants 2–7 | 215.821167 | 8.188667 | 6/6, each 750/750 | 1536 on all six |

Both windows PASS TTFT <250 ms, TPOT <14 ms and <=13.758500 ms.
Every formal request, including cold/setup variants 0/1, passed exact token
accounting and strict full-text counting. No samples were dropped.
The preview model's MTP reference numbers are comparison targets, not the
source of these results. This is the copied source's freshly compiled DSpark
worker, not a donor compute binary.

**Scope:** these results do not claim the draft is fully Graph-captured, or
general model-quality parity beyond the counting protocol. Target-only eager
remains unvalidated after an observed failure; adaptive, multi-batch, PD, CP,
other models/devices and arbitrary prefix-hit ratios are not acceptance-tested.
The existing compact-Graph compressor behavior is retained from the perf
baseline; broader task/long-context equivalence requires separate validation.
Commands and dependencies: [reproduction guide](DEEPSEEK_V4_MUSA_DSPARK_REPRO.md).

## References and implementation contract

Latest local xLLM reference:
`/data/nfs_shared/jiaming.wang/test/fhxllm/xllm_main/xllm`,
observed HEAD `aca1414044ac999a12b006d35a81e5e51d456377`.
Relevant upstream changes include `67459aba`, `0cfaa26d`, and `afd7ddfb`.
The initial local core-port commits (`9ca21451`, `7a8901ea`) were
incomplete integration snapshots, not validated MUSA support.

SGLang reference: host 10.20.34.25, container `wjm_sg_dspark`,
`/workspace/sglang`, observed HEAD
`07f061297417019e501fd801925af502a2906b7f`. The deployment guide
`Deekseepv4 Flash 单机部署指南（09_04）.md` describes the image and general
MUSA tuning but does not itself specify a DSpark launch command.

The checkpoint describes a five-position DSpark block, noise token 128799,
target layers 40/41/42 and Markov rank 256. It has three draft stages
`mtp.0/1/2`, not the preview checkpoint's serial MTP layer.
Target layer outputs are averaged over HC lanes and concatenated, projected
through `mtp.0.main_proj/main_norm`, then injected into each draft layer's
context KV. The five draft positions attend the same trailing prefix plus the
whole diffusion block; Markov sampling produces proposals sequentially from
the block logits. Target verification has six rows in one request.

The weight index was inspected without loading GPU weights. It contains
`mtp.0.main_proj.{weight,scale}`, `mtp.0.main_norm.weight`,
`mtp.2.markov_head.markov_w1.weight`,
`mtp.2.markov_head.markov_w2.weight`, and a confidence-head weight.
There are no dedicated draft embed/head entries in this checkpoint index;
the loader retains the explicit dedicated-first, target-fallback policy.
The static greedy path is the initial validation scope; adaptive/confidence
sampling is not yet validated.

## Changes being integrated

1. Restore the speculative cache-budget/count helpers omitted by the first
   cherry-pick conflict resolution.
2. Register DSpark on MUSA, parse the checkpoint metadata, and select the
   three-stage, uncompressed SWA draft instead of the MTP model.
3. Reuse the DSpark loader with the MUSA V4 model, preserving draft vocabulary
   ownership and MUSA FP8 scale aliases.
4. Capture the requested target outputs, including final layer 42 before
   the final HC head/norm; return these through aux_hidden_states.
5. Add MUSA draft context KV projection/norm/RoPE/cache write and grouped
   non-causal block attention.
6. Keep target verification grouped per request on MUSA. Increase the
   private MoE workspace from two to sixteen rows for five-/six-row blocks.
7. Size the target Graph auxiliary buffer by the three captured layers
   (12288 columns), not by the four pre-HC lanes used by MTP (16384 columns).

The current draft attention implementation is an eager reference route.
Initial DSpark aux capture avoided prefill SP/FC1 sharding. The next candidate
restores SP and gathers captured layer outputs in global token order; FC1
remains disabled for auxiliary capture. This is not a performance completion
claim: Full Graph, SP-compatible capture, compressed-state evolution,
prefix restoration and the latency gate still require runtime validation.

## Build and environment ledger (2026-09-10)

Artifacts are retained under:
`/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/dspark_20260910`.

- Initial configure mistakenly enabled vcpkg manifest installation against
  shared `musa-standalone-db33797/vcpkg_installed`. It removed 118 packages
  and stopped during a replacement download. This was an unintended shared
  environment mutation, not a model/code failure.
- Recovery: preserve the affected tree as
  `musa-standalone-db33797/vcpkg_installed.dspark-interrupted-20260910`;
  restore the complete `musa-current-main/vcpkg_installed` snapshot and
  the original two vcpkg-cmake packages/status. The two snapshots' package
  status files differ only in those two CMake-helper ABIs. A manifest walk
  after recovery checked 17569 entries with zero missing paths.
- All subsequent configuration uses `VCPKG_MANIFEST_MODE=OFF` and
  `VCPKG_MANIFEST_INSTALL=OFF`. Do not run an install against shared deps.
- New worktree submodules were initially absent; checkout completed and the
  top-level submodule SHAs now match their recorded gitlinks.
- CMake 4.3.2 still probes toolkit metadata despite forced MUSA compiler ID.
  An external toolkit-probe shim is retained in `musa-cuda-overlay/bin/nvcc`.
  It is not NVIDIA nvcc; device compilation uses mcc_wrapper/MusaMapping.
- Early configuration incorrectly fell through to CPU LibTorch 2.1 fetching.
  The final recipe explicitly exports LIBTORCH_ROOT and uses installed
  Torch 2.9 with C++11 ABI. Those failed build directories are not runtime assets.
- Python interpreter, MKL, ASIO and xxHash paths are explicit in
  `configure.sh`. The final build directory is `build/dspark-musa-local`.
- First full component build stopped on cpprestsdk -Werror=conversion in Boost
  headers. An external compiler launcher scopes warning relaxation to the
  third-party compilation and adds the local TorchMUSA/msgpack include roots.
- First targeted compile exposed a JsonReader template default and an NPU-only
  KernelConfig accessor on MUSA. Both were corrected.
- All five targeted objects (model registry, worker_impl, dflash_worker_impl,
  speculative_worker_impl, MUSA attention) subsequently compiled successfully.
  The later compact target-verify edit is also included in the ongoing full build.
  Full-build entries 807 and 829 confirm the Graph executor and compact-verify
  objects were rebuilt successfully after those edits.
- The first full build stopped at entry 993/1030: the initial cherry-pick
  added a one-argument WorkerService member declaration but retained the
  existing two-argument anonymous-namespace metrics function in the cpp file.
  The unimplemented member shadowed that function. Remove the stale member
  declaration and its unused position-label member; retain the original
  static-width metrics implementation and call sites. This does not import
  adaptive acceptance statistics or change worker execution.
  Incremental rebuild is retained as `build_full2.log`.
- Full xllm build/link passed: `build_full2.log` reached 128/128 and exit 0.
  Source at launch: `6eff81e2`; binary SHA256:
  `a893098bfddb8639b265292cccfb5c4c004ad693f45f5e9e0e870e545480e3d4`.
  Bare ldd cannot find TVM-FFI; ldd with the launcher's complete runtime
  library path has no missing libraries (`binary_runtime_ldd.txt`).
  Mooncake is still referenced by this branch's link graph; skipping/removing
  it is not claimed. vcpkg is reused and weights are not downloaded.

Reconfigure using the retained environment recipe:

```bash
bash /data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/dspark_20260910/configure.sh
cd /data/nfs_shared/jiaming.wang/xllmmerge/xllm0814/xllm-dsv4-musa-dspark
MUSAMAPPING_PATH=/usr/local/musa/tools/musamapping MUSA_HOME=/usr/local/musa \
  cmake --build build/dspark-musa-local --target xllm -j16
```

## Checks and next gates

The standalone CPU test `dspark_metadata_contract_test` passed 3/3:
capture order/copy ownership, dedicated-vocabulary precedence, and disabled
aux-capture behavior. XML: `contract_test.xml` in the artifact directory.
These are metadata unit tests, not model precision tests.

Edited C++ files were formatted with clang-format 20.1.6.

Follow-up work outside the measured acceptance configuration includes repairing
target-only eager, capturing the complete draft pipeline and evaluating general
model quality, larger batches and additional prefix-state scenarios.
DSpark 2500/1500 strict 1..750 and two independent complete hot-prefix windows
now pass. Historical intermediate results and failures are retained below.

Runtime tools:
`launch_eager.sh` selects the 0731 checkpoint, DSpark with five proposals,
TP8/EP1 and no Graph/prefix/overlap for initial correctness isolation.
`benchmark_0731.py` uses the retained counting protocol with the 0731
tokenizer rather than silently using the preview tokenizer.
`host_during_build.txt` retains the host GPU occupancy snapshot.
`holder_resume_2038/host.txt` confirms the restarted holder reserves about
61694 MiB per card, with all eight devices at 100% compute
at that sample. Holders are stopped only for this task's model experiments.

## Runtime ledger

- `target_eager_01`: 0731 target-only, TP8/EP1, no Graph/overlap/prefix,
  max batch tokens 10240, block 128, original custom-AR/AOT environment.
  Host snapshots show only this task's holders before release and zero
  allocated GPU memory after release. Stop only the validated holder PGID.
  A first `setsid ... &` shell detached without starting any ranks; retry
  with `setsid -f` is the working persistent launch method. Keep both logs.
  All eight workers loaded successfully; health became ready at 20:48:40.
  The two 64/64 counting requests both failed strict 1..32 (only the first
  number 1 appeared, followed by whitespace/punctuation). Input/output usage
  is exactly 64/64, no prefix hit. Retain `short_01.json` and `short_02.json`.
  Client TTFT/TPOT: 8133.739/40.061 ms (cold), 588.722/39.786 ms (warm).
  Neither failed-accuracy sample is a valid performance baseline.
- `target_eager_01/teacher_prefill.json`: six independent raw-completion
  prefills, each generating one token, returned the expected next token for
  prefixes empty, `1`, `1 `, `1 2`, `1 2 `, `1 2 3 `: respectively
  `1`, space, `2`, space, `3`, `4`. This narrows investigation toward the
  continuous decode path; it does not prove every loaded weight or all
  prefill shapes correct. This failure predates enabling DSpark.
- Stop only this launcher's PID using its worker cleanup trap. `host_after.txt`
  records zero MiB and no remaining GPU processes, without reset.
- `target_graph_01`: same binary/checkpoint/topology and no prefix/overlap;
  change only Graph to true (VMM and decode-no-padding remain enabled).
  Source diff at launch is documentation only. The 64/64 request passed
  strict 1..32, client TTFT/TPOT 8114.772/22.479 ms including cold prefill.
  Log explicitly confirms q1 MUSA Graph capture and replay.
  The 2500/1500 request passed strict 750/750 with exact server token usage;
  client TTFT/TPOT 1524.965/22.392 ms; cached tokens 0. Server-side TTFT is
  separately 1052.0 ms (do not substitute it for client TTFT).
  These are target-only functional checks, not DSpark or hot-prefix performance
  acceptance. The eager decode failure remains open and is not fixed merely
  because Full Graph passes. Cleanup left zero MiB/no processes on the host.
- `dspark_graph_01`: same binary/checkpoint, TP8/EP1, five proposals,
  Graph enabled, no prefix/overlap. This is an initial mixed-route diagnostic:
  target verification is intended to capture six rows, draft still uses the
  eager grouped non-causal block route. Do not call it a fully captured DSpark
  pipeline. Both target and three-stage draft weights loaded, including the
  Markov head and explicit target-vocabulary fallback for this checkpoint.
  Health became ready at 20:56:36. First 64/64 request failed before emitting
  content: `kv_caches.size() == layers_.size() (1 vs. 3)` in draft context
  injection, on all ranks. No Graph verify or accuracy result was produced.
  Root cause: legacy non-NPU MTP layer selection reused
  `num_nextn_predict_layers=1` for the three-layer DSpark draft.
  `host_after.txt` confirms cleanup to zero MiB/no processes.
- Cache-layer repair: preserve the configured DSpark layer count in Worker
  allocation and capacity estimation, and normalize the draft engine's model
  descriptor to DSpark with three SWA stages. This is algorithm/checkpoint
  metadata, not a new MUSA runtime flag. Preserve MTP's legacy count rule.
  Add tests for DSpark with a stale nextn=1 and for MTP compatibility.
  `build_cache_layers.log` linked successfully; new binary SHA256:
  `47d0a8daaeb55e3943784ac4497ad80f89df68bde21ea95952cfc36ed3af7b83`.
  Standalone test-harness setup initially missed generated protobuf/xxHash
  includes, then exposed the lack of a CPU PlatformStream configuration.
  The final harness uses the actual built MUSA estimator object and arithmetic
  layout helpers, running only layer-count contracts without GPU allocation.
  Its first link omitted Python, a dependency of the built estimator object;
  the retained script now includes it. These setup failures are not model
  test results. All `build_cache_contract_test*.log` files are retained.
  The two targeted tests now PASS (2/2), retained in
  `cache_contract_test.xml` and `build_cache_contract_test5.log`.
- `dspark_graph_02`: source `9d4c71d8`, binary `47d0a8da...`, same diagnostic
  configuration. Both engine and worker now report three draft SWA caches
  (`layer_crs: [1,1,1]`), confirming the layer-count repair at runtime.
  First 64/64 request failed before content with `Accessing empty ModuleHolder`.
  Address resolution locates `DeepseekV4AttentionImpl::write_context_kv`.
  The pre-existing packed projection loader unregisters and clears `kv_proj_`
  after packing wq_a/wkv; context injection still dereferenced it. The fix
  selects the KV rows from the materialized packed weight and calls the same
  matmul API as existing prefill; unpacked weights keep the original linear.
  This repair is confined to the MUSA private attention file.
  `build_context_projection.log` retains the incremental build. No tokens,
  acceptance rate or model-accuracy pass can be reported for this failed run.
  Host cleanup again reached zero MiB/no GPU processes; holders were restored
  for compilation (`holder_compile_projection_fix/host.txt`).
- `dspark_graph_03`: source `f7d0177f`, binary SHA256
  `5d88dc124ade8fb7613a1bb8553ca7160ebec3197058b69e0034fc35192cf5ff`.
  Context injection advanced successfully to the first draft decode.
  Draft metadata construction then failed with
  `manager_num(3) exceeds group_infos size(1)`: it inherited the target's
  three composite block tables although all draft layers are SWA-only.
  Preserve the target carrier and filter only the builder's local table copy
  for SWA-only models. The existing V4 registration order is SWA, then C4/C128;
  multi-group target construction keeps its existing checks and mapping.
  Repair is in MUSA private `dsa_metadata_builder_musa.cpp`; incremental build
  is `build_draft_metadata.log`. This run produced no accuracy/performance
  result; all rank/client logs and host cleanup snapshots are retained.
- `dspark_graph_04`: source `5d5c6b27`, binary SHA256
  `b30154076ac69f8540d39963c99614e164b087f6924b3931381bf37628db21b9`.
  The draft forward completed and target Graph preparation began, but failed
  with manager count 6 versus 3. One content chunk `1` reached the client,
  with no server usage/finish event; the client's fallback TPOT=0 is INVALID,
  not a latency measurement or successful request.
  Root cause: speculative preparation called ForwardInput::to on already
  materialized fields while the original serialized-layout flag remained set.
  Re-deserializing into a copy appended the original three manager tables.
  Invalidate that stale layout before preparing target validation; likewise
  invalidate it whenever the shared speculative helper rebuilds token/position
  tensors. Retain the original buffer owner so other host views stay alive.
  This is a device-independent input-lifecycle repair, not a MUSA algorithm
  added to the shared serializer. Add a serialized-layout invalidation test;
  retain build/test evidence under `build_input_layout.log` and
  `build_input_contract_test.log`.
  The model build/link passed, binary SHA256
  `93e902d9892c5862f156db80cc7a300491b19f2c2d6c86d24542ffab0289336d`.
  The focused `spec_input_layout_test.cpp` is included in the normal
  spec_input_builder_test CMake target. Its standalone run passed 1/1, checking
  stale-layout invalidation, backing-buffer lifetime and replacement host
  tokens/positions (`input_contract_test_runtime.log`, XML of the same name
  without `_runtime`). Broader standalone test linking initially required
  unrelated multimodal/protobuf dependencies; the focused regression remains
  a normal source test, not a mocked implementation. The retained runner uses
  matching generated TorchMUSA API headers and preloads torch_python followed
  by musa_python. Keep all failed harness logs as environment setup evidence.
- `dspark_graph_05`: source `d7bcc845`, binary `93e902d9...`. Both repairs
  passed runtime validation: target verification captured/replayed six-token
  Graphs, and the complete three-stage draft/context/Markov/verify loop ran.
  Configuration: TP8/EP1/DP1/CP1, DSpark five proposals, target Graph + draft
  eager block, no prefix/overlap, batch 1, block 128, batch token limit 10240,
  memory utilization 0.8, four response threads. All server usages exact.

  | Request | Client TTFT ms | Client TPOT ms | Strict result | Cached tokens |
  | --- | ---: | ---: | --- | ---: |
  | short_01, 64/64 | 8109.204 | 12.994 | 32/32 PASS | 0 |
  | short_02, 64/64 | 561.333 | 5.570 | 32/32 PASS | 0 |
  | full_01, 2500/1500 | 1422.460 | 8.181 | 750/750 PASS | 0 |
  | full_02, 2500/1500 | 1230.839 | 8.107 | 750/750 PASS | 0 |
  | short_03, 64/64 | 131.662 | 8.792 | 32/32 PASS | 0 |

  Each full request had draft/accepted counter deltas 1250/1250. Only the
  existing fixed-width draft/accepted counters are used: the newly declared
  step/committed gauges are not wired by this compatibility worker and stay 0.
  The first four requests were handled by different request threads; the
  fifth returned to the first thread and removed the roughly 427-ms client
  versus server TTFT gap. This is evidence of settling behavior, not permission
  to mix short requests or remove slow samples from a formal hot window.
  The full requests meet TPOT but NOT TTFT. They are not hot-prefix gate runs.
  Preserve the working executable as `bin/xllm_dspark_functional_93e902d9`.
  Host teardown is clean; holders restored before compilation.
- Next candidate: capture SP-local HC-averaged layer states, gather/unpad them
  before copying to the draft context buffer, and enable existing Prefix Cache
  in the external launcher. Target-only's TP-prefill guard and DSpark draft's
  no-SP route remain unchanged. Decode Graph geometry/algorithms are unchanged.
  Build is recorded in `build_sp_capture.log`; precision and hot latency
  still need validation for this candidate.
- `dspark_sp_prefix_06`: SP candidate source `1d586c44`, binary SHA256
  `4b4d22d8a013be62c54dac3fa11df560bf9707e629ce359ad1e60d71724f97af`,
  Prefix Cache enabled. Startup stopped before requests because the draft
  engine constructed a prefix-enabled SWA-only CompositeBlockManager:
  `SWA prefix cache is on but neither C4 nor C128 is present`.
  Keep that public manager's validation intact. The DSpark draft allocation
  manager takes its request mappings from the target and must not independently
  use the target's compressed-prefix policy. Disable prefix participation only
  for the DSpark draft manager; leave target Prefix Cache enabled. Build:
  `build_prefix_manager.log`. No SP precision or latency result was produced
  by this failed startup; host cleanup and resumed holders are retained.
- `dspark_sp_prefix_07`: source `a9cd485c`, binary `78f4a5ac...`.
  Two 64/64 warmups passed strict full-text 1..32. Complete formal variants
  0–7 all passed strict full-text 1..750 and usage=2500/1500. Cold/setup
  TTFTs 925.316 and 649.802 ms are retained, not part of the hot window.
  Hot TTFTs: 219.413, 219.962, 219.193, 212.830, 216.902, 215.923 ms.
  Hot TPOTs: 8.311, 8.285, 8.169, 8.152, 8.156, 8.199 ms.
  Every hot request reported 1536 cached tokens. `window_2_7.summary.json`
  is PASS. After shutdown, host inspection confirmed zero GPU memory and
  no processes before the independent service was started.
- `dspark_sp_prefix_08_fresh`: identical source/binary/configuration, fresh
  processes and reloaded target/draft weights. Two short warmups and all eight
  formal requests passed. Cold/setup TTFTs 924.949 and 681.381 ms are retained.
  Hot TTFTs: 219.031, 217.122, 211.151, 214.940, 217.633, 215.050 ms.
  Hot TPOTs: 8.223, 8.185, 8.228, 8.166, 8.140, 8.190 ms.
  Every hot request reported 1536 cached tokens. `window_2_7.summary.json`
  is PASS. Host snapshots were collected before/during/after both windows.
- The formal benchmark is a retained copy of the original timing protocol;
  only checkpoint selection and post-timing full-text output/checking were
  added. `strict_text_passed` requires the whitespace-separated output to
  exactly equal canonical strings 1 through 750; usage fallback is rejected.
- `loaded_shared_libraries.txt` in the fresh run records actual mappings,
  including DSA Model1/local-prefill, MLA and TF32 HC prenorm AOT modules beyond
  the launcher's minimal preflight list. Fingerprints are retained in
  `dependency_sha256.txt` and `model_metadata_sha256.txt`.
- Final audit: `final_gate.json` verifies all 16 formal requests and all 12
  hot-window requests; the two summary files both report PASS. All 79 loaded
  shared-library paths have matching SHA256 inventory entries. Script syntax
  checks and git whitespace checks passed. Final documentation changes do not
  alter the measured source revision or executable.
- Final teardown: the fresh service exited cleanly and the host reported zero
  MiB/no GPU processes. Eight 60-GiB holders were then restarted. Snapshot
  `holder_final/host.txt` confirms about 61694 MiB per card (card 0 also has
  the holders' small auxiliary contexts), with all eight at 100% compute at
  that sample. Container holder PGID is 43610, worker PIDs 43612–43619; host
  PIDs differ. Re-verify identity/occupancy before stopping any holder later.
  No GPU reset or remote push occurred.

## Original checkout index warning

The original `xllm-dsv4-musa-perf` source worktree still differs from its HEAD
only in the MoE workspace and tensor-helper files. A final read-only check
also found a separately staged DSpark-related index that does not match those
working files. Its Git directory/index is distinct from this worktree's.
This continuation did not clear/reset that index or commit from the original
checkout. Review `git diff --cached` there before any future original-branch
commit; do not mistake that staging state for this clean DSpark branch.
