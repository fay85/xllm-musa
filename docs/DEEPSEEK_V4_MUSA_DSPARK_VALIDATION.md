# DeepSeek-V4-0731 MUSA DSpark port: work in progress

## Scope and acceptance

- Independent worktree: `xllm0814/xllm-dsv4-musa-dspark`.
- Branch: `dsv4/feat/musa-dspark-20260910`, based on perf checkout
  `0dacf20781d43a8f2c05fea47be4d76b87ebc434`. The original perf checkout,
  including its uncommitted MoE workspace and tensor-helper changes, is untouched.
- Model: `/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`.
  Config SHA256: `9a387b757dc12751447ec0f450a30929b3bfab17f2ee4b9a9a3372300c373fa4`.
- Required comparison: 8 cards TP8/EP1, 2500 input / 1500 output tokens,
  batch 1, prefix cache 0.65 with measured 1536-token hits, Full Graph,
  hot TTFT <250 ms and TPOT <14 ms. Retain the original documented
  <=13.758500 ms TPOT non-regression gate as well.
- Retain complete six-request hot windows, all warmups/failures and strict
  1..750 checks. The old model's Run539/Run540 numbers are reference only;
  they are not evidence for the 0731 model or this branch.
- No GPU reset or remote push. Use host mthreads-gmi for occupancy.

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
DSpark aux capture currently avoids prefill SP/FC1 sharding until equivalent
gather semantics are validated. Neither decision is a performance completion
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

Next: validate 0731 target-only and DSpark eager counting;
compare proposal acceptance and context-KV results; validate DSpark Full
Graph and prefix-state behavior; then run complete hot latency windows.
There is currently no accepted TTFT/TPOT or 1..750 result for this port.

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
