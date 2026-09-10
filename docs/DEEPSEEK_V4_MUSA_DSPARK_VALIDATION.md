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
- Full xllm link and model execution are NOT yet verified. Mooncake is still
  referenced by this branch's link graph; skipping/removing it is not claimed.
  vcpkg is reused and model weights are not downloaded.

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

Next: finish linking; validate 0731 target-only and DSpark eager counting;
compare proposal acceptance and context-KV results; validate DSpark Full
Graph and prefix-state behavior; then run complete hot latency windows.
There is currently no accepted TTFT/TPOT or 1..750 result for this port.

Prepared runtime tools (not executed until the full binary links):
`launch_eager.sh` selects the 0731 checkpoint, DSpark with five proposals,
TP8/EP1 and no Graph/prefix/overlap for initial correctness isolation.
`benchmark_0731.py` uses the retained counting protocol with the 0731
tokenizer rather than silently using the preview tokenizer.
`host_during_build.txt` retains the host GPU occupancy snapshot.
