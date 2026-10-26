<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0. See LICENSE. -->

# MUSA VL / native Service / VMM source snapshot

This branch preserves the existing `1023-vl-bench-kit` history and imports the
working sources used in the 2026-10-25 dev92 Qwen3.8-27B-FP8 experiments.
It includes the native Service protocol/admission observations, graph VMM pool
implementation and lifetime fixes, and the large-KV/RR deployment configuration.
The serving checkout and published image remain unchanged by this Git export.

Companion Service: [fay85/xllm-service](https://github.com/fay85/xllm-service),
implementation commit `2e19a38ec4cd1890694b1f50367df6e384a8b995` over
`73b7e684b94ef75a0e951ba55fc9af068d9ed956`.
The companion repository is private. Its engine-adapter pin remains
`38859261b070743369bfb1617db20d3bce1b0fe9`; a dedicated
`service-adapter-base-38859261` branch makes that exact dependency fetchable.
Do not replace the adapter pin with this runtime MUSA branch.

## Tested deployment

Three CP2 Prefill instances use GPU pairs 0,1 / 2,3 / 4,5. Two TP1 Decode
instances use GPUs 6 / 7. Prefix caching, RR, credit caps8192 and memory
utilization0.90 remain enabled. Prefill uses piecewise graphs; Decode uses
full graphs. The throughput profile uses MTP K3, VMM enabled and16 requests
per instance.

```bash
export VL_PROFILE=throughput_k3_vmm
# In the published container this versioned executable is already present.
export VL_XLLM_BIN=/opt/xllm/releases/musa-large-kv-rr-20261025/bin/xllm
bash tools/musa_qwen3_8_vl_bench/profile_info.sh
bash tools/musa_qwen3_8_vl_bench/start_stack.sh
```

The default source profile remains `large_kv_rr` (memory0.90, RR, K2,
VMM disabled) so the previous control remains reproducible. The new
`throughput_k3_vmm` profile is the equivalent of the tested image overrides.
Run each concurrency with its own start / four-round bench / stop lifecycle;
reusing identical tails in one running cache changes the hit-rate protocol.
For a fresh build outside that image, set `VL_XLLM_BIN` to the resulting MUSA
binary and supply the SDK/Mate/Mooncake paths described by the benchmark kit.

| C | K2/VMM-off QPS | K3/VMM-on QPS | K3 TTFT ms | K3 TPOT ms |
|---|---:|---:|---:|---:|
|5|1.873|1.961|855.5|16.63|
|6|2.192|2.292|881.2|17.04|
|12|3.438|3.547|1129.6|20.74|

All six table cohorts independently used4x60 full warm requests and60 measured
requests. Every warm/measured request logged40960 input,32768 cached (80%) and
100 actual output tokens. Measured graph captures and fatal errors were zero.
The complete tuning campaign has8 qualified cohorts /2400 verified requests,
plus4 failed attempts preserved in the deployment evidence. These are single
60-request observations, not repeated randomized A/B confidence intervals.

K3 improves this campaign's throughput/TPOT, not C12 TTFT: the K2/VMM-off
reference TTFT is1096.5ms. Reducing the per-instance limit to8 or12 caused
measurement-time OOM after four warm rounds; neither configuration is adopted.
One C6 K3 cold start failed a PD lazy-link RPC during tiny warmup; an independent
cold restart passed. That startup issue has not been fixed by publication.
VLM adaptive speculative decode remains gated unsupported.

## VMM and source identities

Each new capture owns a fresh logical MemPool. Physical pages are shared per
executor/phase. Graph handles reconcile the native pool reference exactly once
on reset, covering the installed SDK omission without replacing SDK libraries.
Graph-owned pools and captured tensors are destroyed before physical storage.

Prior VMM checks:9 CPU cases and3 GPU integration cases passed. Native
piecewise/full VMM execution is additionally covered by the successful VL
cohorts above. The existing DeepSeek/DSpark sources are retained as part of the
working snapshot; these Qwen VL results do not qualify every model/backend.

Engine binary SHA256:
`a05664ed4948bbb1567d07c1f58ca8ccd2716cdc6446b377681a7a2087bc81a0`.
Service binary SHA256:
`7757cf147f0fb39a6f27f6340535f3566985f60c6dc3b6fc81e1c75d86ec6c7b`.
Frozen dataset SHA256:
`0c04ad3ed1767a9d70230d9c8636cba4c3ca0cc07f662461d0f4277169e1a61e`.
Serving source base: `4a975abdd04fa9abfc611614461beae54405add1`, plus its
preserved working overlay. The snapshot commit records that previously dirty
source; only the named profile and publication documentation are added on top.

Local image tag: `xllm-musa:large-kv-rr-mtp3-vmm-20261025`.
Image ID: `sha256:a3fae62e495b728e4f5fe8e9dbf8b358226d9237300623ffeb25789078c3c698`.
This tag exists on dev92, and has not been pushed to a registry. Model weights,
compiled binaries, raw article/prompt corpus, crash dumps and temporary backup
files are not part of this Git publication.

The source build helper uses the installed toolchain. Its local `ninja.real`
binary is not published; provide the system Ninja executable when using the
optional Ninja guard. Source hashes and publication verification are recorded
separately from the earlier execution evidence.
