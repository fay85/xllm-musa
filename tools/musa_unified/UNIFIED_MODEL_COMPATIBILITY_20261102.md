# DeepSeek / Qwen unified-runtime compatibility

This note records the controlled experiment on the `d1029-opt3` image. It is
separate from the original pack qualification because the DeepSeek full PD
acceptance run was not available on a healthy three-host topology.

## Results

| Item | Result | Evidence / limitation |
|---|---|---|
| One Service binary for both models | **Yes, with the Native V2 fix below** | Qwen `xllm_master_serving` Router + Coordinator registered DeepSeek P/D and served a strict `1..10` request. The unmodified binary failed closed with `Native Provider Descriptor has invalid repeated fields`. |
| One `xllm` binary for Qwen P/D | **Yes** | Using the Decode binary for all six Qwen Prefill ranks and two Decode ranks: 68/68 image checks; warmed C6 was TTFT 534.3 ms / TPOT 15.68 ms versus the paired-binary 546.0 / 15.17 ms baseline. |
| One `xllm` binary for DeepSeek P/D | **No, not with the current source** | CMake selects different DeepSeek layer/kernel sources for `XLLM_DSV4_PD_ROLE=prefill` and `decode`; the decode overlay is not a runtime option. |
| One `xllm` binary for both DeepSeek and Qwen | **Not claimed** | It would require merging or runtime-dispatching the role-specific DeepSeek overlay, then a new full P/D acceptance run. |
| AOT in one flat directory | **No** | Six relative paths collide with different hashes, including five `mate_gdn_mtp_checkpoint_hq16_hv48_bf16_t{2..6}` binaries and `MANIFEST.txt`. |
| AOT under one parent with model-specific roots | **Yes** | Keep `aot/deepseek` and `aot/qwen` as separate lookup roots. Do not flatten the six collisions. |

## The Service fix

`native_provider_runtime.cpp` built the Native V2 `cache_groups` list directly
from `manager_types`. DeepSeek's grouped cache can contain the same block type
more than once. Qwen Service validates repeated fields and rejected the
descriptor:

```text
Native Provider Descriptor has invalid repeated fields
```

The fix preserves first-seen order and removes duplicate group IDs before the
descriptor is built. It does not modify model math, attention, MoE, KV values,
or kernel dispatch.

The fixed candidate is built from commit `8ea7284` on branch
`bugfix/unified-service-cache-groups-20261102`. Its P/D hashes are recorded in
the shared evidence directory:

```text
/data/nfs_shared/jd/unified_toolkit_20261031/service_fix_bundle/prefill/xllm
/data/nfs_shared/jd/unified_toolkit_20261031/service_fix_bundle/decode/xllm
```

Both candidates pass `ldd -r` when the image's TVM-FFI and Torch-MUSA library
paths are included.

## Precision checks

- Qwen, one binary for all P/D roles: **68/68** intact-five-image checks.
- DeepSeek fixed Native V2 Service: strict count request `1..10` passed.
- DeepSeek GSM8K smoke: 20 frozen questions produced the expected numeric
  answer on **20/20**; a literal string checker reported 19/20 only because
  one correct answer was returned as `60%` rather than `60`.
- The Service fix changes only descriptor construction. It is not a model
  numerical change. A full 1319-question GSM8K and full 2P1D performance
  acceptance still needs a healthy three-host run.

## Performance checks

The fixed DeepSeek 1P1D short window used `3072/100` on the Qwen Service:

- cold primer: TTFT `680.2 ms`, TPOT `30.68 ms`;
- warmed probe (cache was reported as 0 in this one-P configuration): TTFT
  `565.5 ms`, TPOT `8.68 ms`.

For the Qwen one-binary comparison, the paired-binary baseline was `546.0 /
15.17 ms`; the all-Decode-binary warm result was `534.3 / 15.68 ms`. The
Prefill-binary-for-all experiment was slower (`612.3 / 16.12 ms`), so the
Decode binary is the viable common Qwen binary for this image.

These are warm/cold and model-specific measurements; they must not be mixed
with the historical DeepSeek `20K/100` or GSM8K production figures.
