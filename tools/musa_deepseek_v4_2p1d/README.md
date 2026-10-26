# DeepSeek-V4 MUSA 2P1D + PCP benchmark

This directory contains the reusable 40K/100, 80% prefix-sharing benchmark.
Machine-specific process dumps, binary patches, and historical response archives
are kept with the run evidence rather than checked into this toolkit.

## Build and deployment configuration

Initialize the pinned dependencies recursively, including Mooncake's nested
`yalantinglibs` and `pybind11` submodules:

```bash
git submodule update --init --recursive
```

Configure separate build directories with the installed MUSA compiler/toolchain,
TorchMusa, MCCL, mudnn, and C++ dependencies. Load the site's validated CMake
arguments into the Bash array `MUSA_CMAKE_ARGS` first; the compiler, TorchMusa,
vcpkg, and library options are required and are not selected by this benchmark.
The run's build evidence records the complete configure argument array and
environment. Pass the role explicitly:

```bash
declare -p MUSA_CMAKE_ARGS >/dev/null || exit 1
cmake -S . -B build_prefill -G Ninja -DUSE_MUSA=ON -DUSE_CUDA=OFF \
  -DXLLM_DSV4_PD_ROLE=prefill "${MUSA_CMAKE_ARGS[@]}"
cmake --build build_prefill --target xllm --parallel 24
cmake -S . -B build_decode -G Ninja -DUSE_MUSA=ON -DUSE_CUDA=OFF \
  -DXLLM_DSV4_PD_ROLE=decode "${MUSA_CMAKE_ARGS[@]}"
cmake --build build_decode --target xllm --parallel 24
```

The generated `xllm/core/layers/musa/dsv4_pd_layer_selection.txt` records which
role-specific source files were compiled. Keep each role paired with its own
Mooncake `libasio.so`; do not substitute Python Mooncake extensions.
The current MUSA target links custom all-reduce operators itself. Remove the old
external `libxllm_custom_ar_sglang_current.so` entry from `LD_PRELOAD` when using
this build, or PyTorch will reject duplicate operator registration at startup.

| Role | Devices | Attention | MoE | Graph | Speculation |
|---|---:|---|---|---|---|
| Prefill A | 8 | CP2, TP4 per group | TP8, EP1 | enabled | target Prefill lane |
| Prefill B | 8 | CP2, TP4 per group | TP8, EP1 | enabled | target Prefill lane |
| Decode | 8 | TP8, CP1 | TP8, EP1 | enabled | DSpark, 6 draft tokens |

Use `XLLM_DSV4_PCP=1`, `--enable_prefix_cache=true`, `--kv_cache_transfer_mode=PULL`,
and a common model, tokenizer, control-plane instance, and etcd namespace.
Prefill uses 16384-token chunks and disables scheduling overlap; Decode enables
overlap. Device memory limits, library paths, kernel dispatch settings, and all
remaining flags must come from a matching validated deployment profile.

Keep `XLLM_DSV4_GRAPH_JOINT_COMPRESSOR_PROJECTION=0` in the accuracy-first profile.
The layout and request-local GEMM fixes remain available for explicit opt-in,
but passing component tests does not establish whole-model accuracy equivalence.
Do not turn this option on solely for a better throughput result.

## Run the fixed concurrency sweep

The client needs Python 3.10+, `aiohttp`, and `tokenizers`. Set the API origin and
the local model/tokenizer directory. Use a new output directory for every run:

```bash
export API_URL=http://10.121.38.90:24888
export MODEL_DIR=/path/to/DeepSeek-V4-Flash-0731-FP8-mt
export MODEL_ID=DeepSeek-V4-Flash-0731-FP8-mt
export RUN_DIR=/path/to/new/run
python3 tools/musa_deepseek_v4_2p1d/bench.py --repo-root "$PWD" \
  --prepare-only --base-url "$API_URL" --model "$MODEL_ID" \
  --tokenizer "$MODEL_DIR" --output-dir "$RUN_DIR"
python3 tools/musa_deepseek_v4_2p1d/bench.py --repo-root "$PWD" \
  --execute-prepared --output-dir "$RUN_DIR"
```

Start from fresh role processes for cold primers. The run sends two cold primers,
two warm probes, then concurrency 1→5 and 5→1, 32 requests per stage: 320 formal
requests. Each formal request must have exactly 40000 prompt tokens, 100 visible
output tokens, and 32000 cached tokens. Preserve raw SSE, usage, failed requests,
and every attempted stage; do not reuse a partly executed output directory.

The client checks a deterministic counting workload. Run a separately frozen
GSM8K sample with full responses to assess model accuracy. TTFT/TPOT means and
sample accuracy are separate checks; neither establishes worst-case latency or
full-dataset accuracy. Verify both Prefill replicas receive work and each PCP
chunk finishes before accepting a 2P1D + PCP result.
