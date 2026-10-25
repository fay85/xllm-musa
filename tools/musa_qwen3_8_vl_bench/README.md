# Current deployment default: large cache + RR

The default is now the named `large_kv_rr` profile (memory0.90, caps8192,
MTP K2, RR-cursor-patched Service). See [profiles/README.md](profiles/README.md)
for the exact settings, rollback, four-round protocol and historical-result boundaries.
Run `bash profile_info.sh` before launch. The remainder of this document is the
historical benchmark reference; its old "defaults" and short warmup protocol do
not describe the current frozen release.

---

# Qwen3.8-27B-FP8 VL PD benchmark on MUSA

Scripts that build xllm and reproduce the disaggregated prefill/decode (PD)
results for Qwen3.8-27B-FP8 on one host with eight MTT S5000 GPUs:

- **Workload:** closed loop, 40,960 input tokens per request (five 854x480
  images inside), 100 output tokens, 80% KV cache hit.
- **Target at concurrency 5:** average TTFT < 1 s, average TPOT < 20 ms.

The default paths in `env.sh` are the paths inside the benchmark container
image. Change them to match your machine if you run elsewhere.

## Expected results

Measured with `run_sweep.sh 5 6 12` in the benchmark image on dev92 on
2026-10-23. Setup:

- **Build:** `feihu-1016-dev` (`fe740c76`), binary md5 `57581cb4`.
- **Settings:** the defaults of these scripts (`CONC=16`, MTP with 2 draft
  tokens).
- **Data:** the reference data set.
- **Protocol:** 60 measured requests per level, each level on a freshly
  started stack.

| C | TTFT mean | TTFT p50 | TTFT p90 | TTFT p99 | steady TTFT | TPOT mean | TPOT p90 | QPS | KV hit | MTP accept |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 859 ms | 825 ms | 883 ms | 1605 ms | 830 ms | 17.45 ms | 19.41 ms | 1.89 | 80.0% | 0.666 |
| 6 | 885 ms | 832 ms | 926 ms | 1581 ms | 848 ms | 18.31 ms | 19.91 ms | 2.17 | 80.0% | 0.672 |
| 12 | 1123 ms | 877 ms | 1673 ms | 3119 ms | 898 ms | 22.02 ms | 24.02 ms | 3.41 | 80.0% | 0.675 |

- **Reproducibility:** two earlier sweeps with the same binary and prompts
  gave TTFT means of 863-869, 888-898 and 1116-1129 ms and TPOT means of
  17.60-17.61, 18.22-18.41 and 22.32-22.51 ms.
- **Sources of variation:** how requests collide on the prefill instances,
  and the MTP acceptance rate.
- **C=12:** each decode instance carries about six sequences and the three
  prefill instances queue requests, so TPOT and the TTFT tail grow.

## Deployment

| GPUs | Role | Parallelism |
|---|---|---|
| 0-1, 2-3, 4-5 | three PREFILL instances | context parallel over 2 GPUs (`--cp_size=2`), weights not split |
| 6, 7 | two DECODE instances | one GPU each (TP1) |

- **Engine features:** all instances use MTP speculative decoding with 2
  draft tokens, plus the prefix and encoder caches.
- **Prefill:** chunked prefill with 8,192-token chunks and piecewise graphs.
- **Decode:** decode graphs; KV cache is pulled from the prefill instances
  over Mooncake.
- **Front end:** an xllm-service master (round-robin routing) and etcd. The
  master serves `/v1/chat/completions` on `$HOST:64888`.

## Container image

The image `xllm-musa:qwen38-vl-pd-20261023` is the development container the
results were measured in, based on
`registry.mthreads.com/presale/devtech/xllm:musa-mt-main-20261022-pcp`. It
contains:

| Path | Content |
|---|---|
| `/workspace/xllm-mt-main` | xllm source of `feihu-1016-dev`, copied in as files, and its configured build tree; clone the repository for git history |
| `/opt/xllm_service/bin`, `/opt/xllm_service/lib` | xllm-service master (`xllm_master_serving`), etcd, and the `librust_tokenizers.so` the master links against |
| `/workspace/artifacts/libmusa_backend_init.so` | MUSA backend shim preloaded into every xllm process |
| `/opt/mate027_cache/mp31/cached_ops_jd_k8d` | prebuilt Mate 0.2.7 kernels (FA3 attention, GDN, GEMM) that xllm loads instead of compiling |
| `/workspace/xllm_vl_bench_ref` | the reference articles and photos behind the published numbers |

The model is not in the image. Start the container with the GPUs and the
model directory, then point `VL_MODEL` at the checkpoint:

```bash
docker run -it --privileged --network=host --shm-size=64g --ulimit memlock=-1 \
  -v /path/to/models:/models \
  xllm-musa:qwen38-vl-pd-20261023 /bin/bash
export VL_MODEL=/models/Qwen3.8-27B-FP8
```

## Build

Inside the container:

```bash
cd /workspace/xllm-mt-main/tools/musa_qwen3_8_vl_bench
MAX_JOBS=32 bash build.sh
```

`build.sh` prints the binary it produced
(`build/lib.linux-x86_64-cpython-310/xllm/xllm`) and its md5.

- **In the image:** it uses the existing build tree. It sets up the
  environment of the repository's MUSA build (`scripts/build_support`), then
  runs `cmake --build build/cmake.linux-x86_64-cpython-310 --target xllm`.
  - **Offline and incremental:** it needs no network and takes about 2
    minutes when nothing changed.
  - **Same binary:** it reproduces the benchmarked binary, md5 `57581cb4`.
- **In a fresh clone:** there is no build tree, so it runs
  `python3 setup.py build --device musa` instead. Run
  `git submodule update --init --recursive` first.

## Steps

Every script sources `env.sh`. Override a setting by exporting its name with
a `VL_` prefix, e.g. `export VL_MODEL=/models/Qwen3.8-27B-FP8 VL_HOST=10.0.0.1`.
Unprefixed names in the environment are ignored on purpose, because images
predefine some of them (`XLLM_BIN`).

```bash
cd /workspace/xllm-mt-main/tools/musa_qwen3_8_vl_bench
bash prepare_data.sh        # once: master tokenizer and 68 prompts (about 2 minutes)
bash start_stack.sh         # 1-2 minutes, prints READY
bash check.sh               # optional: expects "VL_QA 10/10 passed"
bash bench.sh 5 --out c5.json
bash stop_stack.sh

bash run_sweep.sh 5 6 12    # or all levels, each on a fresh stack (about 11 minutes)
```

- **Output:** `bench.sh` prints one line per measured request, then
  `SUMMARY {...}`.
- **Sweep results:** `run_sweep.sh` writes them to
  `$RESULTS_DIR/sweep_<time>/` and prints a table like the one above.
- **Server logs:** `$RUNS_DIR/current/` holds the server logs, the pid list
  and `stack_info.txt` (binary md5 and settings).
- **Fresh data:** to download the articles and photos again instead of using
  the reference set, export `VL_DATA_DIR` to an empty directory before
  `prepare_data.sh`.
  - **Sites:** this needs lite.cnn.com, text.npr.org and
    commons.wikimedia.org, through `https_proxy` if set.
  - **Filtered hosts:** run `fetch_data.py --out DIR` on a machine that can
    reach those sites, then copy `DIR`.

## Workload

Each prompt is a single user message in this order:

1. Instructions.
2. News articles, with the five photos placed after articles 0, 16, 21, 23
   and 25.
3. An "Additional document N" made of paragraphs from the remaining
   articles.
4. A question about that document.

The builder adds whole paragraphs, then pads with `" ."`, until the prompt is
exactly 40,960 tokens as the instances count it. That count includes the chat
template and 405 vision tokens per image.

- **Shared region:** items 1 and 2 are identical in all 68 prompts and are at
  least 32,864 tokens long (33,710 with the reference articles).
- **Requests:** streaming `/v1/chat/completions`, temperature 0,
  `ignore_eos`, `max_tokens` 100, `enable_thinking` off.
- **Images:** referenced in the text as `<img>file:///abs/path.jpg</img>`.
  The prefill instances open the files themselves, so `DATA_DIR` must be on
  the serving host.

### Why the KV hit is exactly 80%

- **Chunk-aligned reuse:** the prefix cache can only restore the GDN
  (linear attention) state at chunk boundaries, so a request reuses whole
  8,192-token chunks.
- **The arithmetic:** a 32,864+-token shared region therefore gives
  4 x 8,192 = 32,768 cached tokens, 80.0% of 40,960.
- **Seeding every instance:** each prefill instance has its own cache. The
  eight sequential warm-up requests are routed round-robin, so every
  instance sees the shared region before measurement.
- **Measured, not assumed:** `bench.sh` reads `num_prefix_cache_tokens` and
  `prompt_tokens` of every measured request from the prefill logs and reports
  `kv_hit_rate`, which should be 0.80.

### Measurement protocol

1. **Master warm-up:** 2 bursts of 16 tiny requests. The master builds a
   tokenizer per worker thread on first use (about 330 ms each). Without
   this, some measured requests stall for 350-640 ms.
2. **Sequential warm-up:** 8 requests with distinct prompts. This seeds the
   prefix and encoder caches and captures the prefill graphs.
3. **Closed-loop warm-up:** 4 x C requests at concurrency C, cycling the
   warm-up prompts. Decode and verify graphs are captured lazily per batch
   size, so this keeps those captures out of the measurement.
4. **Measurement:** prompts 8-67, closed loop at concurrency C.

Metrics:

- **TTFT:** client-side time to the first streamed token, master included.
- **TPOT:** (last token time - first token time) / (completion tokens - 1).
- **QPS:** measured requests / wall time.
- **Steady TTFT:** TTFT mean without the first C requests.
- **`lazy_captures_during_measurement`:** graphs captured while measuring.
  Expect 0-2. A decode graph costs about 50 ms once; a prefill graph adds
  about 0.6 s to one request.
- **`server_errors`:** fatal lines in the server logs; must be 0.

## Files

| File | Purpose |
|---|---|
| `env.sh` | paths, host, GPU layout, `CONC`, `MTP_K`, ports |
| `build.sh` | builds xllm with `setup.py` |
| `prepare_data.sh` | runs `fetch_data.py`, `master_tokenizer.py` and the prompt build |
| `fetch_data.py` | downloads the articles in `article_urls.txt` and the photos in `images.json` when they are missing |
| `master_tokenizer.py` | writes the master tokenizer directory (the master cannot parse the model's chat template) |
| `start_stack.sh` / `stop_stack.sh` | start or stop etcd, master and the five instances |
| `bench.sh`, `vl_bench.py` | prompt builder and closed-loop client |
| `check.sh`, `vl_check.py` | VL correctness: the model must name the five images in order |
| `run_sweep.sh`, `report.py` | concurrency sweep on fresh stacks and its result table |
| `draft_vocab_65536.txt` | token ids of the MTP draft head's reduced vocabulary |

## Data sources

- **Articles:** `article_urls.txt` lists the articles fetched on 2026-10-22,
  each with its word count then. Their copyright stays with CNN and NPR.
  - **Drift:** news pages change. A re-fetch on 2026-10-23 found one article
    gone and six with 2-74 different words. `fetch_data.py` skips the gone
    ones, names the changed ones, and compares `articles.json` with the
    reference md5 `adc5d434686157b0c7b7b244e2d5862e`.
  - **Effect:** drifted data changes the prompt text, but every prompt is
    still exactly 40,960 tokens with the same 80% hit. A fresh fetch on
    2026-10-23 gave at C=5 a TTFT mean of 860 ms, a TPOT mean of 17.70 ms and
    an 80.0% KV hit.
- **Photos:** from Wikimedia Commons. See `images.json` for each file and
  author.
  - **Re-downloads:** Commons may serve a different rendition, so a
    re-download can differ slightly in pixels from the reference copies (mean
    absolute difference under 6/255).
  - **Token count:** fixed by the 854x480 size, so it does not change.
  - **Credits and licenses:**
    - US Capitol: Architect of the Capitol, public domain.
    - US Supreme Court: Kjetil Ree, CC BY-SA 3.0.
    - Moscow Kremlin: Anton Zelenov, CC BY 3.0.
    - Oil tanker Solomon Sea: ITookSomePhotos, CC BY-SA 4.0.
    - Ryanair Boeing 737: Hugo LUC, CC BY-SA 4.0.

## Troubleshooting

- **`port N is in use` or `GPU N is in use`:** another stack or job is
  running. `stop_stack.sh` stops only the processes listed in the latest
  run's `pids.txt`.
- **An instance fails to start:** read `$RUNS_DIR/current/<name>.log`.
  - **Configuration errors:** missing kernels under `MUSA_OPS_DIR` or a
    wrong `MUSA_PRELOAD` show up there first.
  - **`folly::FutureTimeout` right after engine init:** the MCCL
    process-group test of a CP pair did not finish within 4 s. Run
    `stop_stack.sh` and start again. `run_sweep.sh` retries a failed start
    once.
- **Chat requests fail with a template error:** check that
  `MASTER_TOKENIZER` points to the directory written by
  `master_tokenizer.py`.
