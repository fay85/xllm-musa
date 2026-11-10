# Handoff 2026-11-10: Dual-Host S5000 1P1D (dev10 + dev15) & Internal DP/CP Optimization

## 1. Environment & Containers

| Role | Host Alias | IP / Hostname | Container Name | Image | Local Model Path |
|---|---|---|---|---|---|
| Control Plane + Prefill | `s5000_10` (`dev10`) | `10.121.32.10` (`worker32010`) | `xllm_1110_dev` | `registry.mthreads.com/presale/devtech/xllm:1110-dev` | `/data/models/Qwen3.8-27B-FP8` (verified 83 files, 28.77 GiB on local ext4) |
| Decode | `s5000_15` (`dev15`) | `10.121.32.15` (`worker32015`) | `xllm_1110_dev` | `registry.mthreads.com/presale/devtech/xllm:1110-dev` | `/data/models/Qwen3.8-27B-FP8` (verified 83 files, 28.77 GiB on local ext4) |

- Both containers run with `--runtime=mthreads --privileged --net=host --shm-size=67108864 -v /data:/data --entrypoint sleep infinity`.
- Passwordless SSH public-key authentication from the Windows driver machine to both `s5000_10` and `s5000_15` is configured and verified.
- Active engine binary in both containers:
  - Installed path: `/workspace/ttft_delivery_1110/release/xllm_ttft_fix_1110/bin/xllm`
  - Built output: `/workspace/xllm-mt-main/build/lib.linux-x86_64-cpython-310/xllm/xllm`
  - Backup of original 1110 binary: `/workspace/ttft_delivery_1110/release/xllm_ttft_fix_1110/bin/xllm.bak_1110` (`md5: ac2ff65a1e146022838775ac7bea368c`)
  - Current patched binary MD5: `3e464a5b77092efe063631d333ce6532` (also saved on shared NFS at `/data/nfs_shared/pd_1p1d/xllm.dp4cp2`).

---

## 2. Issues Found & Fixed

### 2.1 Ephemeral Port Range Collision (`start_topology.py`)
- **Symptom**: `Mooncake transfer engine failed to initialize (listen port 55010 may already be in use)` / `Fail to start Brpc rpc server on port 55010`.
- **Root Cause**: Linux `net.ipv4.ip_local_port_range` is `32768-60999`. The original `PORT_BASES` placed `p_tcpstore=54000`, `p_pd=52000`, `p_transfer=55000`, `d_tcpstore=56000`, `d_pd=52100`, `d_transfer=57000` inside the ephemeral port range, so outbound sockets during startup randomly grabbed `55010` before `brpc::Server::Start` bound it.
- **Fix**: Moved explicit service ports in `/workspace/topology_sweep_1110/launcher/start_topology.py` below `32768`:
  - `p_http: 18000, p_tcpstore: 18100, p_pd: 18300, p_transfer: 18400`
  - `d_http: 18600, d_tcpstore: 18700, d_pd: 19100, d_transfer: 19200`

### 2.2 Single-Instance Internal DP (`dp_size > 1`) Crashes & Deadlocks
When running a single engine instance with `dp_size > 1` (e.g. `dp_size=8` or `dp_size=4, cp_size=2`), idle DP shards execute `MTPWorkerImpl::step_empty()` (`q_max_seq_len == 0` / `num_sequences == 0`), which triggered three bugs in `1110-dev` (also responsible for the `agg_dp2` and `agg_tp2_dp2` failures in the image's prior `DONE.json`):
1. **`xllm/core/runtime/musa_graph_executor_impl.cpp`**:
   - `MusaGraphPersistentParam::update` (decode) passed `/*device=*/std::nullopt` and `update_prefill` omitted `device_` when calling `AttentionMetadataBuilder::build`, crashing on `Check failed: device.has_value() dummy attention requires device when new_cache_slots is undefined`.
   - Fixed by passing `device_` in both calls and bypassing graph capture/replay in `MusaGraphExecutorImpl::run()` when `params.meta.q_max_seq_len == 0 || params.meta.num_sequences == 0`.
2. **`xllm/core/layers/common/attention_metadata_builder.cpp` & `xllm/core/layers/musa/flashinfer_attention.cpp`**:
   - `populate_fa3_attention_metadata` ran on `is_dummy` metadata and promoted `kv_cu_seq_lens` to `int64` via `torch::cumsum`, and `FlashInferAttentionImpl::forward` did not early-return on `attn_metadata.is_dummy`, crashing at `attention_decode.cpp:612` (`cu_seqlens_k.scalar_type() == torch::kInt32`).
   - Fixed by early-returning on `attn_metadata.is_dummy` in both functions.
3. **`xllm/core/framework/parallel_state/musa/musa_custom_all_reduce.cpp`**:
   - `MusaCustomAllReduce::finalize_graph_buffers()` unconditionally executed `store_->wait(keys)` across `world_group` (all 8 ranks) even when `local_handles` and `local_offsets` were empty (because custom AR returns `std::nullopt` during MUSA stream capture). When only a subset of DP ranks captured a graph in a step, they deadlocked waiting for the other DP ranks.
   - Fixed by returning early in `finalize_graph_buffers()` when `local_handles.empty() && local_offsets.empty()`.

### 2.3 Single-Instance Internal DP Performance Bottlenecks (`step_empty` & Prefill Scheduler)
1. **Skip `step_empty` dummy model forward for non-EP models** (`worker_impl.cpp`, `mtp_worker_impl.cpp`, `llm_worker_impl.cpp`):
   - Previously, `worker_impl.cpp` padded any `dp_size > 1` empty shard with a fake token, and `MTPWorkerImpl::step_empty` ran a full 64-layer target + 3x draft Eager forward (~22 ms) on idle DP ranks. Because `LLMEngine::step` waits for all 8 ranks via `collectAll(futures)`, active decode ranks running 13 ms captured graphs were forced to wait 22 ms every step.
   - For models without cross-DP collectives (`ep_size <= 1 && n_routed_experts == 0 && mapping_data.empty()`), `need_fake_input_for_empty_shard` is now false and `MTPWorkerImpl::step_empty` / `LLMWorkerImpl::step_internal` immediately record the stream event and return an empty `ForwardOutput{}` (~0 ms).
2. **Per-DP-Rank Prefill Concurrency & Token Budget** (`continuous_scheduler.cpp`, `scheduler_policy.cpp`):
   - Previously, `limit_prefill_to_one_sequence` limited the entire 8-GPU instance to 1 prefill sequence per step, and `remaining_token_budget` was capped at `10240` across all DP ranks combined.
   - Fixed `scheduler_policy.cpp` to allow 1 prefill sequence **per DP rank** (`count_prefill_sequences < dp_size`, up to `dp_size` concurrent prefills per step) with round-robin tie-breaking across available DP ranks (`total_blocks - used_blocks`), and scaled `budget.remaining_token_budget = max_tokens_per_batch * dp_size` in `continuous_scheduler.cpp` while keeping `dp_group_token_caps[dp_rank] = max_tokens_per_batch` (`10240`).

### 2.4 Enabling Single-Instance `DP4CP2` Prefill (`world_size=8, dp_size=4, cp_size=2, tp_size=1`)
1. **`xllm/core/distributed_runtime/master.cpp` (`validate_model_cp`)**:
   - Relaxed `options.cp_size() != 2 || options.dp_size() != 1 || global_world_size != options.cp_size()` to `options.cp_size() != 2 || options.dp_size() < 1 || global_world_size != options.dp_size() * options.cp_size()`.
2. **`xllm/core/distributed_runtime/llm_engine.cpp` (`LLMEngine::get_cache_info`)**:
   - Previously published workers `[worker_clients_num_ - worker_clients_num_ / cp_size_, worker_clients_num_)` (`ranks 4, 5, 6, 7`), which is only valid when `dp_size == 1`. Under `dp_size=4, cp_size=2`, the 4 DP groups are `[0,1], [2,3], [4,5], [6,7]` and the complete KV + GDN state resides on the last CP rank of each DP group (`ranks 1, 3, 5, 7`).
   - Fixed `LLMEngine::get_cache_info` to iterate `dp_rank = 0..dp_size_-1` and publish `dp_rank * dp_local_size_ + (cp_size_ - 1) * dp_local_tp_size_ + tp_rank`.

---

## 3. Benchmark Results Summary (Frozen 20k + 5x480p Workload)

- **Workload**: 88 frozen prompts (`ISL = 22,752`, shared prefix = `20,480` tokens -> **90.01% KV cache hit rate**, `OSL = 100`, 5 images of 854x480 per prompt, 60 formal requests per concurrency after warmup, fresh stack per concurrency, `MTP K=3`).

### 3.1 Complete Comparison Table (C = 4, 8, 12, 16)

| Topology | Concurrency C | Qualified | Throughput (req/s) | Wall (s) | TTFT p50 (ms) | TTFT mean (ms) | TTFT p90 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT mean (ms) | TPOT p90 (ms) | KV Hit Rate | MTP Acceptance | Captures / Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1. Single-Host dev10: 2xCP2 Prefill + 4xTP1 Decode (6 instances, 8 GPUs total)** | 4 | True | 2.364 | 25.4 | 320.9 | 333.5 | 349.6 | 585.2 | 13.14 | 13.32 | 15.17 | 90.01% | 57.93% | 0 / 0 |
| | 8 | True | 3.992 | 15.0 | 337.0 | 426.4 | 636.2 | 863.4 | 14.59 | 14.70 | 16.65 | 90.01% | 57.85% | 0 / 0 |
| | 12 | True | 5.463 | 11.0 | 364.4 | 508.0 | 826.5 | 1143.9 | 15.20 | 15.42 | 17.65 | 90.01% | 59.08% | 0 / 0 |
| | 16 | True | 6.189 | 9.7 | 467.5 | 656.3 | 1370.5 | 2062.4 | 16.66 | 16.80 | 19.20 | 90.01% | 57.13% | 0 / 0 |
| **2. Dual-Host 1P1D: TP4 Prefill (dev10 4 GPUs) + TP4 Decode (dev15 4 GPUs)** | 4 | True* | 1.825 | 32.9 | 317.1 | 362.7 | 483.4 | 817.9 | 17.98 | 17.95 | 19.52 | 90.01% | — | 0 / 0 |
| | 8 | True | 3.070 | 19.5 | 407.8 | 513.9 | 679.5 | 1774.8 | 19.28 | 19.67 | 22.36 | 90.01% | 57.49% | 0 / 0 |
| | 12 | False* | 3.636 | 16.5 | 947.1 | 1061.8 | 1570.8 | 2753.5 | 19.79 | 19.69 | 22.32 | 90.01% | — | 0 / 0 |
| | 16 | False* | 3.682 | 16.3 | 1881.4 | 1964.8 | 2497.5 | 3698.1 | 19.97 | 19.90 | 22.00 | 90.01% | — | 0 / 0 |
| **3. Dual-Host 1P1D: DP8 Prefill (dev10, 1 instance) + DP8 Decode (dev15, 1 instance) [Before Opt]** | 4 | True | 1.335 | 45.0 | 444.4 | 539.9 | 715.9 | 1163.1 | 23.28 | 24.01 | 28.98 | 90.01% | 53.39% | 0 / 0 |
| | 8 | True | 2.446 | 24.5 | 787.3 | 864.3 | 1164.4 | 2583.8 | 21.68 | 22.13 | 26.25 | 90.01% | 51.47% | 0 / 0 |
| | 12 | True | 2.559 | 23.4 | 2076.1 | 2123.4 | 2448.8 | 4046.0 | 20.95 | 22.34 | 28.23 | 90.01% | 52.48% | 0 / 0 |
| | 16 | True | 2.481 | 24.2 | 3499.2 | 3398.4 | 3996.2 | 5483.3 | 21.73 | 22.42 | 27.90 | 90.01% | 53.81% | 0 / 0 |
| **4. Dual-Host 1P1D: DP8 Prefill (dev10, 1 instance) + DP8 Decode (dev15, 1 instance) [After Opt]** | 4 | True | **2.045** | 29.3 | **452.9** | 502.0 | 665.6 | 820.4 | **14.19** | 14.24 | 16.12 | 90.01% | 58.70% | 0 / 0 |
| | 8 | True | **3.526** | 17.0 | **607.1** | 610.8 | 817.7 | 820.7 | **15.25** | 15.11 | 17.04 | 90.01% | 58.36% | 0 / 0 |
| | 12 | True | **4.732** | 12.7 | **650.4** | 666.3 | 816.2 | 1166.0 | **17.57** | 17.71 | 20.44 | 90.01% | 58.81% | 0 / 0 |
| | 16 | True | **5.558** | 10.8 | **693.0** | 746.7 | 1202.0 | 1204.2 | **19.10** | 19.22 | 22.14 | 90.01% | 58.54% | 0 / 0 |
| **5. Dual-Host 1P1D: DP4CP2 Prefill (dev10, 1 instance) + DP8 Decode (dev15, 1 instance) [Latest]** | **4** | **True** | **2.210** | **27.1** | **325.9** | **372.3** | **515.9** | **570.2** | **13.97** | **14.13** | **15.65** | **90.01%** | **59.33%** | **0 / 0** |
| | **8** | **True** | **3.997** | **15.0** | **428.2** | **452.2** | **580.2** | **818.9** | **14.48** | **14.58** | **16.33** | **90.01%** | **60.08%** | **0 / 0** |
| | **12** | **True** | **5.086** | **11.8** | **509.5** | **525.0** | **812.8** | **1049.6** | **17.02** | **17.22** | **19.00** | **90.01%** | **58.80%** | **0 / 0** |
| | **16** | **True** | **5.981** | **10.0** | **594.1** | **646.4** | **1052.4** | **1291.2** | **17.84** | **18.06** | **20.25** | **90.01%** | **59.62%** | **0 / 0** |
| **6. Single-Host dev10 1P1D: DP2CP2 Prefill (1 instance, GPUs 0-3) + DP4 Decode (1 instance, GPUs 4-7)** | **4** | **True** | **2.176** | **27.6** | **349.7** | **388.9** | **499.4** | **569.8** | **13.97** | **14.05** | **15.46** | **90.01%** | **57.80%** | **0 / 0** |
| | **8** | **True** | **3.603** | **16.7** | **421.7** | **476.6** | **615.5** | **1066.7** | **15.94** | **16.03** | **18.03** | **90.01%** | **58.94%** | **0 / 0** |
| | **12** | **True** | **4.698** | **12.8** | **533.7** | **617.5** | **1081.3** | **1553.3** | **18.05** | **17.79** | **19.56** | **90.01%** | **58.67%** | **0 / 0** |
| | **16** | **True** | **5.344** | **11.2** | **548.2** | **719.7** | **1582.1** | **2061.7** | **19.07** | **19.24** | **22.35** | **90.01%** | **58.68%** | **0 / 0** |

---

## 4. Key Configuration & Result Paths (in `xllm_1110_dev` on `dev10`)

- **Topology Plans**:
  - 1P1D (`DP4CP2` Prefill + `DP8` Decode): `/workspace/topology_sweep_1110/launcher/plans/pd_1p1d_dp4cp2_dp8.json`
  - 1P1D (`DP8` Prefill + `DP8` Decode): `/workspace/topology_sweep_1110/launcher/plans/pd_1p1d_dp8.json`
  - 1P1D (`TP4` Prefill + `TP4` Decode): `/workspace/topology_sweep_1110/launcher/plans/pd_tp4_split.json`
- **Result Directories**:
  - `DP4CP2 + DP8`: `/workspace/regression/results/text20k_1p1d_dp4cp2_c{4,8,12,16}`
  - `DP8 + DP8 (Optimized)`: `/workspace/regression/results/text20k_1p1d_dp8_opt_c{4,8,12,16}`
  - `DP8 + DP8 (Baseline)`: `/workspace/regression/results/text20k_1p1d_dp8_c{4,8,12,16}`
  - Single-host `2xCP2 + 4xTP1`: `/workspace/regression/results/text20k_img5_hit90_20261110_151859/c{4,8,12,16}`
- **Modified C++ Source Files in `/workspace/xllm-mt-main`**:
  - `xllm/core/distributed_runtime/master.cpp`
  - `xllm/core/distributed_runtime/llm_engine.cpp`
  - `xllm/core/runtime/worker_impl.cpp`
  - `xllm/core/runtime/mtp_worker_impl.cpp`
  - `xllm/core/runtime/llm_worker_impl.cpp`
  - `xllm/core/runtime/musa_graph_executor_impl.cpp`
  - `xllm/core/layers/common/attention_metadata_builder.cpp`
  - `xllm/core/layers/musa/flashinfer_attention.cpp`
  - `xllm/core/framework/parallel_state/musa/musa_custom_all_reduce.cpp`
  - `xllm/core/scheduler/continuous_scheduler.cpp`
  - `xllm/core/scheduler/scheduler_policy.cpp`

---

## 5. Next Steps for the Following Agent

1. **Clean up / Stop Running C=16 Stack When Idle**:
   - The `20261110_1p1d_dp4cp2_c16_prefill` (on `dev10`) and `20261110_1p1d_dp4cp2_c16_decode` (on `dev15`) stacks from the last C=16 run are currently still up (or can be stopped via `stop_topology.sh --run <run_dir>` before starting new experiments).
2. **Further 1P1D Single-Instance Decode TPOT Optimization (at C=12 / C=16)**:
   - At C=4 and C=8, `1P1D (DP4CP2 + DP8)` achieves `TTFT p50 = 325.9 ms / 428.2 ms`, `TPOT p50 = 13.97 ms / 14.48 ms`, and `3.997 req/s` (matching or beating the 6-instance single-host setup).
   - At C=12 and C=16, `TPOT p50` rises to `17.02 ms / 17.84 ms` because all 8 DP ranks in the single Decode instance step in lock-step (`LLMEngine::step` waits on `collectAll(futures)` across all 8 workers), so any single DP rank doing a 2-request batch (`8 tokens` in spec-verify) or KV pull delays the other 7 DP ranks that only have 1 request (`4 tokens`).
   - **Actionable optimizations if higher C=16 throughput is needed**:
     - Implement DP load-balancing on Decode admission (`BlockManagerPool::get_manager_with_max_available_blocks` currently ranks by total available KV blocks rather than active running decode sequence count per DP rank; balancing by active decode sequences per DP rank will keep all 8 DP ranks at uniform batch size `B=1` up to `C=8` and `B=2` at `C=16`, avoiding `B=3` stragglers).
     - Or decouple per-DP-rank worker step loops inside `LLMEngine` when `ep_size == 1`.
3. **Commit / Solidify Changes into the Delivery Bundle & Image**:
   - Update `/workspace/ttft_delivery_1110/release/xllm_ttft_fix_1110/configs/candidate.json` (and `/workspace/config.json`) if this binary (`md5: 3e464a5b77092efe063631d333ce6532`) and port range (`< 32768`) are to be promoted as the new default delivery artifact or committed to a new Docker image tag.
