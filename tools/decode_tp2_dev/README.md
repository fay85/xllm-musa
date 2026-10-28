# Isolated Qwen3.8 TP2 decode

This harness belongs only to `decode_TP2_dev`. It creates one logical instance
with two rank processes on physical GPUs 6 and 7. The active
`xllm_musa_mt_main` stack is not managed by these commands.

The default is a standalone VLM/MTP K3 instance, full draft vocabulary,
fixed seed 42, and custom all-reduce enabled. Both target and draft bodies
use TP2; `enable_mtp_draft_body_tp1` remains false. The executable must reside
under `/workspace/decode_tp2_dev_artifacts/bin/` and is installed by the main
development/build workflow. No executable is copied from the live process.

```bash
cd /workspace/xllm-mt-main/tools/decode_tp2_dev
PYTHONDONTWRITEBYTECODE=1 python3 launch.py plan
PYTHONDONTWRITEBYTECODE=1 python3 launch.py preflight
PYTHONDONTWRITEBYTECODE=1 python3 launch.py start
PYTHONDONTWRITEBYTECODE=1 python3 launch.py status
PYTHONDONTWRITEBYTECODE=1 python3 launch.py stop --timeout 15
```

`plan` is CPU-only. `preflight` reads host listeners, GPU allocation/process
tables, required paths and executable identity; it performs no GPU computation.
`start` allocates GPUs and must only run when they are available. Rank0's
application startup marker is reported as process readiness; PD linking and
state correctness require separate end-to-end validation.

Admission rejects active listeners anywhere in the host network namespace,
including the full master/TCPStore window and both Mooncake RPC windows. It
also probes port binding with `SO_REUSEADDR`, so TIME_WAIT is distinct from an
active listener. GPU utilization must be zero and used memory at most 1 MiB;
any process above an enumeration context is rejected. GPU ownership is checked at every invocation; this launcher reserves only GPUs 6/7.

The installation pins the container's PID namespace and hostname in config.
If the container is recreated, inspect its identity and deliberately refresh
that pin in this private config. Never copy the pin to the main container.

Runtime evidence is retained under `/workspace/decode_tp2_dev_artifacts/runtime`.
It includes immutable per-run config, both rank logs, executable SHA256,
actual arguments/environment and PID/exe/start-tick/group/session identities.
`stop` verifies every recorded leader before signaling, sends SIGTERM only to
owned groups, waits a bounded interval, rechecks identity before SIGKILL, and
records actual zombie exit/wait status. A PID reuse or unknown surviving group
fails closed. It never searches by process name or signals another service.

For a same-binary MCCL comparison, use a separate config copy with
`custom_all_reduce=false`, stop the owned instance, and start that config.
Preserve model, K3, prompts, concurrency, cache state and all other settings.
The config controls `XLLM_MUSA_ENABLE_TP_CUSTOM_AR`; inspect rank logs to verify
which groups and payloads actually used custom AR. LM-head all-gathers still
use the normal collective path.

PD mode is opt-in: set `mode=pd` and an explicit `private_etcd` local endpoint.
The harness does not create a controller or prefill instance. Main etcd
64379/64380 is rejected. Use only an independently deployed private control
plane and a matching prefill configuration. Keep P/D MTP, full-vocabulary and
sampling policies aligned for correctness comparisons.

`draft_vocab` defaults to empty. Explicit subsets are an experimental variable:
current TP2 code truncates each rank's selected rows to a multiple of 128;
the main stack's 65536-ID file retains 65408 IDs after TP2 sharding, with
strong rank imbalance. Do not silently inherit it as a full-vocabulary baseline.

CPU validation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_launch
```

These tests cover topology, main-etcd rejection, derived-port conflicts,
Mooncake's forbidden ephemeral interval, custom-AR-only A/B selection, GPU
allocation checks, container mismatch and refusal to signal a reused PID.
They do not establish TP2 model correctness or performance.


Validated development candidate (2026-10-28):
`/workspace/decode_tp2_dev_artifacts/bin/xllm.native_tp2_pull_coalesced.r4`
(SHA256 `5817498a4d856329017aaaf713dd97d37db790b80f1e39cac538a9649f14f4ba`).
The config selects this binary. PD mode explicitly enables Native Service
registration; a missing loaded Engine pointer fails before legacy registration.

Generic TP2 reductions currently use two-stage custom AR. Full-model
one-stage runs produced malformed outputs even though synchronous and queued
producer probes passed; this path remains unqualified for the model.

The PD experiment uses Prefill's existing bounded release and this candidate
for Decode. Commands, GPQA results, per-request benchmark checks and source
snapshots are recorded in `/workspace/decode_tp2_dev_status_20261028.md` and
`/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/`.

`XLLM_MUSA_PULL_STAGING_MB` sets the staged PULL buffer (16..1024 MiB;
128 by default, 512 in the paired sweep). Reads coalesce registered-buffer
spans with at most 2x payload amplification; scatter retains exact destinations.
