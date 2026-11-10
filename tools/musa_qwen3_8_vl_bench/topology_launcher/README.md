# Isolated topology launcher

This harness reads binaries/libraries/assets/data from the delivered package and
writes only under the experiment work root. It does not run a benchmark, globally
drain GPUs, search processes by name, modify the package/config, or promote a
candidate. Actual launch/stop use is owned by root.

```bash
# Static JSON + gate + internal-port validation only; no bundle/GPU/process work.
bash launcher/start_topology.sh --plan launcher/plans/agg_tp2_dp2.json --validate-plan

# Effective argv preview only; source the immutable bundle env, no start/writes.
bash launcher/start_topology.sh --bundle-dir /workspace/ttft_delivery_1110/release/xllm_ttft_fix_1110 \
  --plan launcher/plans/agg_tp2_dp2.json --plan-only

# Root's actual startup, into an explicitly unique absent/empty run directory.
MODEL_DIR=/data/nfs_shared/models/Qwen3.8-27B-FP8 HOST_ADDR=10.121.38.92 \
  bash launcher/start_topology.sh \
  --bundle-dir /workspace/ttft_delivery_1110/release/xllm_ttft_fix_1110 \
  --work-root /workspace/topology_sweep_1110 \
  --run-dir /workspace/topology_sweep_1110/runs/agg_tp2_dp2_OWN_UNIQUE_ID \
  --plan launcher/plans/agg_tp2_dp2.json

# Stop only the exact recorded PID + executable + starttime identities.
bash launcher/stop_topology.sh --run /workspace/topology_sweep_1110/runs/agg_tp2_dp2_OWN_UNIQUE_ID
```

`--check-only` verifies runtime/input identities (including 88 CPU token counts),
source cache and listening-port conflicts without starting processes or creating
experiment directories. Missing canonical image aliases must be prepared by the
bundle's data-preparation step beforehand. `--skip-cpu-token-counts` is an explicit
diagnostic override, never used by default.

Do not redirect stdout into a file inside a proposed `--run-dir` before startup:
the directory must be absent or empty, and is never reused. Put controller logs
outside it. `plan.run_dir` can provide the same explicit path; CLI wins. Automatic
naming is available if omitted, but root's benchmark should use the explicit run
path instead of relying on a `current` link.

## Plan schema and logs

```json
{
  "topology_name": "agg_tp2_dp2",
  "mode": "aggregate",
  "concurrency": 16,
  "benchmark_concurrency": 8,
  "groups": [
    {
      "name": "aggregate0",
      "role": "DEFAULT",
      "visible_gpus": [0, 1, 2, 3],
      "world_size": 4,
      "dp_size": 2,
      "cp_size": 1
    }
  ]
}
```

`concurrency` is the engine max-concurrent-request setting; measurement C8 is
separate. `world_size` equals `nnodes` and the length of the physical GPU list.
Derived TP is `world_size/(dp_size*cp_size)`; no invented TP flag is passed. Every
rank sees the **same complete group GPU list**, with `node_rank=0..world-1`.
Groups cannot overlap GPUs. DEFAULT groups cannot mix with PD groups.

Log/PID identity names are `${group.name}_r${rank}`. Examples:
`aggregate0_r0.log`, `aggregate0_r1.log`, `prefill0_r0.log`, `decode3_r0.log`.
The engine role is recorded independently of the filename. No logical-DP replica
warming or per-replica completion coverage is claimed solely from this naming.

Runtime artifacts:

- `plan.json`: original input; `effective_plan.json`: normalized groups/ranks,
  roles, physical GPU IDs, world/DP/CP/derived-TP, lognames and port reservations.
- `effective_commands.json`: per-process argv/env overrides (lab token redacted).
- `identities.json`, `pids.txt`: unchanged copied portable spawn helper format.
- `bin_hashes.json`: actual engine/master/etcd/preloader/tokenizer-lib/draft hashes.
- `inputs_preflight.json`, `cache_preparation.json`, `effective_environment.json`.
- `READY.json`: startup/health only, explicitly not inference qualification.
- On failure: `STARTUP_FAILURE.json`, `STARTUP_CLEANUP.json`; only owned identities
  are terminated, with guarded force cleanup if needed. Original logs remain.

The default engine MD5 is pinned to `ac2ff65a1e146022838775ac7bea368c`. An intentional
new binary requires root to select `--expected-engine-md5`; identity changes are
not silently accepted.

## Supported plans and gates

The plans folder contains single-instance TP2, TP4, TP2×DP2 (world4/DP2), DP2
(world2/DP2), PD4P4D and PD6P2D. All use K3/chunk10240/Pbatch10240/Pgraph4096,
P/default mem0.9/slots32, D mem0.85/slots12, maxseq16, P piecewise=true.
DEFAULT has disaggregation=false and service-routing=true.

Aggregated VLM CP2 and CP×TP are hard-gated and cannot be launched, even with a
plan-only pass. The reserved two CP2-prefill groups + four TP1 decodes plan is
disabled unless root explicitly selects `--allow-pd-cp2`; only P world2/CP2/TP1
is admitted by that opt-in. This is a launch capability, not model qualification.
PD internal DP>1 is deliberately not enabled here.

## Port map

All values may be overridden by plan `port_bases`, with strict internal collision
and 1..65535 checks. Listening TCP ports are checked immediately before start.

| Family | HTTP | TCPStore reservation | PD | transfer per rank | Mooncake RPC range |
|---|---|---|---|---|---|
| P / DEFAULT i | 62300+i | 54000+100i .. +32 | 52000+i | 55000+100i+10r | 33000+1000i+20r .. +16 |
| D j | 62700+j | 56000+100j .. +32 | 52100+j | 57000+100j+10r | 44000+1000j+20r .. +16 |

Router64888/64889, coordinator64890/64891, etcd64379/64380 remain explicit.
HTTP/PD/TCPStore group ports are shared only logically by that group's ranks;
rank transfer and Mooncake reservations are distinct. Every declared reservation
is checked; the legacy launcher’s mismatched aggregate precheck is not reused.

## Bundle/cache ownership

The launcher sources the bundle's env/load_config only to resolve the sealed
runtime assets and original settings, then builds topology argv independently;
it does not edit or invoke the bundle start_stack.sh/verify_owned/bench driver.
All mutable cache/TMP/XDG/HF/torch-extension/run/results paths are redirected to
`/workspace/topology_sweep_1110`. Source precompiled cache is copied into an
owned experiment cache on first use, with resolved-source + engine-MD5 marker;
a matching owned cache is reused, while an unmarked/conflicting destination is
left untouched. Source bundle directories are never written.

Local validation performed: bash -n, py_compile, static validation of all six
requested plan families, CP gate rejection, opt-in reserved P-CP2 validation and
unique/in-range reserved ports. No local or remote GPU operation or service
startup has been executed by the implementing agent.
