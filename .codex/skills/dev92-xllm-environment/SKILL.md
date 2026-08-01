---
name: dev92-xllm-environment
description: Operate the xLLM project on dev92 safely and reproducibly over SSH, including repository inspection, remote editing, containerized MUSA builds, correctness and benchmark runs, artifact collection, and evidence-aware handoffs. Use whenever Codex works on /data/feihu/xllm-git-master, the corresponding /workspace/xllm-git-master checkout inside xllm-musa2.9.1-sdk5.1-dev, or the Qwen3.5 packed-prefill bucket fix and its BF16 or FP8 benchmarks.
---

# Operate xLLM on dev92

## Enforce the remote boundary

- Run every project file read, file write, repository command, build, test, benchmark, and log inspection on dev92 through SSH.
- Do not mount the remote tree locally. Do not create a local mirror as a substitute for the authoritative checkout.
- Use `/data/feihu/xllm-git-master` on the SSH host for repository documents, source edits, and Git inspection.
- Use `/workspace/xllm-git-master` inside `xllm-musa2.9.1-sdk5.1-dev` for builds and runtime commands.
- Treat `/data/feihu -> /workspace` as the current container bind mount, but verify it again if the container is recreated.
- Install a missing command on dev92 or in the container only when it is needed for the task. Record any environment-changing installation.

## Start every task from evidence

1. Connect with `ssh dev92` and change to `/data/feihu/xllm-git-master`.
2. Read the repository-root `AGENTS.md`.
3. Read `PREFILL_BUCKET_FIX_HANDOFF_20260731.md` and `bench_result_0731.md` when working on packed prefill, Qwen3.5, MUSA graphs, BF16, FP8, or the July 31 benchmark baseline.
4. Check the current branch, commit, focused Git status, container state, binary, model path, and relevant result artifacts. Do not assume the dated handoff is still current.
5. Before editing or reviewing anything under `xllm/`, read `.agents/skills/code-review/references/custom-code-style.md`.
6. For a code review, also read `.agents/skills/code-review/SKILL.md`.
7. Preserve the existing dirty worktree and unrelated MUSA refactor changes. Do not clean, revert, stage, commit, or rewrite unrelated files.

Use host-side inspection commands in this form:

~~~bash
ssh dev92 'cd /data/feihu/xllm-git-master && <command>'
~~~

Use container commands in this form:

~~~bash
ssh dev92 docker exec -w /workspace/xllm-git-master   xllm-musa2.9.1-sdk5.1-dev <command>
~~~

## Use the authoritative build and runtime

Build only with the documented MUSA command unless a newer handoff explicitly replaces it:

~~~bash
ssh dev92 docker exec -e MAX_JOBS=16 -e NINJA_TARGET=xllm   -w /workspace/xllm-git-master xllm-musa2.9.1-sdk5.1-dev   bash ./_build_cuda_graph_musa.sh
~~~

Use these authoritative runtime paths:

- Binary: `/workspace/xllm-git-master/build/lib.linux-x86_64-cpython-310/xllm/xllm`
- BF16 model: `/workspace/model_weights/Qwen3.5-27B`
- FP8 model: `/workspace/model_weights/Qwen3.5-27B-FP8`
- Benchmark artifacts: `/workspace/bench_results/`

Record the branch, commit, build log, binary timestamp, model and quantization, device, complete environment overrides, server log, client command, and result JSON for every reported run.

## Respect the 2026-07-31 evidence boundary

Treat the following as a dated baseline to verify, not as timeless truth.

### BF16 packed-piecewise baseline

- Regard the Qwen3.5-27B BF16 packed-piecewise bucket fix as validated for the recorded C=1, C=4, and C=5 checks.
- Preserve the stable token-bucket plus effective-batch cache key and live replay metadata updates.
- Preserve piecewise-owned stable GDN buffers and the required `gdn_output_` tail clearing.
- Keep `XLLM_MAX_PACKED_PREFILL_SEQS=2` as the validated scheduler cap. The recorded C=5 path reuses a `2+2+1` pattern.
- Keep Qwen3.5 packed buckets below 128 tokens on eager prefill. The unguarded 64-token B>1 packed-piecewise path failed semantic correctness.
- Do not reintroduce the direct in-graph variable-length Mate path; it produced incorrect B>1 output.
- Require correctness before performance. The recorded C=5 replay stress completed 50/50 requests without `MultinomialOut`, NaN, or FATAL lines.
- Interpret `concurrent_match_golden=False` as informational only when the semantic expected-substring and not-garbage checks pass.

The recorded 2k-input, 16-output BF16 measurements were:

| Concurrency | Mean TTFT | Mean TPOT | Throughput |
|---|---:|---:|---:|
| C=1 | 385.91 ms | 46.15 ms | 1627.40 TGS |
| C=4 | 1065.91 ms | 73.41 ms | 3462.66 TGS |
| C=5 | 1216.46 ms | 89.93 ms | 3694.95 TGS |

Do not compare these short-output BF16 TGS values directly with the FP8 2k-output benchmark.

### FP8 baseline and open failure

- Keep BF16 and FP8 conclusions separate. The BF16 packed-piecewise fix does not establish FP8 B>1 correctness.
- Treat FP8 B=2 packed-piecewise prefill as invalid: after capture at `actual_num_tokens=4132` and `bucket_num_tokens=4352`, logits became NaN and MUSA Multinomial failed. Do not report performance from that failed C=4 run.
- Use eager packed prefill with cap=2, producing `2+2`, as the current validated C=4 FP8 configuration. Keep decode graph enabled.
- The recorded FP8 ISL=2k/OSL=2k results were C=1 at 317.94 ms mean barrier-aligned TTFT and 68.55 tok/s total TGS, and C=4 at 807.88 ms and 239.00 tok/s.
- State the latency gate exactly: C=4 passed mean TTFT below 1 second, but p99/max were 1078.49/1078.50 ms and therefore did not pass an all-request or p99-below-1-second gate.

## Run changes as falsifiable experiments

1. Inspect the focused diff and identify the smallest relevant correctness test before building.
2. Build with the authoritative command and preserve the complete build log.
3. Start from the validated baseline configuration. Change one factor at a time.
4. Run semantic correctness before performance measurement. For the BF16 path, keep `MAXTOK=256` when the expected `391` token must be observable.
5. For graph changes, inspect capture counts and replay behavior, then run repeated waves long enough to expose stale-buffer or replay corruption.
6. Save each run under a new timestamped directory in `/workspace/bench_results/`. Never overwrite or reinterpret a failed run.
7. Separate warmup from measured requests and report exact success counts, errors, TTFT definition, TPOT, throughput formula, and concurrency release pattern.
8. Stop any server started for the task and verify GPU memory state when the run is complete.
9. Update the handoff only with observed evidence. Label hypotheses, planned work, invalid runs, and deliberately unrun checks explicitly.

If optimizing further, profile the validated 33/16 and 65/64 replay segments with `XLLM_PREFILL_BREAKDOWN=1` or Kineto, then A/B one change at a time. Do not remove GDN output-tail initialization without a long C=5 replay and semantic correctness check.
