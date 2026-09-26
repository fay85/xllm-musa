#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""One full original-protocol evaluation followed by same-service hot verification."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--unit-log", type=Path, required=True)
    parser.add_argument("--reference-socket", type=Path, required=True)
    parser.add_argument("--draft-tokens", type=int, default=6)
    parser.add_argument("--unit-tests", type=int, default=23)
    parser.add_argument("--min-accuracy", type=float, default=96.0)
    parser.add_argument("--max-tpot", type=float, default=9.0)
    parser.add_argument("--max-ttft", type=float, default=240.0)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    candidate = args.candidate_dir.resolve()
    if not run.is_relative_to(ROOT / "logs") or not candidate.is_relative_to(ROOT / "build"):
        parser.error("run/candidate must be inside this checkout")
    for name in ["responses.raw.jsonl", "xllm.summary.json", "eval_client.log", "acceptance.json"]:
        if (run / name).exists():
            parser.error(f"refusing to repeat or overwrite evaluation: {name}")
    launch = dict(line.split("=", 1) for line in
                  (run / "launch_manifest.txt").read_text().splitlines() if "=" in line)
    if Path(launch["xllm_bin"]).resolve() != candidate / "xllm":
        parser.error("candidate does not match the launched binary")
    if int(launch["num_speculative_tokens"]) != args.draft_tokens:
        parser.error("expected draft width does not match the service")
    perf = json.loads((run / "formal/summary.json").read_text())
    if (perf["strict_pass_count"] != 6 or perf["tpot_avg_ms"] >= args.max_tpot
            or perf["ttft_avg_ms"] >= args.max_ttft):
        parser.error("candidate has not passed the initial full performance gate")
    base_url = f"http://127.0.0.1:{launch['http_port']}"
    api_url = base_url + "/v1/chat/completions"
    dataset_root = Path("/data/nfs_shared/jiaming.wang/sglang_gsm8k_20260920.RxaK30")
    reference = ROOT / "logs/goal_gsm8k_full_sgl_protocol_20260925"
    reference_counts = ROOT / "logs/goal_gsm95_tpot10_short_c4_20260926/reference_prompt_tokens.json"
    if not reference_counts.is_file():
        parser.error("verified reference prompt counts are missing")
    ssh = ["ssh", "-p", "22", "-S", str(args.reference_socket.resolve()),
           "-o", "BatchMode=yes", "mccxadmin@10.20.34.28"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def execute(command: list[str], output=None) -> None:
        logger.info("Validation stage: %s", shlex.join(command))
        subprocess.run(command, cwd=ROOT, env=env, check=True,
                       stdout=output, stderr=subprocess.STDOUT)

    # Fail before model requests if reference scoring connectivity is absent.
    execute(ssh + ["docker exec -e PYTHONDONTWRITEBYTECODE=1 wjm_sg_dspark python3 -c "
                   "'import sgl_eval._vendored.nemo_skills.evaluator.math'"])
    with (run / "eval_client.log").open("x") as log:
        execute(["python3", "tools/eval_gsm8k_sgl_protocol.py", "--run-dir", str(run),
                 "--dataset", str(dataset_root / "dataset_test.jsonl"),
                 "--prompt-yaml", str(dataset_root / "eval_prompt.yaml"),
                 "--url", api_url, "--model", "DeepSeek-V4-Flash-FP8", "--thinking-mode", "chat"], log)
    remote = ["docker", "exec", "-e", "PYTHONDONTWRITEBYTECODE=1", "wjm_sg_dspark", "python3",
              str(ROOT / "tools/score_gsm8k_sgl_reference.py"), "--dataset",
              str(dataset_root / "dataset_test.jsonl"), "--input", str(run / "responses.raw.jsonl"),
              "--output-dir", str(run), "--label", "xllm"]
    execute(ssh + [shlex.join(remote)])
    shutil.copyfile(reference_counts, run / "reference_prompt_tokens.json")
    execute(["python3", "tools/compare_gsm8k_sgl_full.py", "--run-dir", str(run),
             "--xllm-scored", str(run / "xllm.scored.jsonl"),
             "--reference-scored", str(reference / "sglang_reference.scored.jsonl")])
    for label, first, last in [("perf_after_gsm_warmup", 0, 1), ("perf_after_gsm_formal", 2, 7)]:
        execute(["python3", "tools/replay_dspark_perf.py", "--url", api_url,
                 "--input-dir", "logs/goal_repro_q6_legacy_perf_20260923",
                 "--output-dir", str(run / label), "--first", str(first), "--last", str(last)])
    execute(["python3", "tools/audit_gsm8k_full_goal.py", "--run-dir", str(run),
             "--candidate-dir", str(candidate), "--unit-log", str(args.unit_log.resolve()),
             "--baseline-perf", "logs/goal_gsm95_tpot10_short_c4_20260926/perf_after_gsm_formal/summary.json",
             "--reference-summary", str(reference / "sglang_reference.summary.json"),
             "--min-accuracy-percent", str(args.min_accuracy), "--max-tpot-ms", str(args.max_tpot),
             "--max-ttft-ms", str(args.max_ttft), "--expected-draft-tokens", str(args.draft_tokens),
             "--expected-unit-tests", str(args.unit_tests), "--permit-scored-truncation"])


if __name__ == "__main__":
    main()
