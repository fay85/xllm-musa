#!/usr/bin/env python3
"""Same frozen 88-input workload for aggregate and disaggregated topology runs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request


def verify(run: Path, bundle: Path) -> dict:
    plan = json.loads((run / "effective_plan.json").read_text())
    records = json.loads((run / "identities.json").read_text())
    expected = {"etcd", "master", "coordinator"} | {r["name"] for r in plan["ranks"]}
    remote = {item for item in os.environ.get("TOPOLOGY_REMOTE_RANKS", "").split(",") if item}
    expected -= remote
    records = [rec for rec in records if rec["name"] not in remote]
    if {r["name"] for r in records} != expected or len(records) != len(expected):
        raise RuntimeError("Recorded roles do not match this topology")
    for rec in records:
        proc = Path("/proc") / str(rec["pid"])
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        wanted = bundle / "bin" / ("etcd" if rec["name"] == "etcd" else
                 "xllm_master_serving" if rec["name"] in {"master", "coordinator"} else "xllm")
        if (stat[0] == "Z" or stat[19] != rec["starttime"] or
                os.readlink(proc / "exe") != str(wanted.resolve()) or rec["exe"] != str(wanted.resolve())):
            raise RuntimeError("Owned identity changed: " + rec["name"])
    return plan


def snapshot(run: Path, dest: Path) -> None:
    plan = json.loads((run / "effective_plan.json").read_text())
    ports = sorted({r["ports"]["http"] for r in plan["ranks"] if r["node_rank"] == 0})
    fields = ("mm_input_loading", "mm_preprocess", "tokenization_latency",
              "chat_template_latency", "request_handling_latency",
              "prefix_cache_prompt_tokens_total", "prefix_cache_hit_tokens_total",
              "prefix_cache_token_hit_rate_perc")
    data = {}
    for port in ports:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/vars", headers={"Accept": "text/plain"})
            raw = urllib.request.urlopen(req, timeout=8).read().decode()
            data[str(port)] = "\n".join(line for line in raw.splitlines() if any(k in line for k in fields))
        except Exception as error:
            data[str(port)] = {"error": repr(error)}
    dest.write_text(json.dumps(data, indent=2))



def parse_bvar_prefix_cache(vars_path: Path, prefill_ports: list[int]) -> dict:
    if not vars_path.exists():
        return {}
    raw_map = json.loads(vars_path.read_text())
    prompt_sum = 0
    hit_sum = 0
    gauges = {}
    for p in prefill_ports:
        snippet = raw_map.get(str(p), "")
        if not isinstance(snippet, str):
            continue
        m_p = re.search(r'id="value-prefix_cache_prompt_tokens_total">([0-9.eE+-]+)<', snippet)
        m_h = re.search(r'id="value-prefix_cache_hit_tokens_total">([0-9.eE+-]+)<', snippet)
        m_g = re.search(r'id="value-prefix_cache_token_hit_rate_perc">([0-9.eE+-]+)<', snippet)
        if not m_p:
            m_p = re.search(r'prefix_cache_prompt_tokens_total\s*:\s*([0-9.]+)', snippet)
        if not m_h:
            m_h = re.search(r'prefix_cache_hit_tokens_total\s*:\s*([0-9.]+)', snippet)
        if not m_g:
            m_g = re.search(r'prefix_cache_token_hit_rate_perc\s*:\s*([0-9.]+)', snippet)
        if m_p:
            prompt_sum += int(float(m_p.group(1)))
        if m_h:
            hit_sum += int(float(m_h.group(1)))
        if m_g:
            gauges[str(p)] = float(m_g.group(1))
    return {"prompt_tokens_total": prompt_sum, "hit_tokens_total": hit_sum, "gauges_perc": gauges}


def qualify(run: Path, out: Path, prompt_tokens: int = 22752, cached_tokens: int = 20480) -> dict:
    plan = json.loads((run / "effective_plan.json").read_text())
    data = json.loads((out / "bench.json").read_text())
    ids = {x["request_id"] for x in data["requests"]}
    terminal = {role: {} for role in ["DEFAULT", "PREFILL", "DECODE"]}
    sync = {}
    for rank in plan["ranks"]:
        if rank["node_rank"] != 0:
            continue
        role = rank["role"]
        path = run / rank["logname"]
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if "request.cpp:" not in line:
                continue
            match = re.search(r"x-request-id: ([^,]+),", line)
            if not match:
                continue
            rid = match[1]
            values = {k: int(v) for k, v in re.findall(
                r"(prompt_tokens|num_prefix_cache_tokens|generated_tokens|speculative_accepted_tokens|speculative_proposed_tokens): (\d+)", line)}
            values.update(log=path.name, line=number, role=role)
            if rid in ids:
                if rid in terminal[role]:
                    raise RuntimeError("Duplicate role/RID terminal: " + role + " " + rid)
                terminal[role][rid] = values
            elif role in {"DEFAULT", "DECODE"} and re.search(r"-wb\d+-\d+$", rid):
                sync[rid] = values
    rows = []
    aggregate = plan["mode"] == "aggregate"
    for request in data["requests"]:
        rid = request["request_id"]
        source = terminal["DEFAULT" if aggregate else "PREFILL"].get(rid, {})
        final = source if aggregate else terminal["DECODE"].get(rid, {})
        valid = (source.get("prompt_tokens") == final.get("prompt_tokens") == prompt_tokens and
                 source.get("num_prefix_cache_tokens") == cached_tokens and
                 source.get("generated_tokens") == (100 if aggregate else 1) and
                 final.get("generated_tokens") == request.get("completion_tokens") == 100)
        rows.append(dict(request_id=rid, valid=valid, source=source, final=final))
    final_stats = terminal["DEFAULT" if aggregate else "DECODE"]
    proposed = sum(r.get("speculative_proposed_tokens", 0) for r in final_stats.values())
    accepted = sum(r.get("speculative_accepted_tokens", 0) for r in final_stats.values())
    summary = data["summary"]
    prefill_role = "DEFAULT" if aggregate else "PREFILL"
    prefill_ports = sorted({r["ports"]["http"] for r in plan["ranks"] if r["node_rank"] == 0 and r["role"] == prefill_role})
    bvar_before = parse_bvar_prefix_cache(out / "vars_before.json", prefill_ports)
    bvar_after = parse_bvar_prefix_cache(out / "vars_after.json", prefill_ports)
    bvar_window_prompt = bvar_after.get("prompt_tokens_total", 0) - bvar_before.get("prompt_tokens_total", 0)
    bvar_window_hit = bvar_after.get("hit_tokens_total", 0) - bvar_before.get("hit_tokens_total", 0)
    bvar_cum_prompt = bvar_after.get("prompt_tokens_total", 0)
    bvar_cum_hit = bvar_after.get("hit_tokens_total", 0)
    summary["bvar_cumulative_prompt_tokens"] = bvar_cum_prompt
    summary["bvar_cumulative_hit_tokens"] = bvar_cum_hit
    summary["bvar_cumulative_hit_rate_perc"] = round(bvar_cum_hit * 100.0 / bvar_cum_prompt, 4) if bvar_cum_prompt > 0 else None
    summary["bvar_port_gauges_perc"] = bvar_after.get("gauges_perc", {})
    summary["bvar_window_prompt_tokens"] = bvar_window_prompt
    summary["bvar_window_hit_tokens"] = bvar_window_hit
    summary["bvar_window_hit_rate"] = round(bvar_window_hit / bvar_window_prompt, 4) if bvar_window_prompt > 0 else None
    summary["kv_logged_requests"] = len(terminal[prefill_role])
    summary["kv_prompt_tokens"] = bvar_window_prompt if bvar_window_prompt > 0 else sum(r["source"].get("prompt_tokens", 0) for r in rows)
    summary["kv_cached_tokens"] = bvar_window_hit if bvar_window_prompt > 0 else sum(r["source"].get("num_prefix_cache_tokens", 0) for r in rows)
    summary["kv_hit_rate"] = summary["bvar_window_hit_rate"] if bvar_window_prompt > 0 else (round(summary["kv_cached_tokens"] / summary["kv_prompt_tokens"], 4) if summary["kv_prompt_tokens"] else None)
    summary["mtp_acceptance"] = round(accepted / proposed, 4) if proposed else None
    for label, values in [("ttft", [r["ttft_ms"] for r in data["requests"]]),
                          ("tpot", [(r["last_ms"] - r["ttft_ms"]) / 99 for r in data["requests"]])]:
        values.sort()
        for p in [50, 90, 95, 99]:
            summary[f"{label}_p{p}_ms"] = round(values[round(p / 100 * (len(values) - 1))], 3)
    sync_expected = 8 * summary["concurrency"]
    valid = (len(ids) == len(rows) == 60 and all(r["valid"] for r in rows) and
             summary.get("lazy_captures_during_measurement") == summary.get("server_errors") == 0 and
             len(sync) == sync_expected and all(r.get("generated_tokens") == 100 for r in sync.values()))
    result = dict(qualified=valid, scope="request length/prefix hit/output/capture/error checks; client timing is authoritative",
                  topology=plan["topology_name"], mode=plan["mode"], summary=summary,
                  formal=rows, sync_warm_expected=sync_expected, sync_warm_terminal_count=len(sync),
                  DP_replica_participation="not inferred from dp_size or terminal log; requires separate runtime evidence")
    (out / "bench.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    (out / "qualification.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument("--warm-closed", type=int, default=240)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--isl", type=int, default=22752)
    parser.add_argument("--shared", type=int, default=21904)
    parser.add_argument("--prefix-cached", type=int, default=20480)
    parser.add_argument("--warm", type=int, default=28)
    parser.add_argument("--requests", type=int, default=60)
    args = parser.parse_args()
    bundle = args.bundle_dir.resolve()
    run = args.run_dir.resolve()
    out = args.out_dir.resolve()
    plan = verify(run, bundle)
    out.mkdir(parents=True, exist_ok=False)
    with (run / "formal_workload_claim.json").open("x") as handle:
        json.dump({"out": str(out), "C": args.concurrency}, handle)
    sys.path.insert(0, str(bundle / "python"))
    spec = importlib.util.spec_from_file_location("topology_frozen_client", bundle / "python/vl_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    original_warm = bench._warm_up
    original_summary = bench._summarize
    original_burst = bench._master_burst

    def warm(a, prompts, tag):
        verify(run, bundle)
        original_warm(a, prompts, tag)
        for burst in range(8):
            start = time.monotonic()
            bench._closed_loop(a.concurrency, a.concurrency, lambda i: bench.stream_chat(
                a.url, a.model, prompts[(burst * 7 + i) % a.warm], a.osl,
                f"vl-{tag}-wb{burst}-{i}", a.timeout))
            bench.logger.info(f"full-output synchronized warm {burst} C{a.concurrency}: {time.monotonic()-start:.2f}s")
        snapshot(run, out / "vars_before.json")
        verify(run, bundle)

    def summarize(*a, **kw):
        snapshot(run, out / "vars_after.json")
        return original_summary(*a, **kw)

    def tiny(a, prefix):
        for attempt in range(30):
            try:
                return original_burst(a, prefix + str(attempt) + "-")
            except Exception as error:
                if attempt == 29:
                    raise
                bench.logger.warning(f"tiny warm-only retry: {error}")
                time.sleep(5)

    bench._warm_up = warm
    bench._summarize = summarize
    bench._master_burst = tiny
    bench._MASTER_BURST = args.concurrency
    url = f'http://127.0.0.1:{plan["port_bases"]["master_http"]}'
    prompts = (args.prompts or (bundle / "data/prompts_88.json")).resolve()
    argv = ["--url", url, "--model", "Qwen3.8-27B-FP8", "--tokenizer", args.model_dir,
            "--data-dir", str(bundle / "data"), "--prompts", str(prompts),
            "--isl", str(args.isl), "--shared", str(args.shared), "--osl", "100", "--concurrency", str(args.concurrency),
            "--requests", str(args.requests), "--warm", str(args.warm), "--warm-concurrency", "4", "--warm-master", "2",
            "--warm-closed", str(args.warm_closed), "--timeout", str(args.request_timeout),
            "--server-log-dir", str(run), "--out", str(out / "bench.json")]
    (out / "provenance.json").write_text(json.dumps(dict(
        plan=plan, client_argv=argv, input_sha256=hashlib.sha256(prompts.read_bytes()).hexdigest(),
        client_sha256=hashlib.sha256((bundle / "python/vl_bench.py").read_bytes()).hexdigest(),
        shim_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    sys.argv = [str(bundle / "python/vl_bench.py"), *argv]
    bench.main()
    hook = os.environ.get("TOPOLOGY_BEFORE_QUALIFY", "")
    if hook:
        subprocess.run(["bash", "-lc", hook], check=True)
    result = qualify(run, out, args.isl, args.prefix_cached)
    print("QUALIFICATION " + json.dumps({"qualified": result["qualified"], "summary": result["summary"]}), flush=True)
    return 0 if result["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
