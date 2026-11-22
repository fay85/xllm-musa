"""Typed 20k benchmark protocols and a versioned, raw-evidence three-reference gate.

No process launches, state changes, or network access. A revision is an immutable
JSON object identified by its canonical SHA; its filename alone is not a binding.
"""
import math
import re
import statistics
from datetime import datetime
from long_20k_200k_gate import canonical_sha as digest, reference_metrics, _judge_two_registered_groups

REVISION = "20261120_TTFT_rootcause_three_references_then_MATE"
MODELS = {"A": "GLM-5.3-Flash", "V": "DeepSeek-V4-Flash", "S": "DeepSeek-V4.1-Flash-1017-W4AFP8-mt"}
V41_IDS = "6ee1cb788cedad06ba68d69bcbfe163bd649aaa1dff8e555911ee86bf30bdf27"
V41_DATA = "246d64155aa0f7fb4410f7d25d732c3f01e15de3d1ccb3edb36ad138385e625e"
MAIN = "pd20k-evalscope-content-v1"
RESOURCE_2K = "xllm-2k-resource-control"
PROTOCOLS = {
    MAIN: (20000, 1024, 1, "1P1D", "client-first-content-last-json"),
    RESOURCE_2K: (2000, 1024, 1, "1P1D", "client-first-content-last-json"),
    "sg-guide-prefill": (32000, 1, 1, "guide-prefill", "sglang-oai"),
    "sg-guide-pd-capacity": (32000, 1000, 1, "1P1D", "sglang-oai"),
    "sg-guide-pd-c48": (32000, 1000, 48, "1P1D", "sglang-oai"),
    "albert-original": (20000, 1, 1, "original-deployment", "vl_bench"),
}
KEYS = ("ttft_ms", "tpot_ms", "e2e_s")


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def guard(state, protocol, purpose, shape, source):
    need(state.get("resume_revision") == REVISION and state.get("execution_authorized") is True
         and "PAUS" not in state["status"] and state["allowed_lengths"] == [20000], "not resumed for this goal")
    need(protocol in PROTOCOLS and tuple(shape) == PROTOCOLS[protocol][:3], "protocol/shape mismatch")
    if protocol != MAIN:
        need(purpose == "auxiliary" and source == {RESOURCE_2K: "X", "albert-original": "A"}.get(protocol, "S"),
             "auxiliary can never emit a main reference or candidate")
    else:
        need((purpose == "reference" and source in MODELS) or
             (purpose == "diagnostic" and source in (*MODELS, "X")) or
             (purpose in ("screen", "formal") and source == "X"), "wrong producer or purpose")


def protocol_hash(protocol):
    return digest([protocol, PROTOCOLS[protocol]])


def rows_mean(rows):
    need(bool(rows), "missing samples")
    for r in rows:
        need(all(math.isfinite(r[k]) and r[k] > 0 for k in KEYS), "invalid timing")
        need(math.isclose((1000*r["e2e_s"]-r["ttft_ms"])/1023, r["tpot_ms"],
                          abs_tol=1e-10, rel_tol=1e-12), "timing boundary/formula")
    return {k: statistics.mean(r[k] for r in rows) for k in KEYS}


def validate_ref(role, ref):
    need(role in MODELS and ref["role"] == role and ref["model"] == MODELS[role], "wrong reference model")
    need(ref["framework"] == ("sglang" if role == "S" else "xllm"), "wrong framework")
    need(ref["protocol_sha256"] == protocol_hash(MAIN), "not the common PD protocol")
    need(ref["version"] and ref["version"] != "r0", "historical control is not a repaired reference")
    for key in ("dataset_sha256", "token_ids_sha256", "tokenizer_sha256",
                "semantic_recipe_sha256", "environment_sha256"):
        need(sha(ref[key]), "unbound reference "+key)
    if role == "S":
        need(ref["dataset_sha256"] == V41_DATA and ref["token_ids_sha256"] == V41_IDS, "wrong V4.1 input")
    runs = ref["runs"]
    need(len(runs) == 2 and len({r["uid"] for r in runs}) == 2, "need two reference restarts")
    cfg, assets = runs[0]["config_sha256"], runs[0]["assets_sha256"]
    need(sha(cfg) and sha(assets), "missing config/asset provenance")
    starts = {}; ids = set(); scored = []
    for run in runs:
        need(run["config_sha256"] == cfg and run["assets_sha256"] == assets, "restart drift")
        need(run["released"] == {"sj9": True, "sj10": True} and run["compile_clear"] is True
             and run["profiler_on"] is False, "unreleased, contaminated or profiled reference")
        for role_name, host in (("P","sj9"),("D","sj10"),("router","sj10")):
            st = run["starts"][role_name]
            # Router may be co-located on P for SGLang.
            need((st["host"] in ("sj9","sj10") if role_name=="router" else st["host"]==host)
                 and sha(st["container"]) and int(st["pid"])>0 and int(st["start_ticks"])>0, "bad start")
            key=(st["host"],st["container"],int(st["pid"]),int(st["start_ticks"]))
            need(key not in starts.setdefault(role_name,set()), "reused process")
            starts[role_name].add(key)
        files=run["evidence_sha256"]
        need(all(sha(files.get(k)) for k in ("warm_db","formal_db","terminal","launch_p","launch_d",
                  "launch_router","release_p","release_d","loaded","compile_window","identity")), "missing raw provenance")
        rows=run["samples"]
        need([r["phase"] for r in rows] == ["warmup"]+["formal"]*5, "need 1+5")
        rows_mean(rows)
        for r in rows:
            need(r["id"] and r["id"]==r["terminal_id"] and r["id"] not in ids and
                 r["unique_terminal_count"]==1 and r["success"] is True and
                 (r["prompt_tokens"],r["completion_tokens"],r["cached_tokens"],r["reasoning_tokens"],
                  r["finish_reason"])==(20000,1024,0,0,"length"), "unverified terminal")
            ids.add(r["id"])
            need(isinstance(r["accepted"],int) and isinstance(r["proposed"],int)
                 and 0<=r["accepted"]<=r["proposed"] and sha(r["text_sha256"]), "missing output/spec audit")
        scored.extend(rows[1:])
    return rows_mean(scored)


def summary(bundle):
    need(bundle["schema"] == "ttft20-reference-set-v1", "wrong reference schema")
    need(bundle["protocol_sha256"] == protocol_hash(MAIN), "protocol binding changed")
    historical=bundle["r0"]
    need(historical["length"]==20000 and historical["dataset_sha256"]==V41_DATA,"wrong r0")
    r0=reference_metrics(historical)
    refs=bundle["references"]; need(set(refs)<=set(MODELS), "unknown reference")
    need(len({v["semantic_recipe_sha256"] for v in refs.values()}) <= 1,"different semantic input recipes")
    limits={k:validate_ref(k,v) for k,v in refs.items()}
    result={"binding":digest(bundle),"complete":set(refs)==set(MODELS),"reference_metrics":limits,
            "historical_s20ab":r0,"protocol":MAIN}
    if result["complete"]:
        aa,v,s=[limits[k] for k in ("A","V","S")]
        result["target"]={"ttft_ms":min(1.25*aa["ttft_ms"],v["ttft_ms"],s["ttft_ms"]),
                          "tpot_ms":min(v["tpot_ms"],s["tpot_ms"]),"e2e_s":min(v["e2e_s"],s["e2e_s"])}
    return result


def compare(bundle, metrics):
    v=summary(bundle)
    need(all(math.isfinite(metrics[k]) and metrics[k]>0 for k in KEYS),"bad candidate metrics")
    target=v.get("target")
    partial={n:{k:metrics[k]-(1.25 if n=="A" else 1)*m[k] for k in (("ttft_ms",) if n=="A" else KEYS)}
             for n,m in v["reference_metrics"].items()}
    return {"reference_sha256":v["binding"],"metrics":metrics,"comparison":"three" if target else "partial",
            "delta_by_reference":partial,"delta_vs_r0":{k:metrics[k]-v["historical_s20ab"][k] for k in KEYS},
            "target":target,"thresholds_met":bool(target and metrics["ttft_ms"]<=target["ttft_ms"]
             and metrics["tpot_ms"]<=target["tpot_ms"] and metrics["e2e_s"]<target["e2e_s"])}


def judge(bundle, groups):
    v=summary(bundle);need(v["complete"],"all three references are required")
    for g in groups:
        need(g["registration"]["reference_sha256"]==g["reference_sha256"]==v["binding"]
             and g["protocol_sha256"]==protocol_hash(MAIN),"candidate reference mismatch")
        reg=g["registration"]
        need(reg["protocol_sha256"]==g["protocol_sha256"] and sha(reg["screen_config_sha256"])
             and reg["screen_config_sha256"]==g["execution_config_sha256"]
             and reg["screen_assets_sha256"]==digest(g["assets"]), "drift from passing screen")
        need(all(x["all_clear"] for x in g["compilation"].values())
             and set(g["compilation"])=={"sj9","sj10"} and g["raw_audit"]["files"],"raw evidence absent")
    return dict(_judge_two_registered_groups(bundle["r0"],groups,v["target"]),reference_sha256=v["binding"])
