#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE.
set -euo pipefail
source "$(dirname "$0")/env.sh"

python3 - "$PROFILE" "$MAX_MEMORY_UTILIZATION" \
  "$BATCH_CREDIT_MAX_GRANT_UNITS" "$BATCH_CREDIT_MAX_GRANT_UNITS_PER_ROUTER_TENANT" \
  "$BATCH_CREDIT_SAFETY_PERMYRIAD" "$MTP_K" "$CONC" \
  "$ENABLE_GRAPH_VMM_POOL" "$XLLM_BIN" "$MASTER_BIN" \
  "$PREFILL_GPUS" "$DECODE_GPUS" "$PREFILL_CHUNK_TOKENS" \
  "${CONFIG_JSON:-/workspace/config.json}" <<'PY'
import json
import sys
from pathlib import Path

keys = ["profile", "max_memory_utilization", "credit_cap", "router_tenant_credit_cap",
        "credit_safety_permyriad", "mtp_k", "max_concurrency", "graph_vmm_pool",
        "engine_binary", "service_binary", "prefill_gpus", "decode_gpus", "prefill_chunk_tokens",
        "config_json"]
values = dict(zip(keys, sys.argv[1:]))
values["max_memory_utilization"] = float(values["max_memory_utilization"])
for name in ["credit_cap", "router_tenant_credit_cap", "credit_safety_permyriad",
             "mtp_k", "max_concurrency", "prefill_chunk_tokens"]:
    values[name] = int(values[name])
values["graph_vmm_pool"] = values["graph_vmm_pool"] == "true"
values.update({"load_balance_policy": "RR", "prefix_cache": True,
               "prefill_graph": "piecewise", "decode_graph": "full"})
cfg_path = Path(values["config_json"])
if cfg_path.is_file():
    cfg = json.loads(cfg_path.read_text())
    values["config_summary"] = {
        "binary_default_on": sorted(k for k, v in (cfg.get("binary_default_on") or {}).items()
                                    if k != "_doc" and v is True),
        "explicit": {k: v for k, v in (cfg.get("explicit") or {}).items() if k != "_doc"},
        "binary_md5": (cfg.get("binary") or {}).get("md5"),
        "how_to_change": cfg.get("how_to_change"),
    }
print(json.dumps(values, indent=2))
PY
