# Load /workspace/config.json into the env used by start/bench scripts.
# Sourced from env.sh after profile defaults. VL_* exports still win when set
# before sourcing env.sh. Does not re-force binary_default_on flags.
#
# Usage: source this only via env.sh (needs python3 + jq-free).

CONFIG_JSON=${VL_CONFIG_JSON:-/workspace/config.json}
if [[ ! -f $CONFIG_JSON ]]; then
  return 0 2>/dev/null || true
fi

eval "$(CONFIG_JSON="$CONFIG_JSON" python3 - <<'CFGJSON'
import json, os, shlex
from pathlib import Path

path = Path(os.environ["CONFIG_JSON"])
c = json.loads(path.read_text())
ex = c.get("explicit", {})
topo = c.get("topology", {})
paths = c.get("paths", {})

def exp(name, val):
    if val is None:
        return
    if isinstance(val, bool):
        val = "true" if val else "false"
    print(f"export {name}={shlex.quote(str(val))}")

# Only set if caller did not already export VL_* / name.
# We always overwrite unprefixed stack vars from config (source of truth),
# except when VL_* was provided (checked in env.sh before this for some).
# Here we export the canonical names that start_stack consumes.

exp("MAX_MEMORY_UTILIZATION", ex.get("max_memory_utilization"))
exp("MTP_K", ex.get("num_speculative_tokens"))
exp("ENABLE_MTP", ex.get("enable_mtp", True))
exp("SPECULATIVE_ALGORITHM", ex.get("speculative_algorithm", "MTP"))
exp("ENABLE_GRAPH_VMM_POOL", ex.get("enable_graph_vmm_pool"))
exp("CONC", ex.get("max_concurrent_requests"))
exp("ENABLE_DISAGG_PD", ex.get("enable_disagg_pd", True))
exp("ENABLE_NATIVE_SERVICE_ADAPTER", ex.get("enable_native_service_adapter", True))
exp("KV_CACHE_TRANSFER_MODE", ex.get("kv_cache_transfer_mode", "PULL"))
exp("KV_SPLIT_SIZE", ex.get("kv_split_size", 1))
piece = ex.get("enable_prefill_piecewise_graph") or {}
if isinstance(piece, dict):
    exp("ENABLE_PREFILL_PIECEWISE_GRAPH_PREFILL", piece.get("prefill", True))
    exp("ENABLE_PREFILL_PIECEWISE_GRAPH_DECODE", piece.get("decode", False))
exp("PREFILL_BATCH_WAIT_MS", ex.get("prefill_batch_wait_ms"))
exp("PREFILL_CHUNK_TOKENS", ex.get("prefill_chunk_tokens"))
exp("ENABLE_IN_BATCH_PREFIX_CACHE", ex.get("enable_in_batch_prefix_cache"))
exp("MAX_TOKENS_PER_BATCH_PREFILL", ex.get("max_tokens_per_batch_prefill"))
exp("MAX_TOKENS_PER_BATCH_DECODE", ex.get("max_tokens_per_batch_decode"))
exp("MAX_TOKENS_FOR_GRAPH_MODE_PREFILL", ex.get("max_tokens_for_graph_mode_prefill"))
exp("MAX_TOKENS_FOR_GRAPH_MODE_DECODE", ex.get("max_tokens_for_graph_mode_decode"))
exp("MAX_TOKENS_PER_CHUNK_FOR_PREFILL_DECODE", ex.get("max_tokens_per_chunk_for_prefill_decode"))
exp("OMP_NUM_THREADS", ex.get("omp_num_threads"))
if ex.get("profile"):
    exp("PROFILE", ex.get("profile"))

if topo.get("prefill_gpus"):
    exp("PREFILL_GPUS", topo["prefill_gpus"])
if topo.get("decode_gpus"):
    exp("DECODE_GPUS", topo["decode_gpus"])
if topo.get("prefill_nnodes") is not None:
    exp("VL_PREFILL_NNODES", topo["prefill_nnodes"])

if paths.get("model"):
    exp("MODEL", paths["model"])
if paths.get("model_id"):
    exp("MODEL_ID", paths["model_id"])
if paths.get("xllm_bin"):
    exp("XLLM_BIN", paths["xllm_bin"])
if paths.get("flashinfer_ops_path"):
    exp("MUSA_OPS_DIR", paths["flashinfer_ops_path"])
if paths.get("musa_preload"):
    exp("MUSA_PRELOAD", paths["musa_preload"])
if paths.get("master_bin"):
    exp("MASTER_BIN", paths["master_bin"])
if paths.get("etcd_bin"):
    exp("ETCD_BIN", paths["etcd_bin"])

ports = ex.get("ports") or {}
if ports.get("etcd_client") is not None:
    exp("ETCD_CLIENT_PORT", ports["etcd_client"])
if ports.get("etcd_peer") is not None:
    exp("ETCD_PEER_PORT", ports["etcd_peer"])
if ports.get("master_http") is not None:
    exp("MASTER_HTTP_PORT", ports["master_http"])
if ports.get("master_rpc") is not None:
    exp("MASTER_RPC_PORT", ports["master_rpc"])

print("export CONFIG_JSON_LOADED=1")
print(f"export CONFIG_JSON={shlex.quote(str(path))}")
CFGJSON
)"
