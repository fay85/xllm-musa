# Stack config (`/workspace/config.json`)

Single source of truth for the slim PD stack on `xllm_musa_mt_main`.

## Sections

| Section | Meaning |
|---------|---------|
| `binary_default_on` | Already ON in the wave2 MUSA binary / fixed runtime. **Do not** re-export these in scripts. |
| `explicit` | Values `start_stack.sh` still applies. **Edit here** to retune. |
| `topology` / `paths` | GPUs, model, binaries, kernel cache. |
| `binary` | Installed engine path + md5. |
| `docs` / `how_to_change` | Short human notes. |

## /workspace entrypoints

| Script | Role |
|--------|------|
| `/workspace/start_xllm_service.sh` | xllm-service only (etcd + master + coordinator) |
| `/workspace/start_xllm_stack.sh` | full PD: service + DP3CP2 Prefill + DP2TP1 Decode |
| `/workspace/start_all.sh` / `stop_all.sh` | cold start / stop wrappers |

See `/workspace/README_START.md`.

## Change a setting

1. Edit `/workspace/config.json` (`explicit.*` or `paths.*`).
2. `bash /workspace/stop_all.sh`
3. `bash /workspace/start_xllm_stack.sh`

One-off without editing the file:

```bash
VL_MTP_K=1 bash start_stack.sh
VL_ENABLE_MTP=false bash start_stack.sh   # same as start_stack_sweep_k0.sh
VL_CONC=8 bash start_stack.sh
```

## Sweep helpers

- `start_stack_sweep_k0.sh` → `VL_ENABLE_MTP=false` then `start_stack.sh`
- `start_stack_sweep_k1.sh` → `VL_MTP_K=1` then `start_stack.sh`

## Note

`libmtml.so` is host-mounted and is **not** part of the Docker image.
