# Mate GDN Prefill Double-Cumsum Fix - 2026-07-11

## Root Cause

xLLM's Mate GDN prefill path was passing a **pre-computed cumulative sum** of the
log-gate (`g_cumsum`) to the Mate `_log1` kernel specialization. The `_log1`
kernel **already performs the chunk-local cumulative sum internally** (it is the
`is_log_space=True` code path in SGLang's `mate.chunk_gated_delta_rule`). Passing
`g_cumsum` instead of raw `log(alpha)` caused the decay integration to be applied
**twice**, corrupting every token after the first one in each chunk.

This bug affected all prefill sequence lengths - divergence from SGLang was
observable starting at ISL ~92 and persisted at every longer length tested.

### Secondary issue: missing `kkt_solve`

The Mate prefill kernel expects a pre-computed `a` parameter
(`A = (I + lower_tri(beta * K @ K^T))^{-1}`). The old code passed a **zero
tensor** (`a_dummy`). SGLang's Mate library computes this via
`mate.gdn_kernels.tilelang.gdn_kkt_solve` before calling the prefill kernel.

### Tertiary issue: state layout for chunked prefill

The chunked-prefill path (Path 3 in `forward()`) conditioned the state transpose
on `fla_ssm_state_layout`, which is `true` for Qwen3.5. However, after the first
prefill chunk the SSM cache carries Mate k-last layout `(B, H, V, K)`, not FLA
k-first `(B, H, K, V)`. The guard `!fla_ssm_state_layout` was therefore never
taken, so the state was passed to `chunk_gated_delta_rule` in the wrong layout
on continuation chunks.

## Fix

### `xllm/core/kernels/musa/gdn_prefill.cpp`

1. **Removed `chunk_local_cumsum_log_space()`** - no longer needed; the Mate
   `_log1` kernel does the cumsum internally.

2. **Pass raw `g_log` to Mate kernel** (was `g_cumsum`):
   ```
   run(..., to_ffi_tensor(g_log), ...)   // was: to_ffi_tensor(g_cumsum)
   ```
   This matches SGLang's `is_log_space=True` semantics.

3. **Added `kkt_solve()` function** - computes
   `A = (I + lower_tri(beta * K @ K^T))^{-1}` using PyTorch forward-substitution,
   matching SGLang's `mate.gdn_kernels.tilelang.gdn_kkt_solve`. The result is
   passed as the `a` parameter to the Mate kernel (was: zero tensor).

4. **Minor**: `output.slice(...)` now calls `.contiguous()` after slicing to
   avoid non-contiguous tensor downstream.

### `xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp`

5. **Always transpose state on chunked steps** (L1497): removed the
   `!fla_ssm_state_layout` guard. The SSM cache always stores Mate k-last
   `(B, H, V, K)` after the first prefill chunk, and `chunk_gated_delta_rule`
   expects k-first `(B, H, K, V)`.

6. **Always write back in Mate k-last layout** (L1541-1544): removed the
   `fla_ssm_state_layout` ternary; always transpose before storing to ensure
   Mate decode sees a consistent k-last state.

## Verification

### Code review

- Caller (`qwen3_gated_delta_net_base.cpp:1332`): `mate_params.g = g` - passes
  raw `g` from `fused_gdn_gating`, no cumsum applied.
- SGLang reference (`gdn_flashinfer.py extend()`): calls
  `self._prefill_fn(g=g.squeeze(0), ..., is_log_space=True)` - same semantics.
- Mate library (`mate/gdn_prefill.py`): `is_log_space=True` is the default;
  docstring confirms `g` is `log(alpha)` and the kernel handles cumsum.
- `kkt_solve` output shape `[B, T, Hv, chunk_size]` matches Mate kernel's
  expected `a` parameter.
- State layout: Mate returns k-last `(B, H, V, K)`, stored directly to
  `ssm_cache` without transpose.
- PyTorch fallback (`chunk_gated_delta_rule` L109-247): unaffected - does its
  own `g.cumsum(-1)` at L182, receives raw `g`.

### Pending runtime validation

The binary has been rebuilt but not yet retested with the needle-in-haystack
test. The fix needs to be validated by:

1. Rebuild: `MAX_JOBS=32 bash _build_cuda_graph_musa.sh`
2. Restart xllm server: `ENABLE_GRAPH=1 MUSA_VISIBLE_DEVICES=0 PORT=8092 DEVICE_INDEX=0 bash run_xllm_musa.sh --background`
3. Run needle test at ISL=629 - expect xllm to return ` M` (start of MANGO-9132),
   matching SGLang.
4. Run threshold sweep (ISL 92-700) - expect xllm to match SGLang at all lengths.

## Files Changed

| File | Lines | Description |
|------|-------|-------------|
| `xllm/core/kernels/musa/gdn_prefill.cpp` | +98 -34 | Remove `chunk_local_cumsum_log_space`, add `kkt_solve`, pass `g_log` instead of `g_cumsum`, pass `a` instead of `a_dummy` |
| `xllm/core/layers/musa/qwen3_gated_delta_net_base.cpp` | +9 -3 | Always transpose state on chunked prefill steps, always write back k-last |

## Environment

- Container: `xllm-musa2.9.1-sdk5.1-dev`
- Mate version: `0.2.3.dev20260602+mu510` (matched between xLLM and SGLang)
- tvm-ffi version: `0.1.9.post3.dev0+musa.1.gf6b52d7f4.d20260603`
- Model: Qwen3.5-27B (64 layers: 48 GDN + 16 full-attention, MRoPE)
- SGLang reference server: port 30000, device 2
- xLLM server: port 8092, device 0
