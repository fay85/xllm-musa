# mhcpresinkhorn_rmsnorm

A PyPTO-Pro MIX kernel for Ascend 950PR (A5), fusing hyper-connection inverse RMS, gating matmul, pre/post gates, Sinkhorn normalization, and RmsNorm. The K dimension is split across 28 Cube cores; Vector processing overlaps the matmul and computes the final normalized output.

All token counts use the single `kernel_k` K-split path (it is the only kernel path this file has ever had). Workspaces are cached per (t, device, stream): entries created while an ACL graph is being captured are pinned, while every eager call uses a bounded LRU (8 entries), so concurrent streams never share one workspace.

[中文说明](../../../../zh/ops/pypto-pro/mhcpresinkhorn_rmsnorm/README_zh.md)

## Requirements

- Ascend 950PR (A5) NPU with a compatible CANN environment (the original development environment used CANN 9.2.0).
- Python with compatible PyTorch, `torch_npu`, and PyPTO-Pro installed.
- `PTO_TILE_LIB_CODE_PATH` pointing to a local `pto-isa` checkout for PyPTO-Pro compilation; it is not included here.
- Optionally set `TILE_FWK_DEVICE_ID` to the target NPU index (default: `0`).

Load the CANN environment for your installation before running the example. The first call triggers JIT compilation. This directory does not provide CPU execution or a precompiled kernel.

## Public API

The entry point is `mhcpresinkhorn_rmsnorm(x, hc_fn, hc_scale, hc_base, gamma, hc_mult=4, sinkhorn_iters=20, eps=1e-6, rms_eps=1e-6)`.

| Input | Shape | dtype |
| --- | --- | --- |
| `x` | `[t,4,4096]` | bf16 |
| `hc_fn` | `[24,16384]` | bf16 |
| `hc_scale` | `[3]` | fp32 |
| `hc_base` | `[24]` | fp32 |
| `gamma` | `[4096]` | bf16 |

Provide a nonempty token dimension (`t >= 1`). `x`, `hc_fn`, `hc_scale`, `hc_base`, and `gamma` must all be contiguous, and `x` must be on an NPU (a CPU input raises `ValueError`). It only accepts `hc_mult=4`, `sinkhorn_iters=20`, and `eps=1e-6`. `rms_eps` is an argument for signature compatibility but only its default `1e-6` is accepted. The wrapper selects the NPU via `TILE_FWK_DEVICE_ID`, moves inputs to it, and allocates all outputs on it.

Returns `(y, rstd, post, comb)` with shapes and dtypes `[t,4096]` bf16, `[t,1]` fp32, `[t,4]` fp32, and `[t,4,4]` fp32, respectively. **`y` is already RmsNorm-normalized**; the intermediate unnormalized bf16 value is not returned.

## Run the example

The full runnable demo is `mhcpresinkhorn_rmsnorm_demo.py` in this directory, and its source is reproduced in [English design](design.md) section 8. From the xllm repository root, in a configured Ascend environment:

```bash
TILE_FWK_DEVICE_ID=0 PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa \
  python3 docs/src/content/docs/en/ops/pypto-pro/mhcpresinkhorn_rmsnorm/mhcpresinkhorn_rmsnorm_demo.py 3
```

The optional trailing argument is `t` (default `3`). The demo constructs sample inputs and compares all four outputs against a PyTorch reference; it requires a working NPU environment. See [English design](design.md) section 8 for the full source.

## Documentation

- Design: [English](design.md) / [中文](../../../../zh/ops/pypto-pro/mhcpresinkhorn_rmsnorm/design.md)
- API reference: [English](api_reference.md) / [中文](../../../../zh/ops/pypto-pro/mhcpresinkhorn_rmsnorm/api_reference.md)
- Runnable demo: source in [English design](design.md) section 8 / [中文设计说明](../../../../zh/ops/pypto-pro/mhcpresinkhorn_rmsnorm/design.md)第 7 节
