# CUDA Graph Path on MUSA (torch_musa + mcc_wrapper)

Port **xllm-git-master** to run the **CUDA graph serving stack** on MUSA hardware.

## Strategy

| Layer | Approach |
|-------|----------|
| Build | `USE_CUDA=ON` + `XLLM_TORCH_MUSA=ON`; `.cu` via `mcc_wrapper` + musamapping |
| Runtime | `CudaGraphExecutorImpl` + FlashInfer + full `cuda_kernels` via torch_musa CUDA compat |
| Not used | Native `USE_MUSA` / `musa_kernels` (no graph executor) |
| Graph logic | **Do not change**; validate vs **sglang** after build |

Reference (build only, no source port): `/data/jiacun/xllm/xllm-musa` — same CUDA-path MUSA build (`USE_CUDA` + `XLLM_TORCH_MUSA` + `mcc_wrapper`), **no CUTLASS** in `cuda_kernels`.

---

## CUTLASS vs MUSA libs (important)

**XLLM_TORCH_MUSA does not need NVIDIA CUTLASS** (SM90/SM100/SM120 FP8 GEMM). Those are CUDA-SM-specific and should be **excluded** from the build.

**Need MUSA libs instead:**

| Category | Libraries |
|----------|-----------|
| Runtime | `musa`, `musart` |
| DNN | `mudnn`, `mudnn_base`, `mudnn_ops`, `mudnn_tensor` |
| BLAS | `mublas`, `mublasLt` (via torch_musa as needed) |
| Collective | `mccl` |
| PyTorch | `musa_python`, torch_musa kernel libs |
| Build | `mcc_wrapper`, `libMusaMapping.so` |

### git-master gap

- Root CMake still adds CUTLASS includes for all `USE_CUDA`.
- `kernels/cuda/CMakeLists.txt` only skips CUTLASS SM **static libs**; still compiles `cutlass_extensions/`, `cutlass_w8a8/`, `fp8_scaled_*`.

### CMake TODO

- [ ] `if(USE_CUDA AND NOT XLLM_TORCH_MUSA)` around CUTLASS includes
- [ ] Remove CUTLASS sources from `cuda_kernels` when `XLLM_TORCH_MUSA`
- [ ] No `ENABLE_SCALED_MM_SM*` / `ENABLE_FP8`
- [ ] Link `musa`, `musart`, (+ `mudnn` if required)

W8A8/FP8 on MUSA: FlashInfer / TVM FFI / torch_musa — not CUTLASS.

---

## Do not modify (graph logic)

- `cuda_graph_executor_impl.{cpp,h}`
- `piecewise_graphs.cpp`, `global_capture_instance.cpp`

Include shims only; no capture/replay algorithm changes.

---

## Infrastructure status

**Done:** XLLM_TORCH_MUSA cmake, platform, MCCL, kernel MUSAGuard, build script.

**Todo:** CUTLASS off (P0), vmm_api, cuda_utils, ATen include shims, Rust seed, MUSA link.

---

## Validation vs sglang (later)

Graph capture/replay, correctness, MemPool, perf, MCCL multi-GPU.

```bash
cd /workspace/xllm-git-master && CLEAN_REBUILD=0 ./_build_cuda_graph_musa.sh
```
