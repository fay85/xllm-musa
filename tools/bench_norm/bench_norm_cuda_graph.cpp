// Microbench: time xllm CUDA-MUSA rms_norm kernels at the shapes the
// Qwen3.5-27B production path hits in graph + eager mode.
//
// Linked against libcuda_kernels.a (musamapping-rewritten to xllm::kernel::musa)
// via the same mcc_wrapper that builds the main xllm target. We declare the
// entries in `xllm::kernel::cuda` and rely on mcc + libMusaMapping to rewrite
// `cuda -> musa` at compile time, matching the symbols in the static archive.
//
// Usage:
//   bench_norm                       # default: bf16, hidden=5120, full sweep
//   bench_norm --hidden=2048         # match the triton bench in /workspace/rms_norm

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace xllm::kernel::cuda {
void rms_norm(torch::Tensor output,
              torch::Tensor input,
              torch::Tensor weight,
              double eps);
void fused_add_rms_norm(torch::Tensor& input,
                        torch::Tensor& residual,
                        torch::Tensor& weight,
                        double epsilon);
void gemma_rms_norm(torch::Tensor output,
                    torch::Tensor input,
                    torch::Tensor weight,
                    double eps);
void fused_add_gemma_rms_norm(torch::Tensor& input,
                              torch::Tensor& residual,
                              torch::Tensor& weight,
                              double epsilon);
}  // namespace xllm::kernel::cuda

namespace {

// Returns average per-call milliseconds across `iters` after `warmup` runs.
double bench_kernel(const std::function<void()>& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  cudaDeviceSynchronize();
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < iters; ++i) fn();
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms;
  cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  return static_cast<double>(ms) / iters;
}

double gbps(double per_call_ms, double bytes_moved) {
  if (per_call_ms <= 0) return 0;
  return (bytes_moved / per_call_ms * 1e3) / (1024.0 * 1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
  int hidden = 5120;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--hidden=", 9) == 0) {
      hidden = std::atoi(argv[i] + 9);
    }
  }

  torch::Device dev("musa:0");
  c10::cuda::CUDAGuard guard(dev);
  auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);

  auto weight = torch::randn({hidden}, opts_bf16);

  const std::vector<int> num_tokens_list = {
      1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

  printf("# xllm CUDA-MUSA rms_norm microbench, dtype=bf16, hidden=%d\n", hidden);
  printf("# Bytes counted: input + weight + output (rms_norm/gemma_rms_norm)\n");
  printf("# Bytes counted: input + residual_io (write) + weight + input_out\n");
  printf("#                                                  (fused_add variants)\n");
  printf("# All times are AVG of 50 iters after 5 warmup runs.\n\n");
  printf("%-12s %16s %16s %16s %16s\n",
         "num_tokens",
         "rms_norm us|GB/s",
         "gemma us|GB/s",
         "fused_add us|GB/s",
         "fused_add_gem us|GB/s");
  printf("%s\n", std::string("------------------------------------------------------------------------------------------------").c_str());

  for (int n : num_tokens_list) {
    auto x = torch::randn({n, hidden}, opts_bf16);
    auto residual = torch::randn({n, hidden}, opts_bf16);
    auto out = torch::empty_like(x);

    // Pre-clone the inputs once outside the timing loop. fused_add variants
    // mutate input and residual in-place, but we want the SAME pristine
    // inputs each iter -- using clone() inside the loop would inflate the
    // measurement with allocator overhead and bias us against the kernels.
    // Instead, we measure the kernel call against a STATIC copy that gets
    // overwritten each iter; the result is unused so the staleness is fine.
    auto x_clone = x.clone();
    auto res_clone = residual.clone();

    const double bytes_rw = static_cast<double>(n) * hidden * 2;  // bf16
    const double bytes_weight = static_cast<double>(hidden) * 2;

    // rms_norm: read x, weight; write out
    double t_rms = bench_kernel(
        [&]() { xllm::kernel::cuda::rms_norm(out, x, weight, 1e-6); }, 5, 50);
    double bw_rms = 2 * bytes_rw + bytes_weight;

    // gemma_rms_norm
    double t_gem = bench_kernel(
        [&]() { xllm::kernel::cuda::gemma_rms_norm(out, x, weight, 1e-6); },
        5,
        50);
    double bw_gem = bw_rms;

    // fused_add_rms_norm: in-place mutates input and residual; reads each.
    // Mem traffic: read input, read residual, read weight, write residual,
    // write input. Use the cloned tensors per call -- it does mutate them
    // but the measurement is still on the captured op.
    double t_fa = bench_kernel(
        [&]() {
          xllm::kernel::cuda::fused_add_rms_norm(
              x_clone, res_clone, weight, 1e-6);
        },
        5,
        50);
    double bw_fa = 4 * bytes_rw + bytes_weight;

    double t_fag = bench_kernel(
        [&]() {
          xllm::kernel::cuda::fused_add_gemma_rms_norm(
              x_clone, res_clone, weight, 1e-6);
        },
        5,
        50);
    double bw_fag = bw_fa;

    printf("%-12d %6.1f us %6.1f GB/s   %6.1f us %6.1f GB/s   "
           "%6.1f us %6.1f GB/s   %6.1f us %6.1f GB/s\n",
           n,
           t_rms * 1000,
           gbps(t_rms, bw_rms),
           t_gem * 1000,
           gbps(t_gem, bw_gem),
           t_fa * 1000,
           gbps(t_fa, bw_fa),
           t_fag * 1000,
           gbps(t_fag, bw_fag));
  }
  printf("\ndone.\n");
  return 0;
}
