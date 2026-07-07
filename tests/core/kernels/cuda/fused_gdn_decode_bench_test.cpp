/* Benchmark xllm's fused_gated_delta_rule_decode kernel with Qwen3.5-27B shapes.

  B=1, H=16 (num_k_heads), HV=48 (num_v_heads), K=128, V=128
  (same shapes as sglang trace: grid=[4,48,1], block=[32,1,1])

  Uses std::chrono with torch::cuda::synchronize() for timing.
  Batched approach: N calls between sync points, then divide by N.
  C++ launch overhead is ~1-5us, so batched time ≈ kernel time.
*/
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <Python.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <functional>

#include "core/kernels/param.h"

// gdn_ops.h declares fused_gated_delta_rule_decode in namespace cuda, but mcc
// compiles .cu files with cuda→musa namespace remapping, so the actual symbol
// in libcuda_kernels.a is xllm::kernel::musa::fused_gated_delta_rule_decode.
// Forward-declare in the musa namespace to match the linker symbol.
namespace xllm::kernel::musa {
torch::Tensor fused_gated_delta_rule_decode(xllm::kernel::MateGatedDeltaRuleDecodeParams&);
}

namespace xllm::kernel::cuda {
namespace test {
using xllm::kernel::musa::fused_gated_delta_rule_decode;

// Initialize MUSA backend by importing torch_musa from Python.
// In C++ there's no exported init function; Python import triggers
// RegisterPrivateUse1HooksInterface via _musa_init.
struct MusaInit {
  MusaInit() {
    Py_Initialize();
    PyRun_SimpleString("import torch; import torch_musa; "
                       "print('torch_musa imported, device count:', torch.musa.device_count())");
    fprintf(stderr, "MusaInit: Python init done\n");
  }
};
static MusaInit musa_init;

// Force device sync by copying a tiny GPU tensor to CPU.
// Avoids including MUSA/CUDA runtime headers that #define cuda→musa.
static inline void dev_sync() {
  torch::empty({1}, torch::TensorOptions().device(torch::kPrivateUse1, 0)).cpu();
}

class FusedGdnDecodeBench : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::manual_seed(2026);
    device_ = torch::Device(torch::kPrivateUse1, 0);
    fprintf(stderr, "Device type: %d (CUDA=%d, PrivateUse1=%d)\n",
            (int)device_.type(), (int)torch::kCUDA, (int)torch::kPrivateUse1);
    // Test: can we create a tensor on this device?
    auto test = torch::empty({4}, torch::TensorOptions().device(device_));
    fprintf(stderr, "Test tensor device: %d, is_musa: %d\n",
            (int)test.device().type(), (int)(test.device().type() == torch::kPrivateUse1));
    // Test: can we fill it?
    test.zero_();
    fprintf(stderr, "zero_ worked\n");
    // Test: randn on CPU then .to()
    auto cpu_t = torch::randn({4});
    auto musa_t = cpu_t.to(device_);
    fprintf(stderr, "randn->to worked, device: %d\n", (int)musa_t.device().type());
  }

  // Batched timing: N calls between sync points.
  // Returns us per call (includes ~1-5us C++ launch overhead).
  double bench_batched(const std::function<void()>& fn,
                       int warmup = 50, int rep = 1000) {
    for (int i = 0; i < warmup; ++i) fn();
    dev_sync();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rep; ++i) fn();
    dev_sync();
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return total_us / rep;
  }

  // Per-call timing: sync after each call. Higher overhead but gives min/max.
  struct PerCallResult { double mean, min, max, p50; };
  PerCallResult bench_per_call(const std::function<void()>& fn,
                                int warmup = 50, int rep = 200) {
    for (int i = 0; i < warmup; ++i) fn();
    dev_sync();

    std::vector<double> times;
    times.reserve(rep);
    for (int i = 0; i < rep; ++i) {
      auto t0 = std::chrono::high_resolution_clock::now();
      fn();
      dev_sync();
      auto t1 = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    double sum = 0;
    for (auto t : times) sum += t;
    return {sum / rep, times.front(), times.back(), times[rep / 2]};
  }

  torch::Device device_{torch::kCPU};
};

// Qwen3.5-27B GDN decode shapes (TP=1):
//   H=16, HV=48, K=128, V=128
//   mixed_qkv: [B, 2*H*K + HV*V] = [B, 10240]
//   state:     [pool, HV, V, K] = [pool, 48, 128, 128] fp32
//   output:    [B, HV, V] = [B, 48, 128]
TEST_F(FusedGdnDecodeBench, Qwen35_27B_B1) {
  const int64_t B = 1;
  const int64_t H = 16, HV = 48, K = 128, V = 128;
  const int64_t pool = 64;
  const int64_t qk_cols = H * K;
  const int64_t v_cols = HV * V;
  const int64_t mixed_dim = 2 * qk_cols + v_cols;

  const auto bf16 = torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  const auto f32 = torch::TensorOptions().device(device_).dtype(torch::kFloat32);

  torch::Tensor mixed_qkv = torch::randn({B, mixed_dim}, bf16) * 0.05f;
  torch::Tensor state = torch::zeros({pool, HV, V, K}, f32);
  torch::Tensor A_log = torch::full({HV}, -2.0f, bf16);
  torch::Tensor a = torch::randn({B, HV}, bf16) * 0.01f;
  torch::Tensor dt_bias = torch::zeros({HV}, bf16);
  torch::Tensor b = torch::randn({B, HV}, bf16) * 0.01f;
  torch::Tensor state_indices =
      torch::arange({B}, torch::TensorOptions().device(device_).dtype(torch::kInt32));
  torch::Tensor output = torch::empty({B, HV, V}, bf16);

  MateGatedDeltaRuleDecodeParams params;
  params.mixed_qkv = mixed_qkv;
  params.state = state;
  params.A_log = A_log;
  params.a = a;
  params.dt_bias = dt_bias;
  params.b = b;
  params.state_indices = state_indices;
  params.num_k_heads = H;
  params.num_v_heads = HV;
  params.head_k_dim = K;
  params.head_v_dim = V;
  params.scale = 1.0 / std::sqrt(static_cast<double>(K));
  params.use_qk_l2norm = true;
  params.decode_output = output;

  // Correctness: output should be finite
  fused_gated_delta_rule_decode(params);
  ASSERT_TRUE(torch::isfinite(output.cpu()).all().item<bool>())
      << "kernel produced non-finite output";

  auto fn = [&]() { fused_gated_delta_rule_decode(params); };
  double batched = bench_batched(fn, 50, 1000);
  auto pc = bench_per_call(fn, 50, 200);

  printf("\n  === xllm fused_gated_delta_rule_decode (B=%ld, H=%ld, HV=%ld, K=%ld, V=%ld) ===\n",
         B, H, HV, K, V);
  printf("  Batched (1000 reps): %.1f us/call\n", batched);
  printf("  Per-call (200 reps):  mean=%.1f  p50=%.1f  min=%.1f  max=%.1f us\n",
         pc.mean, pc.p50, pc.min, pc.max);
  printf("  Grid: (%ld, %ld, 1) = %ld blocks, Block: %d threads\n",
         B, HV, B * HV, (int)std::max(K, V));
  printf("  For 48 GDN layers/step: %.1f us total (batched)\n", batched * 48);
}

TEST_F(FusedGdnDecodeBench, Qwen35_27B_B2_B4) {
  const int64_t H = 16, HV = 48, K = 128, V = 128;
  const int64_t qk_cols = H * K;
  const int64_t v_cols = HV * V;
  const int64_t mixed_dim = 2 * qk_cols + v_cols;
  const auto bf16 = torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  const auto f32 = torch::TensorOptions().device(device_).dtype(torch::kFloat32);

  for (int64_t B : {2, 4}) {
    int64_t pool = std::max(B + 16, (int64_t)64);
    torch::Tensor mixed_qkv = torch::randn({B, mixed_dim}, bf16) * 0.05f;
    torch::Tensor state = torch::zeros({pool, HV, V, K}, f32);
    torch::Tensor A_log = torch::full({HV}, -2.0f, bf16);
    torch::Tensor a = torch::randn({B, HV}, bf16) * 0.01f;
    torch::Tensor dt_bias = torch::zeros({HV}, bf16);
    torch::Tensor b = torch::randn({B, HV}, bf16) * 0.01f;
    torch::Tensor state_indices =
        torch::arange({B}, torch::TensorOptions().device(device_).dtype(torch::kInt32));
    torch::Tensor output = torch::empty({B, HV, V}, bf16);

    MateGatedDeltaRuleDecodeParams params;
    params.mixed_qkv = mixed_qkv;
    params.state = state;
    params.A_log = A_log;
    params.a = a;
    params.dt_bias = dt_bias;
    params.b = b;
    params.state_indices = state_indices;
    params.num_k_heads = H;
    params.num_v_heads = HV;
    params.head_k_dim = K;
    params.head_v_dim = V;
    params.scale = 1.0 / std::sqrt(static_cast<double>(K));
    params.use_qk_l2norm = true;
    params.decode_output = output;

    auto fn = [&]() { fused_gated_delta_rule_decode(params); };
    double batched = bench_batched(fn, 50, 1000);
    auto pc = bench_per_call(fn, 50, 200);

    printf("\n  === xllm fused_gated_delta_rule_decode (B=%ld, H=%ld, HV=%ld, K=%ld, V=%ld) ===\n",
           B, H, HV, K, V);
    printf("  Batched (1000 reps): %.1f us/call\n", batched);
    printf("  Per-call (200 reps):  mean=%.1f  p50=%.1f  min=%.1f  max=%.1f us\n",
           pc.mean, pc.p50, pc.min, pc.max);
    printf("  Grid: (%ld, %ld, 1) = %ld blocks\n", B, HV, B * HV);
  }
}

}  // namespace test
}  // namespace xllm::kernel::cuda
