/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// FusedMoE All2All path unit tests
// Tests for the DeepEP communication mode with multi-device setup

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/fused_moe.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/model_stream_registry.h"
#include "platform/platform.h"
#include "tests/core/forked_rank_session.h"
#include "tests/core/layers/mlu/deep_ep_test_scope.h"
#include "util/net.h"
#include "util/tensor_helper.h"

#if defined(USE_MLU)
#include "framework/parallel_state/mlu_process_group.h"
#elif defined(USE_CUDA)
#include "framework/parallel_state/cuda_process_group.h"
#endif

namespace xllm {
namespace layer {
namespace test {

// Special exit code definition for skipping test
constexpr int32_t EXIT_CODE_SKIP = 77;

namespace {
thread_local std::unique_ptr<xllm::ProcessGroup> rank_process_group;

// Helper function to create ProcessGroup
xllm::ProcessGroup* create_test_process_group(int32_t rank,
                                              int32_t world_size,
                                              int32_t port,
                                              const std::string& host,
                                              const torch::Device& device) {
  if (!rank_process_group) {
    rank_process_group =
        xllm::create_process_group(rank,
                                   world_size,
                                   world_size,
                                   port,
                                   false,
                                   host,
                                   "fused_moe_all2all_test_group",
                                   device);
  }
  return rank_process_group.get();
}

struct All2AllTestParams {
  int32_t rank;
  int32_t world_size;
  int32_t port;
  std::string host;
  int32_t device_index;
  // Model parameters
  int64_t hidden_size;
  int64_t intermediate_size;
  int64_t num_experts;
  int64_t top_k;
  int64_t batch_size;
  int64_t seq_len;
  bool is_smoothquant;
  int64_t moe_weight_bits = 8;
  int64_t group_size = 0;
  // Expected output stats (used when perform_precise_validation is true)
  double expected_min = 0.0;
  double expected_max = 0.0;
  double expected_sum = 0.0;
  // When true, validate output against expected_min/max/sum (avoids fragile
  // "any non-zero expected" heuristic; allows validation when all are 0.0).
  bool perform_precise_validation = false;
};

// Helper to create model args
ModelArgs create_model_args(const All2AllTestParams& params) {
  ModelArgs args;
  args.n_routed_experts() = static_cast<int32_t>(params.num_experts);
  args.num_experts_per_tok() = static_cast<int32_t>(params.top_k);
  args.n_group() = 1;
  args.topk_group() = static_cast<int32_t>(params.top_k);
  args.routed_scaling_factor() = 1.0f;
  args.hidden_size() = params.hidden_size;
  args.moe_intermediate_size() = static_cast<int32_t>(params.intermediate_size);
  args.n_shared_experts() = 0;
  args.norm_topk_prob() = true;
  args.hidden_act() = "silu";
  args.scoring_func() = "softmax";
  args.topk_method() = "greedy";
  return args;
}

// Helper to create quant args
QuantArgs create_quant_args(const All2AllTestParams& params) {
  QuantArgs args;
  if (params.is_smoothquant) {
    args.quant_method() = "smoothquant";
    args.bits() = 8;
    args.activation_dynamic() = true;
    args.moe_weight_bits() = params.moe_weight_bits;
    args.group_size() = params.group_size;
  }
  return args;
}

// Helper to create test weights for All2All MoE using seeded tensors
std::unordered_map<std::string, torch::Tensor> create_all2all_test_weights(
    int64_t num_experts,
    int64_t hidden_size,
    int64_t intermediate_size,
    bool is_smoothquant,
    int64_t moe_weight_bits,
    int64_t group_size,
    int64_t world_size,
    const torch::Device& device) {
  std::unordered_map<std::string, torch::Tensor> weight_dict;

  for (size_t expert_id = 0; expert_id < num_experts; ++expert_id) {
    std::string expert_prefix = "experts." + std::to_string(expert_id) + ".";
    std::string seed_prefix =
        "fused_moe_all2all_tests.expert_" + std::to_string(expert_id);

    if (is_smoothquant && moe_weight_bits == 4) {
      CHECK_GT(group_size, 0);
      CHECK_EQ(intermediate_size % world_size, 0);
      const int64_t local_intermediate_size = intermediate_size / world_size;
      test::append_w4a8_expert_weights(weight_dict,
                                       expert_prefix,
                                       seed_prefix,
                                       hidden_size,
                                       intermediate_size,
                                       intermediate_size,
                                       local_intermediate_size,
                                       group_size,
                                       device);
    } else if (is_smoothquant) {
      // Create quantized weights using seeded tensors
      auto gate_weight_fp =
          test::seeded_tensor(seed_prefix + ".gate_proj",
                              {intermediate_size, hidden_size},
                              torch::kBFloat16,
                              device);
      auto gate_qweight = gate_weight_fp.to(torch::kInt8);
      auto gate_scale = test::seeded_tensor(seed_prefix + ".gate_proj.scale",
                                            {intermediate_size},
                                            torch::kFloat32,
                                            device);
      auto gate_smooth = test::seeded_tensor(seed_prefix + ".gate_proj.smooth",
                                             {hidden_size},
                                             torch::kFloat32,
                                             device);

      auto up_weight_fp = test::seeded_tensor(seed_prefix + ".up_proj",
                                              {intermediate_size, hidden_size},
                                              torch::kBFloat16,
                                              device);
      auto up_qweight = up_weight_fp.to(torch::kInt8);
      auto up_scale = test::seeded_tensor(seed_prefix + ".up_proj.scale",
                                          {intermediate_size},
                                          torch::kFloat32,
                                          device);

      auto down_weight_fp =
          test::seeded_tensor(seed_prefix + ".down_proj",
                              {hidden_size, intermediate_size},
                              torch::kBFloat16,
                              device);
      auto down_qweight = down_weight_fp.to(torch::kInt8);
      auto down_scale = test::seeded_tensor(seed_prefix + ".down_proj.scale",
                                            {hidden_size},
                                            torch::kFloat32,
                                            device);
      auto down_smooth = test::seeded_tensor(seed_prefix + ".down_proj.smooth",
                                             {intermediate_size},
                                             torch::kFloat32,
                                             device);

      weight_dict[expert_prefix + "gate_proj.qweight"] = gate_qweight;
      weight_dict[expert_prefix + "gate_proj.per_channel_scale"] = gate_scale;
      weight_dict[expert_prefix + "gate_proj.smooth"] = gate_smooth;

      weight_dict[expert_prefix + "up_proj.qweight"] = up_qweight;
      weight_dict[expert_prefix + "up_proj.per_channel_scale"] = up_scale;
      weight_dict[expert_prefix + "up_proj.smooth"] = gate_smooth;

      weight_dict[expert_prefix + "down_proj.qweight"] = down_qweight;
      weight_dict[expert_prefix + "down_proj.per_channel_scale"] = down_scale;
      weight_dict[expert_prefix + "down_proj.smooth"] = down_smooth;
    } else {
      // Create BF16 weights using seeded tensors
      auto gate_weight = test::seeded_tensor(seed_prefix + ".gate_proj.weight",
                                             {intermediate_size, hidden_size},
                                             torch::kBFloat16,
                                             device);
      auto up_weight = test::seeded_tensor(seed_prefix + ".up_proj.weight",
                                           {intermediate_size, hidden_size},
                                           torch::kBFloat16,
                                           device);
      auto down_weight = test::seeded_tensor(seed_prefix + ".down_proj.weight",
                                             {hidden_size, intermediate_size},
                                             torch::kBFloat16,
                                             device);

      weight_dict[expert_prefix + "gate_proj.weight"] = gate_weight;
      weight_dict[expert_prefix + "up_proj.weight"] = up_weight;
      weight_dict[expert_prefix + "down_proj.weight"] = down_weight;
    }
  }

  // Gate weights (router)
  auto gate_weight = test::seeded_tensor("fused_moe_all2all_tests.gate.weight",
                                         {num_experts, hidden_size},
                                         torch::kBFloat16,
                                         device);
  weight_dict["gate.weight"] = gate_weight;

  return weight_dict;
}

// Child process test function for basic All2All test
int32_t run_all2all_basic_test_child(All2AllTestParams params) {
  try {
    // 0. Set expert_parallel_degree to enable DeepEP
    // This is required for All2All path to work
    EPLBConfig::get_instance().expert_parallel_degree(2);

    // 1. Check devices
    int32_t dev_count = xllm::Platform::device_count();
    if (dev_count < params.world_size) {
      LOG(WARNING) << "Rank " << params.rank
                   << ": Insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }
    params.device_index = params.rank % dev_count;

    // 2. Set device
    xllm::Device xllm_device(params.device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();

    // 3. Create ProcessGroup
    auto process_group = create_test_process_group(
        params.rank, params.world_size, params.port, params.host, device);

    CHECK(process_group) << "Rank " << params.rank
                         << ": Failed to create ProcessGroup";

    // 4. Create ParallelArgs with EP mode
    ParallelArgs parallel_args(params.rank, params.world_size, process_group);
    parallel_args.moe_ep_group_ = process_group;
    parallel_args.ep_size_ = params.world_size;
    parallel_args.moe_tp_group_ = process_group;

    // 5. Create tensor options
    // Note: smoothquant mode still uses BF16 for input/output, with int8 for
    // internal quantized computation. Using float32 would cause CNNL GroupGemm
    // to fail with "Data type mismatch" error.
    auto options = torch::TensorOptions()
                       .dtype(torch::kBFloat16)
                       .device(device)
                       .requires_grad(false);

    // 6. Create model args and quant args
    ::xllm::test::DeepEpTestScope deep_ep_scope;
    ModelArgs model_args = create_model_args(params);
    QuantArgs quant_args = create_quant_args(params);
    ModelStreamRegistry stream_registry(device);

    // 7. Create FusedMoE
    FusedMoE fused_moe(FusedMoEImpl(
        model_args,
        FusedMoEArgs{.is_gated = true},
        quant_args,
        parallel_args,
        options,
        stream_registry.get(ExecutionStreamRole::COMMUNICATION),
        stream_registry.get(ExecutionStreamRole::AUXILIARY_COMPUTE)));

    // 8. Create and load weights using seeded tensors
    auto weight_dict = create_all2all_test_weights(params.num_experts,
                                                   params.hidden_size,
                                                   params.intermediate_size,
                                                   params.is_smoothquant,
                                                   params.moe_weight_bits,
                                                   params.group_size,
                                                   params.world_size,
                                                   device);
    StateDict state_dict(weight_dict);
    fused_moe->load_state_dict(state_dict);

    LOG(INFO) << "Rank " << params.rank << ": FusedMoE created and loaded";

    // 9. Create input tensor using seeded tensor for determinism
    int64_t num_tokens = params.batch_size * params.seq_len;
    std::string seed_prefix = params.is_smoothquant
                                  ? "fused_moe_all2all_tests.smoothquant"
                                  : "fused_moe_all2all_tests.basic";
    auto hidden_states = test::seeded_tensor(seed_prefix + ".hidden_states",
                                             {num_tokens, params.hidden_size},
                                             torch::kBFloat16,
                                             device);

    // 10. Run forward with All2All enabled
    auto output =
        fused_moe->forward_experts(hidden_states,
                                   /*enable_all2all_communication=*/true);

    // 11. Verify output
    xllm_device.synchronize_default_stream();

    CHECK_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
    CHECK_EQ(output.size(0), num_tokens) << "Token count should match";
    CHECK_EQ(output.size(1), params.hidden_size) << "Hidden size should match";

    // Compute and log output stats for setting expected values
    auto flat_output = output.flatten().to(torch::kFloat32).cpu();
    double actual_min = torch::min(flat_output).item<double>();
    double actual_max = torch::max(flat_output).item<double>();
    double actual_sum = torch::sum(flat_output).item<double>();

    LOG(INFO) << "Rank " << params.rank << ": Output stats - "
              << "min=" << actual_min << ", max=" << actual_max
              << ", sum=" << actual_sum;

    // Basic sanity check (will be replaced with precise validation later)
    CHECK_NE(actual_sum, 0.0) << "Output should not be all zeros";

    if (params.perform_precise_validation) {
      test::expect_tensor_stats(output,
                                params.expected_min,
                                params.expected_max,
                                params.expected_sum);
      LOG(INFO) << "Rank " << params.rank
                << ": All2All test passed with precise validation.";
    } else {
      LOG(INFO) << "Rank " << params.rank
                << ": All2All test passed (basic validation only).";
    }

    return 0;

  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << params.rank << ": Exception: " << e.what();
    return 1;
  }
}

// Child process test function for SmoothQuant All2All test
int32_t run_all2all_smoothquant_test_child(All2AllTestParams params) {
  params.is_smoothquant = true;
  return run_all2all_basic_test_child(params);
}

int32_t run_all2all_selected_route_test_child(All2AllTestParams params) {
  try {
    EPLBConfig::get_instance().expert_parallel_degree(2);

    int32_t dev_count = xllm::Platform::device_count();
    if (dev_count < params.world_size) {
      LOG(WARNING) << "Rank " << params.rank
                   << ": Insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }
    params.device_index = params.rank % dev_count;

    xllm::Device xllm_device(params.device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();

    auto process_group = create_test_process_group(
        params.rank, params.world_size, params.port, params.host, device);
    CHECK(process_group) << "Rank " << params.rank
                         << ": Failed to create ProcessGroup";

    ParallelArgs parallel_args(params.rank, params.world_size, process_group);
    parallel_args.moe_ep_group_ = process_group;
    parallel_args.ep_size_ = params.world_size;
    parallel_args.moe_tp_group_ = process_group;

    auto options = torch::TensorOptions()
                       .dtype(torch::kBFloat16)
                       .device(device)
                       .requires_grad(false);

    ::xllm::test::DeepEpTestScope deep_ep_scope;
    ModelArgs model_args = create_model_args(params);
    QuantArgs quant_args = create_quant_args(params);
    ModelStreamRegistry stream_registry(device);
    FusedMoE fused_moe(FusedMoEImpl(
        model_args,
        FusedMoEArgs{.is_gated = true},
        quant_args,
        parallel_args,
        options,
        stream_registry.get(ExecutionStreamRole::COMMUNICATION),
        stream_registry.get(ExecutionStreamRole::AUXILIARY_COMPUTE)));

    auto weight_dict = create_all2all_test_weights(params.num_experts,
                                                   params.hidden_size,
                                                   params.intermediate_size,
                                                   params.is_smoothquant,
                                                   params.moe_weight_bits,
                                                   params.group_size,
                                                   params.world_size,
                                                   device);
    StateDict state_dict(weight_dict);
    fused_moe->load_state_dict(state_dict);

    int64_t num_tokens = params.batch_size * params.seq_len;
    auto hidden_states =
        test::seeded_tensor("fused_moe_all2all_tests.selected.hidden_states",
                            {num_tokens, params.hidden_size},
                            torch::kBFloat16,
                            device);
    FusedMoEImpl::RouteInfo route_info = fused_moe->prep_route(hidden_states);
    route_info.expert_id = route_info.expert_id.to(torch::kLong);
    torch::Tensor expected =
        fused_moe->forward_experts(hidden_states,
                                   /*enable_all2all_communication=*/true);
    torch::Tensor actual =
        fused_moe->forward_experts(hidden_states,
                                   /*enable_all2all_communication=*/true,
                                   route_info);

    xllm_device.synchronize_default_stream();
    CHECK(torch::allclose(actual, expected, /*rtol=*/1e-3, /*atol=*/1e-4))
        << "Selected MoE route all2all output mismatch";
    return 0;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << params.rank << ": Exception: " << e.what();
    return 1;
  }
}

std::vector<::xllm::test::ForkedRankSession::Case> all2all_cases() {
  return {
      {"BasicAll2AllFlow",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         All2AllTestParams params{};
         params.device_index = -1;
         params.hidden_size = 512;
         params.intermediate_size = 256;
         params.num_experts = 4;
         params.top_k = 2;
         params.batch_size = 2;
         params.seq_len = 4;
         params.is_smoothquant = false;
         params.rank = rank;
         params.world_size = 2;
         params.port = port;
         params.host = "127.0.0.1";
         return run_all2all_basic_test_child(params);
       }},
      {"SmoothQuantAll2AllFlow",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         All2AllTestParams params{};
         params.device_index = -1;
         params.hidden_size = 512;
         params.intermediate_size = 256;
         params.num_experts = 4;
         params.top_k = 2;
         params.batch_size = 2;
         params.seq_len = 4;
         params.is_smoothquant = true;
         params.rank = rank;
         params.world_size = 2;
         params.port = port;
         params.host = "127.0.0.1";
         return run_all2all_smoothquant_test_child(params);
       }},
      {"BasicAll2AllPreciseValidation",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         All2AllTestParams params = ([]() {
           All2AllTestParams params{};
           params.device_index = -1;
           params.hidden_size = 512;
           params.intermediate_size = 256;
           params.num_experts = 4;
           params.top_k = 2;
           params.batch_size = 2;
           params.seq_len = 4;
           params.is_smoothquant = false;
           params.expected_min = 835584.0;
           params.expected_max = 1269760.0;
           params.expected_sum = 4209990000.0;
           params.perform_precise_validation = true;
           return params;
         })();
         params.rank = rank;
         params.world_size = 2;
         params.port = port;
         params.host = "127.0.0.1";
         return run_all2all_basic_test_child(params);
       }},
      {"SmoothQuantAll2AllPreciseValidation",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         All2AllTestParams params = ([]() {
           All2AllTestParams params{};
           params.device_index = -1;
           params.hidden_size = 512;
           params.intermediate_size = 256;
           params.num_experts = 4;
           params.top_k = 2;
           params.batch_size = 2;
           params.seq_len = 4;
           params.is_smoothquant = true;
           // Note: min=0 is valid for SmoothQuant due to quantization effects
           params.expected_min = 0.0;
           params.expected_max = 0.104004;
           params.expected_sum = 3.88289;
           params.perform_precise_validation = true;
           return params;
         })();
         params.rank = rank;
         params.world_size = 2;
         params.port = port;
         params.host = "127.0.0.1";
         return run_all2all_smoothquant_test_child(params);
       }},
      {"W4A8All2AllSmoke",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         All2AllTestParams params = ([]() {
           All2AllTestParams params{};
           params.device_index = -1;
           params.hidden_size = 512;
           params.intermediate_size = 256;
           params.num_experts = 4;
           params.top_k = 2;
           params.batch_size = 2;
           params.seq_len = 4;
           params.is_smoothquant = true;
           params.moe_weight_bits = 4;
           params.group_size = 128;
           params.perform_precise_validation = false;
           return params;
         })();
         params.rank = rank;
         params.world_size = 2;
         params.port = port;
         params.host = "127.0.0.1";
         return run_all2all_smoothquant_test_child(params);
       }},
      {"SelectedRouteMatchesGateRouteAll2All",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         All2AllTestParams params = ([]() {
           All2AllTestParams params{};
           params.device_index = -1;
           params.hidden_size = 512;
           params.intermediate_size = 256;
           params.num_experts = 4;
           params.top_k = 2;
           params.batch_size = 2;
           params.seq_len = 4;
           params.is_smoothquant = false;
           return params;
         })();
         params.rank = rank;
         params.world_size = 2;
         params.port = port;
         params.host = "127.0.0.1";
         return run_all2all_selected_route_test_child(params);
       }},
  };
}
}  // namespace

class FusedMoEAll2AllMultiDeviceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0);
    session_ = std::make_unique<::xllm::test::ForkedRankSession>(
        2, port, all2all_cases(), [] { rank_process_group.reset(); });
  }
  static void TearDownTestSuite() {
    if (session_) {
      ::xllm::test::expect_rank_result(session_->close());
      session_.reset();
    }
  }
  void run_case(const std::string& name) {
    ::xllm::test::expect_rank_result(session_->run(name));
  }

 private:
  inline static std::unique_ptr<::xllm::test::ForkedRankSession> session_;
};

TEST_F(FusedMoEAll2AllMultiDeviceTest, BasicAll2AllFlow) {
  run_case("BasicAll2AllFlow");
}

TEST_F(FusedMoEAll2AllMultiDeviceTest, SmoothQuantAll2AllFlow) {
  run_case("SmoothQuantAll2AllFlow");
}

TEST_F(FusedMoEAll2AllMultiDeviceTest, BasicAll2AllPreciseValidation) {
  run_case("BasicAll2AllPreciseValidation");
}

TEST_F(FusedMoEAll2AllMultiDeviceTest, SmoothQuantAll2AllPreciseValidation) {
  run_case("SmoothQuantAll2AllPreciseValidation");
}

TEST_F(FusedMoEAll2AllMultiDeviceTest, W4A8All2AllSmoke) {
  run_case("W4A8All2AllSmoke");
}

TEST_F(FusedMoEAll2AllMultiDeviceTest, SelectedRouteMatchesGateRouteAll2All) {
  run_case("SelectedRouteMatchesGateRouteAll2All");
}

}  // namespace test
}  // namespace layer
}  // namespace xllm
