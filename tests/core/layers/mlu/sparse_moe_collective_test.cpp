/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <framework/core/device.h>
#include <framework/graphs/MLUGraph.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/config/eplb_config.h"
#include "framework/parallel_state/process_group.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/glm5_next/glm5_next_pcp_context.h"
#include "layers/mlu/sparse_moe_block.h"
#include "platform/device.h"
#include "platform/model_stream_registry.h"
#include "platform/platform.h"
#include "tests/core/forked_rank_session.h"
#include "tests/core/layers/mlu/deep_ep_test_scope.h"
#include "util/net.h"

namespace xllm::layer {
namespace {

constexpr int32_t kExitCodeSkip = 77;

ModelArgs make_model_args(bool with_shared) {
  ModelArgs args;
  args.model_type() = "glm5_next";
  args.hidden_size() = 256;
  args.moe_intermediate_size() = 256;
  args.n_routed_experts() = 4;
  args.num_experts_per_tok() = 2;
  args.n_shared_experts() = with_shared ? 1 : 0;
  args.n_group() = 1;
  args.topk_group() = 1;
  args.routed_scaling_factor() = 1.0f;
  args.norm_topk_prob() = true;
  args.hidden_act() = "silu";
  args.scoring_func() = "softmax";
  args.topk_method() = "greedy";
  args.swiglu_limit() = 10.0f;
  return args;
}

torch::Tensor make_values(int64_t rows,
                          int64_t columns,
                          float phase,
                          const torch::TensorOptions& options) {
  // Construct on CPU so every rank receives identical full weights and inputs.
  // Positive values avoid cancellation amplifying BF16 shard-rounding error
  // when TP partial sums are compared with a single unsharded GEMM.
  return (torch::arange(rows * columns, torch::kFloat32) * 0.17f + phase)
      .sin()
      .add(1.5f)
      .reshape({rows, columns})
      .mul(0.02f)
      .to(options);
}

StateDict make_weights(const ModelArgs& args,
                       const torch::TensorOptions& options) {
  std::unordered_map<std::string, torch::Tensor> weights;
  const int64_t hidden = args.hidden_size();
  const int64_t intermediate = args.moe_intermediate_size();
  const auto add_expert = [&](const std::string& prefix, float phase) {
    weights.emplace(prefix + "gate_proj.weight",
                    make_values(intermediate, hidden, phase, options));
    weights.emplace(prefix + "up_proj.weight",
                    make_values(intermediate, hidden, phase + 0.7f, options));
    weights.emplace(prefix + "down_proj.weight",
                    make_values(hidden, intermediate, phase + 1.3f, options));
  };
  for (int32_t expert = 0; expert < args.n_routed_experts(); ++expert) {
    add_expert("experts." + std::to_string(expert) + ".",
               static_cast<float>(expert));
  }
  if (args.n_shared_experts() > 0) {
    add_expert("shared_experts.", 5.0f);
  }
  weights.emplace("gate.weight",
                  make_values(args.n_routed_experts(), hidden, 0.0f, options));
  return StateDict(std::move(weights));
}

ParallelArgs make_parallel_args(int32_t rank,
                                int32_t world_size,
                                bool use_ep,
                                ProcessGroup* collective,
                                ProcessGroup* single_rank) {
  ParallelArgs args(rank, world_size, collective);
  args.tp_group_ = collective;
  args.single_rank_group_ = single_rank;
  args.moe_ep_group_ = use_ep ? collective : single_rank;
  args.moe_tp_group_ = use_ep ? single_rank : collective;
  args.ep_size_ = use_ep ? world_size : 1;
  return args;
}

void check_output(const torch::Tensor& actual, const torch::Tensor& expected) {
  CHECK_EQ(actual.sizes(), expected.sizes());
  CHECK(torch::isfinite(actual).all().item<bool>());
  if (expected.numel() == 0) {
    return;
  }
  CHECK_GT(expected.abs().max().item<float>(), 1e-4f);
  CHECK(torch::allclose(actual, expected, /*rtol=*/0.02, /*atol=*/2e-4))
      << "Selected MoE differs from the unsharded reference; maximum error: "
      << (actual - expected).abs().max().item<float>();
}

template <typename Forward, typename Expected, typename Update>
void check_selected(Device& device,
                    bool graph_replay,
                    const Forward& selected,
                    const Expected& expected,
                    const Update& update_inputs) {
  // Warm the real communicator and kernels before capture on a nondefault
  // stream.
  for (int32_t iteration = 0; iteration < 3; ++iteration) {
    check_output(selected(), expected());
  }
  torch_mlu::synchronize();
  if (!graph_replay) {
    return;
  }

  auto capture_stream = device.get_stream_from_pool();
  torch_mlu::MLUGraph graph;
  torch::Tensor captured;
  {
    auto guard = capture_stream->set_stream_guard();
    graph.capture_begin();
    captured = selected();
    graph.capture_end();
  }
  for (int32_t iteration = 0; iteration < 3; ++iteration) {
    update_inputs(iteration);
    graph.replay();
    check_output(captured, expected());
  }
  torch_mlu::synchronize();
}

thread_local std::unique_ptr<ProcessGroup> rank_collective;

int32_t run_rank(int32_t rank,
                 int32_t world_size,
                 int32_t port,
                 bool use_ep,
                 bool with_shared,
                 bool graph_replay,
                 bool use_cp = false,
                 bool use_deep_ep = false,
                 bool use_dp = false) {
  if (Platform::device_count() < world_size) {
    return kExitCodeSkip;
  }
  torch::InferenceMode inference_mode;
  Device device(rank);
  device.set_device();
  EPLBConfig::get_instance().expert_parallel_degree(use_deep_ep ? 2 : 1);
  if (!rank_collective) {
    rank_collective = create_process_group(rank,
                                           world_size,
                                           world_size,
                                           port,
                                           /*trans=*/false,
                                           "127.0.0.1",
                                           "selected_moe_collective_test",
                                           device.unwrap());
  }
  ProcessGroup* collective = rank_collective.get();
  ProcessGroup single_rank(/*rank=*/0, /*world_size=*/1, device.unwrap());
  ParallelArgs parallel_args =
      make_parallel_args(rank, world_size, use_ep, collective, &single_rank);
  if (use_cp) {
    parallel_args.tp_group_ = &single_rank;
    parallel_args.cp_group_ = collective;
  }
  if (use_dp) {
    parallel_args = ParallelArgs(rank, world_size, world_size, collective);
    parallel_args.tp_group_ = &single_rank;
    parallel_args.single_rank_group_ = &single_rank;
    parallel_args.moe_ep_group_ = collective;
    parallel_args.moe_tp_group_ = &single_rank;
    parallel_args.dp_local_process_group_ = collective;
    parallel_args.ep_size_ = world_size;
  }
  const ParallelArgs reference_args = make_parallel_args(
      /*rank=*/0,
      /*world_size=*/1,
      /*use_ep=*/false,
      &single_rank,
      &single_rank);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  ::xllm::test::DeepEpTestScope deep_ep_scope;
  const ModelArgs model_args = make_model_args(with_shared);
  ModelStreamRegistry streams(device.unwrap());
  SparseMoEBlock block(model_args,
                       QuantArgs(),
                       parallel_args,
                       options,
                       /*use_hash=*/false,
                       streams.get(ExecutionStreamRole::COMMUNICATION),
                       streams.get(ExecutionStreamRole::AUXILIARY_COMPUTE));
  // No sharding or collective in this reference: all experts run on each rank.
  FusedMoE reference(
      model_args,
      FusedMoEArgs{.is_gated = true, .enable_result_reduction = false},
      QuantArgs(),
      reference_args,
      options,
      streams.get(ExecutionStreamRole::COMMUNICATION),
      streams.get(ExecutionStreamRole::AUXILIARY_COMPUTE));
  const StateDict weights = make_weights(model_args, options);
  block->load_state_dict(weights);
  reference->load_state_dict(weights);

  if (use_dp) {
    for (const std::vector<int32_t>& counts :
         {std::vector<int32_t>{3, 5}, {1, 5}, {5, 1}}) {
      // A count of one also represents V4.1's empty-rank collective row.
      torch::Tensor local = make_values(counts[rank],
                                        model_args.hidden_size(),
                                        0.5f + static_cast<float>(rank),
                                        options)
                                .mul(10.0f);
      ModelInputParams input;
      input.parallel.dp_global_token_nums = counts;
      const auto route = block->prep_route(local);
      const torch::Tensor actual = block->forward_selected(
          local, route.reduce_weight, route.expert_id, input);
      const auto reference_route = reference->prep_route(local, std::nullopt);
      torch::Tensor expected = reference->forward_experts(
          local, /*enable_all2all_communication=*/false, reference_route);
      const torch::Tensor shared = reference->forward_shared(local);
      if (shared.defined()) {
        expected.add_(shared);
      }
      check_output(actual, expected);
    }
    torch_mlu::synchronize();
    return 0;
  }

  if (use_deep_ep) {
    for (int64_t row_count : {8, 3, 1, 0}) {
      torch::Tensor hidden =
          make_values(row_count, model_args.hidden_size(), 0.5f, options)
              .mul(10.0f)
              .reshape({1, row_count, model_args.hidden_size()});
      torch::Tensor ids =
          torch::arange(row_count * 2, options.dtype(torch::kInt32))
              .reshape({row_count, 2})
              .remainder(model_args.n_routed_experts());
      torch::Tensor routing_weights =
          torch::tensor({0.25f, 0.75f}, options.dtype(torch::kFloat32))
              .repeat({row_count, 1});
      ModelInputParams input_params;
      input_params.parallel.dp_is_decode = {1};
      const auto selected = [&]() {
        return block->forward_selected(
            hidden, routing_weights, ids, input_params);
      };
      const auto expected = [&]() {
        if (row_count == 0) {
          return torch::empty_like(hidden);
        }
        FusedMoEImpl::RouteInfo route;
        route.expert_id = ids;
        route.reduce_weight = routing_weights;
        torch::Tensor output = reference->forward_experts(
            hidden.reshape({row_count, model_args.hidden_size()}),
            /*enable_all2all_communication=*/false,
            route);
        const torch::Tensor shared = reference->forward_shared(
            hidden.reshape({row_count, model_args.hidden_size()}));
        if (shared.defined()) {
          output.add_(shared);
        }
        return output.reshape(hidden.sizes());
      };
      const auto update_inputs = [&](int32_t /*unused*/) {
        hidden.mul_(/*other=*/0.8f);
        ids.add_(/*other=*/1).remainder_(model_args.n_routed_experts());
        routing_weights.mul_(/*other=*/0.9f);
      };
      check_selected(device, graph_replay, selected, expected, update_inputs);
    }
    return 0;
  }

  if (use_cp) {
    for (const std::vector<int32_t>& lengths :
         {std::vector<int32_t>{128}, {129}, {65, 129}, {1, 0, 7}}) {
      const auto context = glm5_next_pcp::build_context(
          lengths, rank, collective, /*chunk_size=*/64, device.unwrap());
      const torch::Tensor full =
          make_values(context.geometry.total_tokens, 16, 0.5f, options);
      const torch::Tensor local = glm5_next_pcp::shard_rows(full, context);
      CHECK(torch::equal(glm5_next_pcp::gather_restore(local, context), full));
    }
    for (const std::vector<int32_t>& lengths :
         {std::vector<int32_t>{3, 5}, {0, 5}, {5, 0}}) {
      const int64_t offset = rank == 0 ? 0 : lengths[0];
      const torch::Tensor full =
          make_values(
              lengths[0] + lengths[1], model_args.hidden_size(), 0.5f, options)
              .mul(10.0f);
      torch::Tensor local = full.narrow(0, offset, lengths[rank]);
      const torch::Tensor actual =
          block->forward_cp(local, std::nullopt, lengths, collective);
      if (local.size(0) == 0) {
        CHECK_EQ(actual.sizes(), local.sizes());
        continue;
      }
      const auto route = reference->prep_route(local, std::nullopt);
      torch::Tensor expected = reference->forward_experts(
          local, /*enable_all2all_communication=*/false, route);
      const torch::Tensor shared = reference->forward_shared(local);
      if (shared.defined()) {
        expected.add_(shared);
      }
      check_output(actual, expected);
    }
    torch_mlu::synchronize();
    return 0;
  }

  torch::Tensor hidden =
      make_values(/*rows=*/8, model_args.hidden_size(), 0.5f, options)
          .mul(10.0f)
          .reshape({2, 4, model_args.hidden_size()});
  torch::Tensor ids =
      torch::tensor({0, 2}, options.dtype(torch::kInt32)).repeat({8, 1});
  torch::Tensor routing_weights =
      torch::tensor({0.25f, 0.75f}, options.dtype(torch::kFloat32))
          .repeat({8, 1});
  const auto selected = [&]() {
    return block->forward_selected(
        hidden, routing_weights, ids, ModelInputParams());
  };
  const auto expected = [&]() {
    const torch::Tensor rows = hidden.reshape({-1, model_args.hidden_size()});
    FusedMoEImpl::RouteInfo route;
    route.expert_id = ids;
    route.reduce_weight = routing_weights;
    torch::Tensor output = reference->forward_experts(
        rows, /*enable_all2all_communication=*/false, route);
    const torch::Tensor shared = reference->forward_shared(rows);
    if (shared.defined()) {
      output.add_(shared);
    }
    return output.reshape(hidden.sizes());
  };

  const auto update_inputs = [&](int32_t iteration) {
    hidden.mul_(/*other=*/0.8f);
    ids.fill_(iteration % model_args.n_routed_experts());
  };
  check_selected(device, graph_replay, selected, expected, update_inputs);
  return 0;
}

std::vector<::xllm::test::ForkedRankSession::Case> collective_cases() {
  return {
      {"DpEpPreservesUnevenAndDummyRows",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         return run_rank(rank,
                         /*world_size=*/2,
                         port,
                         /*use_ep=*/true,
                         /*with_shared=*/true,
                         /*graph_replay=*/false,
                         /*use_cp=*/false,
                         /*use_deep_ep=*/false,
                         /*use_dp=*/true);
       }},
      {"EpSharedMatchesUnshardedReference",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         return run_rank(rank,
                         /*world_size=*/2,
                         port,
                         /*use_ep=*/true,
                         /*with_shared=*/true,
                         /*graph_replay=*/false);
       }},
      {"DeepEpTpShardPreservesSelectedRows",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         return run_rank(rank,
                         /*world_size=*/2,
                         port,
                         /*use_ep=*/true,
                         /*with_shared=*/false,
                         /*graph_replay=*/false,
                         /*use_cp=*/false,
                         /*use_deep_ep=*/true);
       }},
      {"DeepEpTpShardSharedGraphPreservesRows",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         return run_rank(rank,
                         /*world_size=*/2,
                         port,
                         /*use_ep=*/true,
                         /*with_shared=*/true,
                         /*graph_replay=*/true,
                         /*use_cp=*/false,
                         /*use_deep_ep=*/true);
       }},
      {"CpSharedPreservesRaggedAndEmptyRanks",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         return run_rank(rank,
                         /*world_size=*/2,
                         port,
                         /*use_ep=*/true,
                         /*with_shared=*/true,
                         /*graph_replay=*/false,
                         /*use_cp=*/true);
       }},
      {"CpWithoutSharedPreservesRaggedRanks",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         return run_rank(rank,
                         /*world_size=*/2,
                         port,
                         /*use_ep=*/true,
                         /*with_shared=*/false,
                         /*graph_replay=*/false,
                         /*use_cp=*/true);
       }},
      {"TpSharedMatchesUnshardedReference",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         return run_rank(rank,
                         /*world_size=*/2,
                         port,
                         /*use_ep=*/false,
                         /*with_shared=*/true,
                         /*graph_replay=*/false);
       }},
      {"EpWithoutSharedMatchesUnshardedReference",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         return run_rank(rank,
                         /*world_size=*/2,
                         port,
                         /*use_ep=*/true,
                         /*with_shared=*/false,
                         /*graph_replay=*/false);
       }},
      {"SingleRankWithoutSharedPreservesOutput",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         return run_rank(rank,
                         /*world_size=*/1,
                         port,
                         /*use_ep=*/false,
                         /*with_shared=*/false,
                         /*graph_replay=*/false);
       }},
      {"EpSharedGraphUsesUpdatedInputsAndRoutes",
       [](int32_t rank, int32_t port) {
         torch::manual_seed(0);
         return run_rank(rank,
                         /*world_size=*/2,
                         port,
                         /*use_ep=*/true,
                         /*with_shared=*/true,
                         /*graph_replay=*/true);
       }},
  };
}
}  // namespace

class SparseMoECollectiveTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    failed_ = false;
    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0);
    session_ = std::make_unique<::xllm::test::ForkedRankSession>(
        2, port, collective_cases(), [] { rank_collective.reset(); });
  }
  static void TearDownTestSuite() {
    if (session_) {
      ::xllm::test::expect_rank_result(session_->close());
      session_.reset();
    }
  }
  void run_case(const std::string& name, bool single_rank = false) {
    if (failed_) {
      GTEST_SKIP() << "Earlier rank failure in this suite";
    }
    if (single_rank) {
      const auto closed = session_->close();
      if (closed.status != ::xllm::test::ForkedRankSession::Status::PASSED) {
        failed_ = true;
        ::xllm::test::expect_rank_result(closed);
        return;
      }
      ::xllm::test::ForkedRankSession fresh(
          1, net::get_local_free_port(), collective_cases(), [] {
            rank_collective.reset();
          });
      const auto result = fresh.run(name);
      failed_ =
          result.status == ::xllm::test::ForkedRankSession::Status::FAILED;
      ::xllm::test::expect_rank_result(result);
      ::xllm::test::expect_rank_result(fresh.close());
    } else {
      const auto result = session_->run(name);
      failed_ =
          result.status == ::xllm::test::ForkedRankSession::Status::FAILED;
      ::xllm::test::expect_rank_result(result);
    }
  }

 private:
  inline static std::unique_ptr<::xllm::test::ForkedRankSession> session_;
  inline static bool failed_ = false;
};

TEST_F(SparseMoECollectiveTest, DpEpPreservesUnevenAndDummyRows) {
  run_case("DpEpPreservesUnevenAndDummyRows");
}

TEST_F(SparseMoECollectiveTest, EpSharedMatchesUnshardedReference) {
  run_case("EpSharedMatchesUnshardedReference");
}

TEST_F(SparseMoECollectiveTest, DeepEpTpShardPreservesSelectedRows) {
  run_case("DeepEpTpShardPreservesSelectedRows");
}

TEST_F(SparseMoECollectiveTest, DeepEpTpShardSharedGraphPreservesRows) {
  run_case("DeepEpTpShardSharedGraphPreservesRows");
}

TEST_F(SparseMoECollectiveTest, CpSharedPreservesRaggedAndEmptyRanks) {
  run_case("CpSharedPreservesRaggedAndEmptyRanks");
}

TEST_F(SparseMoECollectiveTest, CpWithoutSharedPreservesRaggedRanks) {
  run_case("CpWithoutSharedPreservesRaggedRanks");
}

TEST_F(SparseMoECollectiveTest, TpSharedMatchesUnshardedReference) {
  run_case("TpSharedMatchesUnshardedReference");
}

TEST_F(SparseMoECollectiveTest, EpWithoutSharedMatchesUnshardedReference) {
  run_case("EpWithoutSharedMatchesUnshardedReference");
}

TEST_F(SparseMoECollectiveTest, EpSharedGraphUsesUpdatedInputsAndRoutes) {
  run_case("EpSharedGraphUsesUpdatedInputsAndRoutes");
}

TEST_F(SparseMoECollectiveTest, SingleRankWithoutSharedPreservesOutput) {
  run_case("SingleRankWithoutSharedPreservesOutput", /*single_rank=*/true);
}

}  // namespace xllm::layer
