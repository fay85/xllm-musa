/* Copyright 2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cerrno>
#include <unordered_map>
#include <vector>

#include "framework/parallel_state/process_group.h"
#include "layers/common/attention_metadata_builder.h"
#include "layers/mlu/glm5_next/glm5_next_kda.h"
#include "layers/mlu/glm5_next/glm5_next_pcp_context.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "tests/core/layers/mlu/glm5_next_kpool_test_utils.h"
#include "util/net.h"

namespace xllm::layer {
namespace {

constexpr int32_t kWorldSize = 2;
constexpr int32_t kSkip = 77;

using RankCheck = ::testing::AssertionResult (*)(Device&,
                                                 ProcessGroup*,
                                                 int32_t);

::testing::AssertionResult check_kda_state(Device& device,
                                           ProcessGroup* cp_group,
                                           int32_t rank) {
  torch::InferenceMode inference_mode;
  ProcessGroup single_rank(/*rank=*/0, /*world_size=*/1, device.unwrap());
  ParallelArgs parallel(rank,
                        kWorldSize,
                        /*dp_size=*/1,
                        /*cp_size=*/kWorldSize,
                        cp_group,
                        /*ep_size=*/kWorldSize);
  parallel.tp_group_ = &single_rank;
  const auto options =
      torch::TensorOptions().device(device.unwrap()).dtype(torch::kBFloat16);
  const auto ints = options.dtype(torch::kInt32);
  // Generate on CPU so both ranks receive identical parameters and inputs.
  torch::manual_seed(/*seed=*/20261012);
  const auto random_tensor = [&](const std::vector<int64_t>& shape,
                                 torch::ScalarType dtype) {
    return torch::randn(shape, torch::TensorOptions().dtype(dtype))
        .to(device.unwrap());
  };
  ModelArgs args;
  args.model_type("glm5_next")
      .hidden_size(128)
      .linear_num_key_heads(8)
      .linear_num_value_heads(8)
      .linear_key_head_dim(128)
      .linear_value_head_dim(128)
      .linear_conv_kernel_dim(4)
      .rms_norm_eps(1e-6);
  Glm5NextKDA layer(args, QuantArgs(), parallel, options);
  for (auto& parameter : layer->named_parameters()) {
    parameter.value().copy_(random_tensor(parameter.value().sizes().vec(),
                                          parameter.value().scalar_type()) *
                            0.03f);
  }
  const int32_t chunk_size = static_cast<int32_t>(
      kernel::mlu::kda_prefill_chunk_size(/*num_heads=*/8,
                                          /*use_qk_l2norm=*/true));
  const std::vector<int32_t> lengths{chunk_size + 1, chunk_size + 2, 3};
  auto context = glm5_next_pcp::build_context(
      lengths, rank, cp_group, chunk_size, device.unwrap());
  context.local_metadata.q_cu_seq_lens = context.local_cu_seqlens;
  std::vector<int32_t> conv_batch(2048, -1);
  std::vector<int32_t> conv_offsets(2048, -1);
  int32_t conv_blocks = 0;
  for (size_t request = 0; request < lengths.size(); ++request) {
    for (int32_t offset = 0; offset < context.local_query_lengths[request];
         offset += 64) {
      conv_batch[conv_blocks] = static_cast<int32_t>(request);
      conv_offsets[conv_blocks] = offset / 64;
      ++conv_blocks;
    }
  }
  context.local_metadata.batch = torch::tensor(conv_batch, ints);
  context.local_metadata.token_block_offset = torch::tensor(conv_offsets, ints);
  context.local_metadata.tot = conv_blocks;
  const torch::Tensor targets = torch::tensor({2, 3, 4}, ints);
  for (const int64_t stride : {1, 4}) {
    const torch::Tensor conv = random_tensor(
        {5, stride + 2, 3 * (8 / kWorldSize) * 128}, torch::kBFloat16);
    const torch::Tensor ssm =
        random_tensor({5 * stride, 8 / kWorldSize, 128, 128}, torch::kFloat32) *
        0.03f;
    KVCache copied_cache(
        LinearAttentionKVCacheTensors{conv.clone(), ssm.clone()});
    KVCache direct_cache(
        LinearAttentionKVCacheTensors{conv.clone(), ssm.clone()});
    for (const int64_t target : {2, 3}) {
      copied_cache.get_conv_cache()[target].copy_(conv[1]);
      copied_cache.get_ssm_cache()
          .narrow(/*dim=*/0, target * stride, stride)
          .copy_(ssm.narrow(/*dim=*/0, stride, stride));
    }
    ModelInputParams direct;
    direct.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
    direct.embedding.linear_state_ids = {2, 3, 4};
    direct.embedding.linear_state_indices = targets;
    direct.embedding.linear_state_read_ids = {1, 1, 4};
    direct.embedding.linear_state_read_indices = torch::tensor({1, 1, 4}, ints);
    ModelInputParams copied = direct;
    copied.embedding.linear_state_read_ids = copied.embedding.linear_state_ids;
    copied.embedding.linear_state_read_indices = targets;
    AttentionMetadata metadata{};
    metadata.is_chunked_prefill = true;
    metadata.q_cu_seq_lens = torch::tensor(
        {0, lengths[0], lengths[0] + lengths[1], context.geometry.total_tokens},
        ints);
    AttentionMetadataBuilder::build_linear_prefill(metadata, chunk_size);
    metadata.has_initial_states =
        torch::tensor({true, true, false}, options.dtype(torch::kBool));
    for (int32_t step = 0; step < 2; ++step) {
      const torch::Tensor hidden =
          random_tensor({context.geometry.total_tokens, 128}, torch::kBFloat16);
      const torch::Tensor local_hidden =
          glm5_next_pcp::shard_rows(hidden, context);
      const torch::Tensor expected = layer->forward(
          local_hidden, metadata, copied_cache, copied, &context);
      AttentionMetadata direct_metadata = metadata;
      if (step == 0) {
        direct_metadata.linear_state_read_indices =
            direct.embedding.linear_state_read_indices;
      }
      const torch::Tensor actual = layer->forward(
          local_hidden, direct_metadata, direct_cache, direct, &context);
      device.synchronize_default_stream();
      const auto failure = [&](const char* item) {
        return ::testing::AssertionFailure()
               << "KDA " << item << " mode=pcp rank=" << rank
               << " stride=" << stride << " step=" << step;
      };
      if (!torch::equal(actual, expected)) {
        return failure("output");
      }
      if (!torch::equal(direct_cache.get_conv_cache()
                            .index_select(/*dim=*/0, targets)
                            .narrow(/*dim=*/1, /*start=*/0, /*length=*/3),
                        copied_cache.get_conv_cache()
                            .index_select(/*dim=*/0, targets)
                            .narrow(/*dim=*/1, /*start=*/0, /*length=*/3))) {
        return failure("destination conv state");
      }
      if (!torch::equal(direct_cache.get_ssm_cache().index_select(
                            /*dim=*/0, targets * stride),
                        copied_cache.get_ssm_cache().index_select(
                            /*dim=*/0, targets * stride))) {
        return failure("destination ssm state");
      }
      if (!torch::equal(direct_cache.get_conv_cache()[1], conv[1])) {
        return failure("shared source conv state");
      }
      if (!torch::equal(
              direct_cache.get_ssm_cache().narrow(/*dim=*/0, stride, stride),
              ssm.narrow(/*dim=*/0, stride, stride))) {
        return failure("shared source ssm state");
      }
      // Resume from the destinations on the next distributed chunk.
      direct.embedding.linear_state_read_ids =
          direct.embedding.linear_state_ids;
      direct.embedding.linear_state_read_indices = targets;
      metadata.has_initial_states.fill_(true);
    }
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult check_kda_head_tp(Device& device,
                                             ProcessGroup* cp_group,
                                             int32_t rank) {
  torch::InferenceMode guard;
  ProcessGroup single(/*rank=*/0, /*world_size=*/1, device.unwrap());
  ParallelArgs reference_parallel(/*rank=*/0, /*world_size=*/1, &single);
  reference_parallel.tp_group_ = &single;
  ParallelArgs parallel(rank,
                        /*world_size=*/kWorldSize,
                        /*dp_size=*/1,
                        /*cp_size=*/kWorldSize,
                        cp_group,
                        /*ep_size=*/kWorldSize);
  parallel.tp_group_ = &single;
  parallel.cp_group_ = cp_group;
  ModelArgs args;
  args.model_type(/*model_type=*/"glm5_next")
      .hidden_size(/*hidden_size=*/128)
      .linear_num_key_heads(/*linear_num_key_heads=*/8)
      .linear_num_value_heads(/*linear_num_value_heads=*/8)
      .linear_key_head_dim(/*linear_key_head_dim=*/128)
      .linear_value_head_dim(/*linear_value_head_dim=*/128)
      .linear_conv_kernel_dim(/*linear_conv_kernel_dim=*/4)
      .rms_norm_eps(/*rms_norm_eps=*/1e-6);
  const auto options =
      torch::TensorOptions().device(device.unwrap()).dtype(torch::kBFloat16);
  const auto ints = options.dtype(torch::kInt32);
  torch::manual_seed(/*seed=*/20261029);
  const auto random = [&](const std::vector<int64_t>& shape) {
    return (torch::randn(shape, torch::kFloat32) * 0.03f).to(options);
  };
  std::unordered_map<std::string, torch::Tensor> weights;
  for (const std::string name : {"q_proj", "k_proj", "v_proj"}) {
    weights.emplace(name + ".weight", random(/*shape=*/{1024, 128}));
  }
  weights.emplace(/*key=*/"b_proj.weight", random(/*shape=*/{8, 128}));
  for (const std::string name : {"f_a_proj", "g_a_proj"}) {
    weights.emplace(name + ".weight", random(/*shape=*/{128, 128}));
  }
  for (const std::string name : {"f_b_proj", "g_b_proj"}) {
    weights.emplace(name + ".weight", random(/*shape=*/{1024, 128}));
  }
  for (const std::string name : {"q_conv1d", "k_conv1d", "v_conv1d"}) {
    weights.emplace(name + ".weight", random(/*shape=*/{1024, 1, 4}));
  }
  weights.emplace(/*key=*/"o_proj.weight", random(/*shape=*/{128, 1024}));
  weights.emplace(/*key=*/"o_norm.weight",
                  torch::ones(/*size=*/{128}, options));
  weights.emplace(/*key=*/"A_log", random(/*shape=*/{8}).to(torch::kFloat32));
  weights.emplace(/*key=*/"dt_bias",
                  random(/*shape=*/{1024}).to(torch::kFloat32));
  const StateDict state_dict(std::move(weights));
  Glm5NextKDA reference(args, QuantArgs(), reference_parallel, options);
  Glm5NextKDA sharded(args, QuantArgs(), parallel, options);
  reference->load_state_dict(state_dict);
  sharded->load_state_dict(state_dict);
  reference->verify_loaded_weights();
  sharded->verify_loaded_weights();
  constexpr int64_t kLocalHeads = 8 / kWorldSize;
  constexpr int64_t kStride = 4;
  KVCache reference_cache(LinearAttentionKVCacheTensors{
      torch::zeros(/*size=*/{3, 6, 3 * 8 * 128}, options),
      torch::zeros(/*size=*/{3 * kStride, 8, 128, 128},
                   options.dtype(torch::kFloat32))});
  KVCache sharded_cache(LinearAttentionKVCacheTensors{
      torch::zeros(/*size=*/{3, 6, 3 * kLocalHeads * 128}, options),
      torch::zeros(/*size=*/{3 * kStride, kLocalHeads, 128, 128},
                   options.dtype(torch::kFloat32))});
  ModelInputParams params;
  params.embedding.linear_state_ids = {1, 2};
  params.embedding.linear_state_indices = torch::tensor(/*data=*/{1, 2}, ints);
  // Ragged packed requests, continuation, decode, then speculative
  // verification.
  for (int32_t step = 0; step < 4; ++step) {
    const bool prefill = step < 2;
    const bool verify = step == 3;
    const std::vector<int32_t> lengths =
        prefill ? std::vector<int32_t>{129, 67}
                : std::vector<int32_t>{verify ? 4 : 1, verify ? 4 : 1};
    const int32_t tokens = lengths[0] + lengths[1];
    AttentionMetadata metadata{};
    metadata.is_chunked_prefill = prefill || verify;
    metadata.is_spec_verify = verify;
    metadata.max_query_len = lengths[0];
    metadata.q_cu_seq_lens =
        torch::tensor(/*data=*/{0, lengths[0], tokens}, ints);
    metadata.has_initial_states =
        torch::full(/*size=*/{2}, step > 0, options.dtype(torch::kBool));
    AttentionMetadataBuilder::build_linear_prefill(metadata, /*block_size=*/64);
    params.meta.batch_forward_type = prefill || verify
                                         ? BatchForwardType::CHUNKED_PREFILL
                                         : BatchForwardType::DECODE;
    params.is_spec_verify = verify;
    params.num_accepted_tokens = torch::ones(/*size=*/{2}, ints);
    const torch::Tensor hidden = random(/*shape=*/{tokens, 128}) / 0.03f;
    const torch::Tensor expected =
        reference->forward(hidden, metadata, reference_cache, params);
    torch::Tensor actual;
    if (prefill) {
      const auto context = glm5_next_pcp::build_context(
          lengths, rank, cp_group, /*chunk_size=*/64, device.unwrap());
      const torch::Tensor local = glm5_next_pcp::shard_rows(hidden, context);
      actual = glm5_next_pcp::gather_restore(
          sharded->forward(local, metadata, sharded_cache, params, &context),
          context);
    } else {
      actual = sharded->forward(hidden, metadata, sharded_cache, params);
    }
    const auto failure = [&](const char* item) {
      return ::testing::AssertionFailure()
             << item << " rank=" << rank << " step=" << step;
    };
    if (!torch::allclose(actual.to(torch::kFloat32),
                         expected.to(torch::kFloat32),
                         /*rtol=*/0.03,
                         /*atol=*/0.003)) {
      return failure(/*item=*/"KDA head-TP output");
    }
    const torch::Tensor expected_ssm = reference_cache.get_ssm_cache().narrow(
        /*dim=*/1, rank * kLocalHeads, kLocalHeads);
    if (!torch::allclose(sharded_cache.get_ssm_cache(),
                         expected_ssm,
                         /*rtol=*/0.02,
                         /*atol=*/2e-5)) {
      return failure(/*item=*/"KDA head-TP recurrent state");
    }
    std::vector<torch::Tensor> conv_parts;
    conv_parts.reserve(/*new_cap=*/3);
    for (const torch::Tensor& part :
         reference_cache.get_conv_cache().split(/*split_size=*/8 * 128,
                                                /*dim=*/2)) {
      conv_parts.emplace_back(
          part.narrow(/*dim=*/2, rank * kLocalHeads * 128, kLocalHeads * 128));
    }
    if (!torch::allclose(sharded_cache.get_conv_cache().to(torch::kFloat32),
                         torch::cat(conv_parts, /*dim=*/2).to(torch::kFloat32),
                         /*rtol=*/0.02,
                         /*atol=*/2e-5)) {
      return failure(/*item=*/"KDA head-TP convolution state");
    }
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult check_kpool_state(Device& device,
                                             ProcessGroup* cp_group,
                                             int32_t rank) {
  return test::check_kpool_state(device.unwrap(), cp_group, rank);
}

int32_t run_rank(int32_t rank, int32_t port, RankCheck check) {
  ::alarm(240);
  if (Platform::device_count() < kWorldSize) {
    return kSkip;
  }
  Device device(rank);
  device.set_device();
  auto cp_group = create_process_group(rank,
                                       kWorldSize,
                                       kWorldSize,
                                       port,
                                       /*trans=*/false,
                                       "127.0.0.1",
                                       "state_pcp_test",
                                       device.unwrap());
  const auto result = check(device, cp_group.get(), rank);
  if (!result) {
    LOG(ERROR) << result.message();
    return 1;
  }
  return 0;
}

void check_two_ranks(RankCheck check) {
  const int32_t port = net::get_local_free_port();
  ASSERT_GT(port, 0);
  // This standalone binary forks before the parent initializes MLU.
  std::vector<pid_t> children;
  children.reserve(kWorldSize);
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      ::_exit(run_rank(rank, port, check));
    }
    children.emplace_back(child);
  }
  int32_t skipped = 0;
  for (const pid_t child : children) {
    int32_t status = 0;
    pid_t waited;
    do {
      waited = ::waitpid(child, &status, /*options=*/0);
    } while (waited < 0 && errno == EINTR);
    EXPECT_EQ(waited, child);
    EXPECT_TRUE(WIFEXITED(status));
    if (WIFEXITED(status) && WEXITSTATUS(status) == kSkip) {
      ++skipped;
      continue;
    }
    EXPECT_EQ(WIFEXITED(status) ? WEXITSTATUS(status) : -1, 0);
  }
  if (skipped == kWorldSize) {
    GTEST_SKIP() << "Requires two visible MLU devices.";
  }
  EXPECT_EQ(skipped, 0);
}

TEST(Glm5NextStatePcpTest, KdaMatchesRestoreCopyAndPreservesSharedSource) {
  check_two_ranks(check_kda_state);
}

TEST(Glm5NextStatePcpTest,
     HeadTpMatchesSingleRankThroughPrefillDecodeAndVerify) {
  check_two_ranks(check_kda_head_tp);
}

TEST(Glm5NextStatePcpTest, KpoolMatchesCopyThroughVerifyAndDecode) {
  check_two_ranks(check_kpool_state);
}

}  // namespace
}  // namespace xllm::layer
