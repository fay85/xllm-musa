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

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <exception>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/layers/mlu/deepseek_v4/deepseek_v4_cp_context.h"
#include "core/layers/mlu/deepseek_v41/deepseek_v41_engram.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/util/net.h"

namespace xllm::layer {
namespace {

constexpr int32_t kWorldSize = 2;
constexpr int32_t kSkip = 77;
constexpr std::chrono::seconds kChildTimeout{120};
using Weights = std::unordered_map<std::string, torch::Tensor>;

enum class ChildOutcome : int8_t { PASSED, SKIPPED, FAILED };

void verify_cp_engram(ProcessGroup& group, const torch::Device& device) {
  torch::InferenceMode inference;
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  ModelArgs args;
  args.hidden_size(8)
      .vocab_size(8)
      .hc_mult(2)
      .engram_layer_ids({0})
      .engram_num_embeddings({18})
      .engram_head_dim(32)
      .engram_n_heads(1)
      .engram_max_ngram_size(3)
      .engram_pad_token_id(0)
      .engram_compressed_vocab_size(8)
      .rms_norm_eps(1e-5);
  // Each rank's TP group has one device; EP spans both CP ranks. Choosing TP
  // for the Engram reduction would leave half of these lookup rows missing.
  ProcessGroup single(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs cp_parallel(group.rank(), kWorldSize, &group);
  cp_parallel.cp_size(kWorldSize).ep_size(kWorldSize).kv_split_size(1);
  cp_parallel.tp_group_ = &single;
  cp_parallel.moe_ep_group_ = &group;
  cp_parallel.cp_group_ = &group;
  ParallelArgs reference_parallel(/*rank=*/0, /*world_size=*/1, &single);
  reference_parallel.tp_group_ = &single;
  auto wkv = torch::zeros({24, 64}, torch::kBFloat16);
  auto channels = torch::arange(8, torch::kFloat32);
  // Zero keys fix the gate near 0.5; the two hash columns inject distinct
  // signed value directions. Missing either table shard exceeds the tolerance.
  wkv.slice(0, 16).slice(1, 0, 32).copy_(
      channels.remainder(2).mul(2).sub(1).div(32).unsqueeze(1));
  wkv.slice(0, 16).slice(1, 32, 64).copy_(channels.remainder(4)
                                              .ge(2)
                                              .to(torch::kFloat32)
                                              .mul(2)
                                              .sub(1)
                                              .div(32)
                                              .unsqueeze(1));
  const Weights weights{
      {"embed.weight",
       torch::arange(18)
           .add(0x30)
           .unsqueeze(1)
           .expand({18, 32})
           .contiguous()
           .to(torch::kUInt8)},
      {"embed.scale", torch::full({18, 1}, 127, torch::kUInt8)},
      {"wkv.weight", wkv},
      {"q_weight", torch::ones({2, 8}, torch::kBFloat16)},
      {"k_weight", torch::ones({2, 8}, torch::kBFloat16)},
  };
  const StateDict hash_weights({
      {"token_map", torch::arange(8, torch::kInt64)},
      {"multipliers", torch::tensor({{1, 3, 5}}, torch::kInt64)},
      {"primes", torch::tensor({{7, 11}}, torch::kInt64)},
      {"offsets", torch::tensor({{0, 7}}, torch::kInt64)},
  });
  for (bool cpu_offload : {true, false}) {
    ModelConfig::get_instance().engram_config() =
        cpu_offload ? R"({"cpu_offload":true})" : R"({"cpu_offload":false})";
    DeepseekV41Engram sharded(
        ModelContext(cp_parallel, args, QuantArgs(), options), /*engram_id=*/0);
    DeepseekV41Engram reference(
        ModelContext(reference_parallel, args, QuantArgs(), options),
        /*engram_id=*/0);
    for (auto layer : {sharded, reference}) {
      layer->load_state_dict(StateDict(weights));
      layer->load_hash_state(hash_weights);
      layer->verify_loaded_weights();
    }
    // Neither request lengths nor per-rank row counts are rectangular. All
    // halves are nonempty, matching the real model's CP admission contract.
    for (const std::vector<int32_t>& lengths :
         {std::vector<int32_t>{7, 11}, std::vector<int32_t>{5, 9, 7}}) {
      const int32_t total =
          std::accumulate(lengths.begin(), lengths.end(), int32_t{0});
      ModelInputParams input;
      input.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
      input.meta.num_sequences = static_cast<int32_t>(lengths.size());
      input.meta.actual_num_sequences = input.meta.num_sequences;
      input.attention.host.q_seq_lens = {0};
      input.attention.host.q_seq_lens.reserve(lengths.size() + 1);
      input.embedding.engram_lookback_token_ids.reserve(lengths.size() * 2);
      std::vector<int64_t> position_values;
      position_values.reserve(total);
      for (size_t seq = 0; seq < lengths.size(); ++seq) {
        input.attention.host.q_seq_lens.emplace_back(
            input.attention.host.q_seq_lens.back() + lengths[seq]);
        input.embedding.engram_lookback_token_ids.emplace_back(
            static_cast<int32_t>(seq) + 1);
        input.embedding.engram_lookback_token_ids.emplace_back(
            7 - static_cast<int32_t>(seq));
        for (int32_t row = 0; row < lengths[seq]; ++row) {
          position_values.emplace_back(3 + 2 * static_cast<int64_t>(seq) + row);
        }
      }
      const auto positions =
          torch::tensor(position_values, options.dtype(torch::kInt64));
      const auto cp = mlu_v4_cp::build_cp_context(
          kWorldSize, group.rank(), &group, lengths, positions);
      CHECK(cp.has_value());
      for (const auto& rows : cp->geometry.scatter_rows_by_rank) {
        CHECK(std::find(rows.begin(), rows.end(), cp->geometry.total_tokens) ==
              rows.end())
            << "Fixture must use CP without dummy rows";
      }
      CHECK_NE(cp->geometry.tokens_per_rank[0],
               cp->geometry.tokens_per_rank[1]);
      const auto tokens_cpu =
          (torch::arange(total, torch::kInt64) * 3 + lengths.front())
              .remainder(7)
              .add(1);
      const auto tokens = tokens_cpu.to(device);
      const auto hidden =
          (torch::arange(total * 2 * 8, torch::kFloat32).mul(0.19).cos() * 0.7)
              .reshape({total, 2, 8})
              .to(options);
      // Check the fixture against the small independent 2/3-gram hash formula:
      // global queries touch both physical table shards before local selection.
      std::array<int32_t, kWorldSize> shard_hits{};
      int64_t offset = 0;
      const auto token_values = tokens_cpu.accessor<int64_t, 1>();
      for (size_t seq = 0; seq < lengths.size(); ++seq) {
        int64_t previous = input.embedding.engram_lookback_token_ids[seq * 2];
        int64_t earlier =
            input.embedding.engram_lookback_token_ids[seq * 2 + 1];
        for (int32_t row = 0; row < lengths[seq]; ++row) {
          const int64_t token = token_values[offset++];
          const int64_t hash2 = (token ^ (previous * 3)) % 7;
          const int64_t hash3 =
              (token ^ (previous * 3) ^ (earlier * 5)) % 11 + 7;
          ++shard_hits[hash2 / 9];
          ++shard_hits[hash3 / 9];
          earlier = previous;
          previous = token;
        }
      }
      CHECK_GT(shard_hits[0], 0);
      CHECK_GT(shard_hits[1], 0);
      const auto expected = reference->forward(hidden, tokens, input);
      const auto local_hidden = mlu_v4_cp::shard_rows(hidden, *cp);
      // Keep the global token pack and global request histories on both ranks.
      // Production Engram reduces their shard contributions before applying
      // exactly the model's zigzag local_row_indices.
      const auto actual = sharded->forward(
          local_hidden, tokens, input, nullptr, cp->local_row_indices);
      const auto expected_local =
          expected.index_select(0, cp->local_row_indices);
      CHECK(torch::allclose(actual, expected_local, 0.02, 0.02))
          << "Engram CP local rows differ: rank=" << group.rank()
          << " cpu_offload=" << cpu_offload;
      const auto restored = mlu_v4_cp::gather_restore(actual, *cp);
      CHECK(torch::allclose(restored, expected, 0.02, 0.02))
          << "Engram CP restore differs: cpu_offload=" << cpu_offload;
      CHECK_GT((expected - hidden).abs().max().item<float>(), 0.05f)
          << "Fixture must have a visible Engram contribution";
      // Refresh only the per-request history, preserving tokens and geometry.
      // This exposes implementations that hash local chunks as new sequences.
      for (auto& token : input.embedding.engram_lookback_token_ids) {
        token = (token + 2) % 7 + 1;
      }
      const auto next_expected = reference->forward(hidden, tokens, input);
      CHECK(!torch::allclose(next_expected, expected, 0.02, 0.02))
          << "Fixture must distinguish request histories";
      const auto next_actual = sharded->forward(
          local_hidden, tokens, input, nullptr, cp->local_row_indices);
      CHECK(
          torch::allclose(next_actual,
                          next_expected.index_select(0, cp->local_row_indices),
                          0.02,
                          0.02));
      CHECK(torch::allclose(mlu_v4_cp::gather_restore(next_actual, *cp),
                            next_expected,
                            0.02,
                            0.02));
    }
  }
}

int32_t run_child(int32_t rank, int32_t port) {
  try {
    if (Platform::device_count() < kWorldSize) {
      return kSkip;
    }
    Device xllm_device(rank);
    xllm_device.set_device();
    const torch::Device device = xllm_device.unwrap();
    auto group = create_process_group(rank,
                                      kWorldSize,
                                      kWorldSize,
                                      port,
                                      /*trans=*/false,
                                      /*host=*/"127.0.0.1",
                                      /*group_name=*/"deepseek_v41_engram_cp",
                                      device);
    CHECK(group != nullptr);
    verify_cp_engram(*group, device);
    xllm_device.synchronize_default_stream();
    return 0;
  } catch (const std::exception& error) {
    LOG(ERROR) << "Engram CP rank " << rank << " failed: " << error.what();
    return 1;
  }
}

ChildOutcome wait_children(const std::vector<pid_t>& pids) {
  std::array<bool, kWorldSize> finished{};
  int32_t remaining = kWorldSize;
  int32_t skipped = 0;
  bool failed = false;
  const auto deadline = std::chrono::steady_clock::now() + kChildTimeout;
  while (remaining > 0 && !failed &&
         std::chrono::steady_clock::now() < deadline) {
    for (int32_t rank = 0; rank < kWorldSize; ++rank) {
      if (finished[rank]) {
        continue;
      }
      int status = 0;
      const pid_t result = ::waitpid(pids[rank], &status, WNOHANG);
      if (result == 0 || (result < 0 && errno == EINTR)) {
        continue;
      }
      finished[rank] = true;
      --remaining;
      if (result > 0 && WIFEXITED(status) && WEXITSTATUS(status) == kSkip) {
        ++skipped;
        continue;
      }
      if (result < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        failed = true;
        LOG(ERROR) << "Engram CP rank " << rank << " failed, status=" << status;
      }
    }
    if (remaining > 0 && !failed) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    if (finished[rank]) {
      continue;
    }
    failed = true;
    LOG(ERROR) << "Stopping stalled Engram CP rank " << rank;
    ::kill(pids[rank], SIGKILL);
    int status = 0;
    while (::waitpid(pids[rank], &status, 0) < 0 && errno == EINTR) {
    }
  }
  if (failed || (skipped != 0 && skipped != kWorldSize)) {
    return ChildOutcome::FAILED;
  }
  return skipped == kWorldSize ? ChildOutcome::SKIPPED : ChildOutcome::PASSED;
}

TEST(DeepseekV41EngramCpTest,
     TwoRanksMatchReplicatedForUnevenRequestsAndHistory) {
  const int32_t port = net::get_local_free_port();
  ASSERT_GT(port, 0);
  std::vector<pid_t> pids;
  pids.reserve(kWorldSize);
  // The parent never initializes MLU; both ranks select their device after
  // fork.
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    const pid_t pid = ::fork();
    if (pid == 0) {
      _exit(run_child(rank, port));
    }
    if (pid < 0) {
      for (pid_t child : pids) {
        ::kill(child, SIGKILL);
        int status = 0;
        while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
        }
      }
      FAIL() << "Cannot fork Engram CP rank " << rank << ": "
             << std::strerror(errno);
      return;
    }
    pids.emplace_back(pid);
  }
  const ChildOutcome outcome = wait_children(pids);
  if (outcome == ChildOutcome::SKIPPED) {
    GTEST_SKIP() << "Requires two visible MLU devices";
  }
  EXPECT_EQ(outcome, ChildOutcome::PASSED);
}

}  // namespace
}  // namespace xllm::layer
