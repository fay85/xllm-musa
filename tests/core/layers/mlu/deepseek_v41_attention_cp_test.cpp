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
#include <vector>

#include "core/framework/parallel_state/process_group.h"
#include "core/layers/mlu/deepseek_v4/deepseek_v4_cp_context.h"
#include "core/layers/mlu/deepseek_v41/deepseek_v41_attention.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/util/net.h"

namespace xllm::layer {
namespace {

constexpr int32_t kWorldSize = 2;
constexpr int32_t kSkip = 77;
constexpr std::chrono::seconds kChildTimeout{120};

enum class ChildOutcome : int8_t { PASSED, SKIPPED, FAILED };

void verify_cp_attention(ProcessGroup& group, const torch::Device& device) {
  torch::InferenceMode inference;
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto ints = options.dtype(torch::kInt32);
  ProcessGroup single(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &single);
  parallel.tp_group_ = &single;
  auto wkv = torch::zeros({512, 32}, options);
  wkv.select(1, 0).fill_(0.125);
  auto wo_a = torch::zeros({1, 8, 4096}, options);
  wo_a[0].slice(1, 0, 8).copy_(torch::eye(8, options));
  auto wo_b = torch::zeros({32, 8}, options);
  wo_b.slice(0, 0, 8).copy_(torch::eye(8, options));
  const StateDict weights({
      {"wq_a.weight", torch::zeros({16, 32}, options)},
      {"wq_b.weight", torch::zeros({4096, 16}, options)},
      {"wkv.weight", wkv},
      {"q_norm.weight", torch::ones({16}, options)},
      {"kv_norm.weight", torch::ones({512}, options)},
      {"wo_a.weight", wo_a},
      {"wo_b.weight", wo_b},
      {"attn_sink", torch::zeros({8}, options)},
      {"compressor.wkv.weight", wkv},
      {"compressor.wgate.weight", wkv},
      {"compressor.norm.weight", torch::ones({512}, options)},
      {"indexer.wk.weight", torch::full({128, 512}, 0.125, options)},
      {"indexer.k_norm.weight", torch::ones({128}, options)},
  });
  auto rope =
      torch::cat({torch::ones({1024, 64}, options.dtype(torch::kFloat32)),
                  torch::zeros({1024, 64}, options.dtype(torch::kFloat32))},
                 1);
  for (int32_t ratio : {0, 1, 2}) {
    ModelArgs args;
    args.hidden_size(32)
        .n_heads(8)
        .head_dim(512)
        .q_lora_rank(16)
        .qk_rope_head_dim(64)
        .o_groups(1)
        .o_lora_rank(8)
        .window_size(16)
        .rms_norm_eps(1e-20)
        .compress_ratios({ratio, ratio})
        .kv_source_layer_ids({0})
        .index_head_dim(128)
        .index_topk(2)
        .num_speculative_tokens(5);
    const ModelContext context(parallel, args, QuantArgs(), options);
    std::vector<DeepseekV41Attention> layers;
    layers.reserve(2);
    std::vector<KVCache> cp_caches, reference_caches;
    cp_caches.reserve(2);
    reference_caches.reserve(2);
    for (int32_t id = 0; id < 2; ++id) {
      layers.emplace_back(context, id);
      layers.back()->load_state_dict(weights);
      DeepSeekV4KVCacheTensors tensors;
      tensors.swa_cache = torch::full({2, 128, 1, 512}, 0.25, options);
      tensors.key_cache = torch::full({12, 128, 1, 512}, 0.5, options);
      tensors.index_cache = torch::full({12, 128, 1, 128}, -1, options);
      tensors.compress_kv_state =
          torch::full({2, 32, 512}, 0.25, options.dtype(torch::kFloat32));
      tensors.compress_score_state =
          torch::zeros_like(tensors.compress_kv_state);
      cp_caches.emplace_back(tensors);
      tensors.swa_cache = tensors.swa_cache.clone();
      tensors.key_cache = tensors.key_cache.clone();
      tensors.index_cache = tensors.index_cache.clone();
      tensors.compress_kv_state = tensors.compress_kv_state.clone();
      tensors.compress_score_state = tensors.compress_score_state.clone();
      reference_caches.emplace_back(tensors);
    }
    std::vector<int32_t> begins = {125, 3};
    for (const std::vector<int32_t>& lengths :
         {std::vector<int32_t>{7, 11}, std::vector<int32_t>{390, 9}}) {
      const int32_t total = lengths[0] + lengths[1];
      auto positions =
          torch::cat({torch::arange(begins[0], begins[0] + lengths[0], ints),
                      torch::arange(begins[1], begins[1] + lengths[1], ints)})
              .to(torch::kInt64);
      const auto cp = mlu_v4_cp::build_cp_context(
          kWorldSize, group.rank(), &group, lengths, positions);
      CHECK(cp.has_value());
      auto hidden = torch::arange(total * 32, options.dtype(torch::kFloat32))
                        .remainder(9)
                        .sub(4)
                        .div(8)
                        .view({total, 32})
                        .to(options);
      const auto local_hidden = mlu_v4_cp::shard_rows(hidden, *cp);
      ModelInputParams input;
      input.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
      input.meta.num_sequences = 2;
      input.attention.host.q_seq_lens = {0, lengths[0], total};
      input.multi_block_tables = {
          torch::tensor({1, 0}, torch::kInt32).view({2, 1}),
          torch::tensor({6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5}, torch::kInt32)
              .view({2, 6}),
          torch::tensor({6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5}, torch::kInt32)
              .view({2, 6})};
      Csa2Context cp_shared, reference_shared;
      deepseek_v41::prepare_metadata(
          cp_shared, input, positions, /*num_sequences=*/2);
      deepseek_v41::prepare_metadata(
          reference_shared, input, positions, /*num_sequences=*/2);
      deepseek_v41::prepare_workspace(cp_shared, input, args, total, options);
      CHECK(!cp_shared.selected_kv.defined());
      cp_shared.num_valid_input_rows.fill_(local_hidden.size(0));
      cp_shared.topk.resize(4);
      reference_shared.topk.resize(2);
      const auto make_topk = [&](int64_t rows) {
        return ratio == 0
                   ? torch::empty({rows, 0}, options.dtype(torch::kInt64))
                   : torch::tensor({0, 1}, options.dtype(torch::kInt64))
                         .repeat({rows, 1});
      };
      for (int32_t seq = 0; seq < 2; ++seq) {
        reference_shared.topk[seq] = make_topk(lengths[seq]);
        cp_shared.topk[seq * 2] =
            make_topk(cp->geometry.front_seq_lens_by_rank[group.rank()][seq]);
        cp_shared.topk[seq * 2 + 1] =
            make_topk(cp->geometry.back_seq_lens_by_rank[group.rank()][seq]);
      }
      for (int32_t id = 0; id < 2; ++id) {
        const auto expected = layers[id]->forward(
            hidden, positions, rope, reference_caches, input, reference_shared);
        const auto actual = layers[id]->forward(local_hidden,
                                                cp->local_positions,
                                                rope,
                                                cp_caches,
                                                input,
                                                cp_shared,
                                                &*cp);
        CHECK(torch::allclose(
            actual, mlu_v4_cp::shard_rows(expected, *cp), 0.01, 0.01))
            << "CP query mapping failed: rank=" << group.rank()
            << " ratio=" << ratio << " layer=" << id << " rows=" << total;
        CHECK(torch::allclose(
            mlu_v4_cp::gather_restore(actual, *cp), expected, 0.01, 0.01));
        CHECK(torch::equal(cp_caches[id].get_swa_cache(),
                           reference_caches[id].get_swa_cache()));
        CHECK(torch::equal(cp_caches[id].get_k_cache(),
                           reference_caches[id].get_k_cache()));
        CHECK(torch::equal(cp_caches[id].get_index_cache(),
                           reference_caches[id].get_index_cache()));
        CHECK(torch::equal(cp_caches[id].get_compress_kv_state(),
                           reference_caches[id].get_compress_kv_state()));
        CHECK(torch::equal(cp_caches[id].get_compress_score_state(),
                           reference_caches[id].get_compress_score_state()));
      }
      begins[0] += lengths[0];
      begins[1] += lengths[1];
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
    auto group =
        create_process_group(rank,
                             kWorldSize,
                             kWorldSize,
                             port,
                             /*trans=*/false,
                             /*host=*/"127.0.0.1",
                             /*group_name=*/"deepseek_v41_attention_cp",
                             device);
    CHECK(group != nullptr);
    verify_cp_attention(*group, device);
    xllm_device.synchronize_default_stream();
    return 0;
  } catch (const std::exception& error) {
    LOG(ERROR) << "Attention CP rank " << rank << " failed: " << error.what();
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
      int32_t status = 0;
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
        LOG(ERROR) << "Attention CP rank " << rank
                   << " failed, status=" << status;
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
    LOG(ERROR) << "Stopping stalled Attention CP rank " << rank;
    ::kill(pids[rank], SIGKILL);
    int32_t status = 0;
    while (::waitpid(pids[rank], &status, 0) < 0 && errno == EINTR) {
    }
  }
  if (failed || (skipped != 0 && skipped != kWorldSize)) {
    return ChildOutcome::FAILED;
  }
  return skipped == kWorldSize ? ChildOutcome::SKIPPED : ChildOutcome::PASSED;
}

TEST(DeepseekV41AttentionCpTest,
     TwoRanksMatchReplicatedAcrossCompressionAndRingWrap) {
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
        int32_t status = 0;
        while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
        }
      }
      FAIL() << "Cannot fork Attention CP rank " << rank << ": "
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
