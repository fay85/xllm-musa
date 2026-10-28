/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "framework/block/block_manager_pool.h"
#include "framework/block/musa/deepseek_v4_prefix_transfer.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"

namespace xllm {
namespace {

class DeepseekV4PDPrefixPolicyTest : public ::testing::Test {
 protected:
  std::unique_ptr<Sequence> make_sequence() {
    SequenceParams params;
    params.seq_capacity = 1024;
    params.stopping_checker = &stopping_checker_;
    params.sampling_param = &sampling_params_;
    params.enable_schedule_overlap = false;
    return std::make_unique<Sequence>(
        /*index=*/0,
        std::vector<int32_t>(212, 42),
        torch::Tensor(),
        MMData(),
        IncrementalDecoder("", 1, false, false),
        params);
  }

  BlockManagerPool::Options options(bool decode) {
    BlockManagerPool::Options result;
    result.num_blocks(4096)
        .block_size(128)
        .enable_prefix_cache(true)
        .enable_disagg_pd(true)
        .instance_is_decode(decode)
        .sliding_window_size(128)
        .swa_blocks_per_seq(2)
        .swa_num_blocks(/*rings, burst, snapshots, padding=*/2 * 16 + 10 + 16 +
                        2)
        .max_tokens_per_batch(1280)
        .max_seqs_per_batch(16)
        .num_embedding_blocks(16)
        .num_speculative_tokens(6)
        .manager_types({1, 0, 0})
        .compress_ratios({0, 4, 128});
    return result;
  }

  RequestSamplingParam sampling_params_;
  StoppingChecker stopping_checker_;
};

TEST_F(DeepseekV4PDPrefixPolicyTest, CompressedAdmissionExcludesNullBlocks) {
  BlockManagerPool pool(options(/*decode=*/true));
  // C4 and C128 reserve one physical page each. Admission is bounded by
  // min((1024 - 1) * 4, (32 - 1) * 128) base blocks.
  EXPECT_EQ(pool.num_free_blocks(), (std::vector<size_t>{3968}));
}

TEST_F(DeepseekV4PDPrefixPolicyTest, DecodeDoesNotPublishUnwrittenSnapshot) {
  BlockManagerPool pool(options(/*decode=*/true));
  auto first = make_sequence();
  ASSERT_TRUE(pool.allocate(first.get(), /*num_tokens=*/212));
  EXPECT_TRUE(pool.get_swap_block_transfer_infos()->at(0).empty());
  // PULL imported the complete prompt. No prefill forward produced the
  // intermediate 128-token snapshot on this instance.
  first->kv_state().set_kv_cache_tokens_num(212);
  pool.deallocate(first.get());
  pool.get_swap_block_transfer_infos()->at(0).clear();
  auto second = make_sequence();
  ASSERT_TRUE(pool.allocate(second.get(), /*num_tokens=*/212));
  EXPECT_EQ(second->kv_state().kv_cache_tokens_num(), 0);
  EXPECT_TRUE(pool.get_swap_block_transfer_infos()->at(0).empty());
  pool.deallocate(second.get());
}

TEST_F(DeepseekV4PDPrefixPolicyTest, PrefillStillRequestsBoundarySnapshot) {
  BlockManagerPool pool(options(/*decode=*/false));
  auto sequence = make_sequence();
  ASSERT_TRUE(pool.allocate(sequence.get(), /*num_tokens=*/212));
  const auto& transfers = pool.get_swap_block_transfer_infos()->at(0);
  ASSERT_EQ(transfers.size(), 1);
  EXPECT_EQ(musa::decode_deepseek_v4_prefix_transfer(transfers.front()).kind,
            musa::DeepseekV4PrefixTransferKind::STATE_SNAPSHOT);
  pool.deallocate(sequence.get());
}

}  // namespace
}  // namespace xllm
