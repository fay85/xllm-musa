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

#include "layers/mlu/deepseek_v41/engram_shared_memory.h"

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cerrno>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/config/model_config.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_engram.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"

namespace xllm::layer {
namespace {

constexpr int32_t kSkip = 77;
using RankCheck = void (*)(ProcessGroup&);

// Emulate two hosts using real collectives on one physical test machine. Each
// process sees its four host peers with the local boot ID and the others with
// a different ID. This exercises per-host leaders and load ranges, including
// a host whose leader has nonzero world rank.
class SimulatedHostsProcessGroup final : public ProcessGroup {
 public:
  explicit SimulatedHostsProcessGroup(ProcessGroup& world)
      : ProcessGroup(world.rank(), world.world_size(), world.device()),
        world_(world) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    auto output = world_.allgather_base_sync(input);
    if (input.numel() != 1) {
      return output;
    }
    auto hosts = output.cpu();
    int64_t* ids = hosts.data_ptr<int64_t>();
    for (int32_t peer = 0; peer < world_size(); ++peer) {
      if (peer / 4 == rank() / 4) {
        continue;
      }
      ids[peer] ^= 1;
    }
    return hosts.to(input.device());
  }

  void allreduce(torch::Tensor& input) override { world_.allreduce(input); }

 private:
  ProcessGroup& world_;
};

void synchronize(ProcessGroup& group) {
  auto ready = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kInt64).device(group.device()));
  group.allreduce(ready);
  CHECK_EQ(ready.cpu().item<int64_t>(), group.world_size());
}

void check_shared_table(ProcessGroup& group, int64_t rows, int64_t columns) {
  auto expected = torch::arange(rows * columns)
                      .remainder(251)
                      .reshape({rows, columns})
                      .to(torch::kUInt8);
  // Only this rank's source range is valid. A private copy cannot reproduce
  // the other ranks' rows, including when some ranks own no rows at all.
  auto source = torch::full_like(expected, 255);
  const int64_t begin = rows * group.rank() / group.world_size();
  const int64_t end = rows * (group.rank() + 1) / group.world_size();
  source.slice(0, begin, end).copy_(expected.slice(0, begin, end));
  source = source.t().contiguous().t();
  auto table = load_shared_engram_tensor(
      source,
      "table_" + std::to_string(rows) + "_" + std::to_string(columns),
      group);
  CHECK(table.device().is_cpu());
  CHECK_EQ(table.scalar_type(), torch::kUInt8);
  CHECK(torch::equal(table, expected)) << "rank=" << group.rank();
  source.fill_(255);
  CHECK(torch::equal(table, expected)) << "Shared table aliases checkpoint";

  // A tensor view keeps the mapping alive independently of the first handle.
  auto alias = table.view({-1});
  table = torch::Tensor();
  CHECK(torch::equal(alias, expected.reshape({-1})));
  // Observe one writer from every process: equal initial contents alone do
  // not prove that the mappings share physical storage.
  synchronize(group);
  if (group.rank() == 0) {
    alias[0] = 203;
  }
  synchronize(group);
  CHECK_EQ(alias[0].item<uint8_t>(), 203);
}

void check_shared_forward_with_context(const ParallelArgs& parallel) {
  ProcessGroup& group = *parallel.tp_group_;
  ProcessGroup& world = *parallel.process_group_;
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
  auto options =
      torch::TensorOptions().device(group.device()).dtype(torch::kBFloat16);
  ModelContext context(parallel, args, QuantArgs(), options);
  const std::unordered_map<std::string, torch::Tensor> weights = {
      {"embed.weight",
       torch::arange(18 * 32).remainder(96).reshape({18, 32}).to(
           torch::kUInt8)},
      {"embed.scale",
       torch::arange(18).remainder(4).add(125).reshape({18, 1}).to(
           torch::kUInt8)},
      {"wkv.weight",
       torch::linspace(-0.1, 0.1, 24 * 64)
           .reshape({24, 64})
           .to(torch::kBFloat16)},
      {"q_weight", torch::ones({2, 8}, torch::kBFloat16)},
      {"k_weight", torch::ones({2, 8}, torch::kBFloat16)}};
  const StateDict hash_weights({
      {"token_map", torch::arange(8, torch::kInt64)},
      {"multipliers", torch::tensor({{1, 3, 5}}, torch::kInt64)},
      {"primes", torch::tensor({{7, 11}}, torch::kInt64)},
      {"offsets", torch::tensor({{0, 7}}, torch::kInt64)},
  });
  const std::string saved_config = ModelConfig::get_instance().engram_config();
  auto create = [&](const std::string& config, const StateDict& checkpoint) {
    ModelConfig::get_instance().engram_config() = config;
    DeepseekV41Engram layer(context, /*engram_id=*/0);
    layer->load_state_dict(checkpoint);
    layer->load_hash_state(hash_weights);
    layer->verify_loaded_weights();
    return layer;
  };
  auto sharded =
      create(R"({"placement":"cpu","replication":"none"})", StateDict(weights));
  auto shared_weights = weights;
  for (const std::string name : {"embed.weight", "embed.scale"}) {
    const auto& checkpoint = weights.at(name);
    const int64_t begin =
        checkpoint.size(0) * world.rank() / world.world_size();
    const int64_t end =
        checkpoint.size(0) * (world.rank() + 1) / world.world_size();
    // Each process supplies only its node-wide load range. Loading within
    // independent TP groups would leave holes and fail the numerical checks.
    auto local = torch::zeros_like(checkpoint);
    local.slice(0, begin, end).copy_(checkpoint.slice(0, begin, end));
    shared_weights[name] = std::move(local);
  }
  auto shared = create(R"({"placement":"cpu","replication":"per_node"})",
                       StateDict(shared_weights));
  ModelConfig::get_instance().engram_config() = saved_config;

  ModelInputParams input;
  input.meta.num_sequences = 4;
  input.meta.actual_num_sequences = 4;
  input.attention.host.q_seq_lens = {0, 1, 2, 3, 4};
  const int32_t dp_rank = parallel.rank() / group.world_size();
  input.embedding.engram_lookback_token_ids.assign(8, dp_rank + 1);
  auto tokens = torch::tensor({2, 5, 1, 7}, torch::kInt64)
                    .add(dp_rank)
                    .remainder(8)
                    .to(group.device());
  auto hidden =
      torch::linspace(-1, 1, 4 * 2 * 8).reshape({4, 2, 8}).to(options);
  auto sharded_state = sharded->create_hash_state(/*rows=*/4);
  auto shared_state = shared->create_hash_state(/*rows=*/4);
  sharded_state->prepare(input);
  shared_state->prepare(input);
  // Ensure the input queries rows loaded by both halves of the node.
  auto hashes = shared_state->forward(tokens).cpu();
  CHECK(hashes.lt(9).any().item<bool>());
  CHECK(hashes.ge(9).any().item<bool>());
  auto expected = sharded->forward(hidden, tokens, input).cpu();
  CHECK(torch::equal(shared->forward(hidden, tokens, input).cpu(), expected));
  CHECK(torch::equal(
      sharded->forward(hidden, tokens, input, sharded_state.get()).cpu(),
      expected));
  CHECK(torch::equal(
      shared->forward(hidden, tokens, input, shared_state.get()).cpu(),
      expected));
  torch_mlu::synchronize();

  torch_mlu::MLUGraph sharded_graph;
  torch_mlu::MLUGraph shared_graph;
  torch::Tensor sharded_output;
  torch::Tensor shared_output;
  {
    torch_mlu::mlu::MLUStreamGuard stream(torch_mlu::getStreamFromPool(
        /*isHighPriority=*/false, group.device().index()));
    sharded_graph.capture_begin();
    sharded_output =
        sharded->forward(hidden, tokens, input, sharded_state.get());
    sharded_graph.capture_end();
    shared_graph.capture_begin();
    shared_output = shared->forward(hidden, tokens, input, shared_state.get());
    shared_graph.capture_end();
  }
  for (int32_t round = 0; round < 3; ++round) {
    // First change history alone, then change tokens alone. Both must refresh
    // replay's lookup results while preserving the captured tensor addresses.
    if (round == 1) {
      input.embedding.engram_lookback_token_ids.assign(8, dp_rank + 2);
    } else if (round == 2) {
      tokens.copy_(
          torch::tensor({6, 3, 7, 2}, torch::kInt64).add(dp_rank).remainder(8));
    }
    sharded_state->prepare(input);
    shared_state->prepare(input);
    auto previous = expected;
    expected = sharded->forward(hidden, tokens, input).cpu();
    CHECK(torch::equal(shared->forward(hidden, tokens, input).cpu(), expected))
        << "eager rank=" << group.rank() << " round=" << round;
    CHECK(round == 0 || !torch::equal(previous, expected))
        << "Test inputs must change the expected output";
    sharded_graph.replay();
    shared_graph.replay();
    CHECK(torch::equal(sharded_output.cpu(), expected))
        << "sharded replay rank=" << group.rank() << " round=" << round;
    CHECK(torch::equal(shared_output.cpu(), expected))
        << "shared replay rank=" << group.rank() << " round=" << round;
  }
}

void check_shared_forward(ProcessGroup& group) {
  ParallelArgs parallel(group.rank(), group.world_size(), &group);
  parallel.tp_group_ = &group;
  check_shared_forward_with_context(parallel);
}

void check_dp_shared_forward(ProcessGroup& world) {
  constexpr int32_t kDpSize = 2;
  constexpr int32_t kTpSize = 4;
  CHECK_EQ(world.world_size(), kDpSize * kTpSize);
  auto ports = torch::zeros({kDpSize}, torch::kInt64);
  if (world.rank() == 0) {
    ports[0] = net::get_local_free_port();
    do {
      ports[1] = net::get_local_free_port();
    } while (ports[1].item<int64_t>() == ports[0].item<int64_t>());
  }
  ports = ports.to(world.device());
  world.broadcast(ports, /*root_rank=*/0);
  ports = ports.cpu();
  const int32_t dp_rank = world.rank() / kTpSize;
  auto tp = create_process_group(world.rank(),
                                 world.world_size(),
                                 kTpSize,
                                 ports[dp_rank].item<int32_t>(),
                                 /*trans=*/false,
                                 "127.0.0.1",
                                 "engram_attention_tp",
                                 world.device());
  ParallelArgs parallel(world.rank(),
                        world.world_size(),
                        kDpSize,
                        /*cp_size=*/1,
                        &world,
                        /*ep_size=*/1);
  parallel.tp_size(kTpSize);
  parallel.tp_group_ = tp.get();
  check_shared_forward_with_context(parallel);
}

void check_shared_tables(ProcessGroup& group) {
  for (const int64_t rows : {19, 3}) {
    for (const int64_t columns : {32, 1}) {
      check_shared_table(group, rows, columns);
    }
  }
}

void check_separate_hosts(ProcessGroup& world) {
  SimulatedHostsProcessGroup group(world);
  const int32_t node = world.rank() / 4;
  const int32_t local_rank = world.rank() % 4;
  auto expected =
      torch::arange(19 * 32).add(node * 37).remainder(251).reshape({19, 32}).to(
          torch::kUInt8);
  auto source = torch::full_like(expected, 255);
  const int64_t begin = 19 * local_rank / 4;
  const int64_t end = 19 * (local_rank + 1) / 4;
  source.slice(0, begin, end).copy_(expected.slice(0, begin, end));
  auto table = load_shared_engram_tensor(source, "per_host_table", group);
  CHECK(torch::equal(table, expected));
  synchronize(group);
  if (local_rank == 0) {
    table[0][0] = 200 + node;
  }
  synchronize(group);
  CHECK_EQ(table[0][0].item<int32_t>(), 200 + node);
}

int32_t run_rank(int32_t rank,
                 int32_t world_size,
                 int32_t port,
                 RankCheck check) {
  ::alarm(240);
  if (Platform::device_count() < world_size) {
    return kSkip;
  }
  torch::InferenceMode inference;
  Device device(rank);
  device.set_device();
  auto group = create_process_group(rank,
                                    world_size,
                                    world_size,
                                    port,
                                    /*trans=*/false,
                                    "127.0.0.1",
                                    "engram_shared_memory_test",
                                    device.unwrap());
  check(*group);
  device.synchronize_default_stream();
  return 0;
}

void run_ranks(int32_t world_size, RankCheck check) {
  const int32_t port = net::get_local_free_port();
  ASSERT_GT(port, 0);
  // This standalone binary forks before the parent initializes MLU.
  std::vector<pid_t> children;
  children.reserve(world_size);
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      ::_exit(run_rank(rank, world_size, port, check));
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
  if (skipped == world_size) {
    GTEST_SKIP() << "Requires " << world_size << " visible MLU devices.";
  }
  EXPECT_EQ(skipped, 0);
}

TEST(EngramSharedMemoryTest, EightRanksShareRawBytesAndMappingLifetime) {
  run_ranks(/*world_size=*/8, check_shared_tables);
}

TEST(EngramSharedMemoryTest, TwoRanksMatchShardedEagerAndGraphReplay) {
  run_ranks(/*world_size=*/2, check_shared_forward);
}

TEST(EngramSharedMemoryTest, TwoDpFourTpShareAcrossEagerAndGraphReplay) {
  run_ranks(/*world_size=*/8, check_dp_shared_forward);
}

TEST(EngramSharedMemoryTest,
     SeparateHostsUseIndependentTablesAndLocalLoadRanges) {
  run_ranks(/*world_size=*/8, check_separate_hosts);
}

}  // namespace
}  // namespace xllm::layer
