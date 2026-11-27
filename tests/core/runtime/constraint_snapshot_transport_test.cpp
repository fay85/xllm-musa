/* Copyright 2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/runtime/forward_shared_memory_manager.h"
#include "core/runtime/params_utils.h"
#include "core/scheduler/disagg_pd_scheduler.h"

namespace xllm {
namespace {

// The RPC receive adapter allocates NPU-pinned memory even for a CPU device.
// Exercise its descriptor payload using ordinary CPU storage so these codec
// regressions do not require device initialization.
ForwardInput stage_packed_input_on_cpu(
    const proto::PackedForwardInput& packed) {
  ForwardInput input;
  input.input_host_buffer = torch::empty(
      {static_cast<int64_t>(packed.payload().size())},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  std::memcpy(input.input_host_buffer.data_ptr(),
              packed.payload().data(),
              packed.payload().size());
  input.input_host_buffer_has_layout = true;
  return input;
}

class ConstraintSnapshotTransportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& config = ExecutionConfig::get_instance();
    previous_contiguous_ = config.use_contiguous_input_buffer();
    // Exercise the real host SHM descriptor codec without device upload or
    // process-global NPU initialization; device materialization is separate.
    config.use_contiguous_input_buffer() = false;
  }
  void TearDown() override {
    ExecutionConfig::get_instance().use_contiguous_input_buffer() =
        previous_contiguous_;
  }

 private:
  bool previous_contiguous_ = false;
};

TEST_F(ConstraintSnapshotTransportTest,
       ForwardInputPackedRoundTripKeepsGrammarSpec) {
  // G9: the per-request grammar identity rides the packed input transport
  // (v8 descriptor), so a cross-process receiver can rebuild the grammar.
  ForwardInput input;
  input.json_object_state_snapshots.resize(1);
  auto& snapshot = input.json_object_state_snapshots[0];
  snapshot.enabled = true;
  snapshot.token_ids = {0, 1};
  GrammarSpec spec;
  spec.type = GrammarSpec::Type::JSON_SCHEMA;
  spec.source = R"({"type":"object"})";
  spec.strict = true;
  snapshot.grammar_spec = std::make_shared<const GrammarSpec>(std::move(spec));

  bool is_creator = false;
  auto shm_name =
      ForwardSharedMemoryManager::create_unique_name("batch_test_grammar_spec",
                                                     /*dp_group=*/0,
                                                     ForwardType::RAW_INPUT,
                                                     /*rank=*/0);
  ForwardSharedMemoryManager writer_manager(
      shm_name, 1 << 20, is_creator, ForwardType::RAW_INPUT);
  bool is_reader_creator = false;
  ForwardSharedMemoryManager reader_manager(
      shm_name, 1 << 20, is_reader_creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(writer_manager.input_write(input));

  ForwardInput round_trip;
  reader_manager.input_read(round_trip, torch::Device(torch::kCPU));
  ASSERT_EQ(round_trip.json_object_state_snapshots.size(), 1u);
  const auto& out = round_trip.json_object_state_snapshots[0];
  EXPECT_TRUE(out.enabled);
  EXPECT_EQ(out.token_ids, std::vector<int32_t>({0, 1}));
  ASSERT_NE(out.grammar_spec, nullptr);
  EXPECT_EQ(out.grammar_spec->type, GrammarSpec::Type::JSON_SCHEMA);
  EXPECT_TRUE(out.grammar_spec->strict);
  EXPECT_EQ(out.grammar_spec->source, R"({"type":"object"})");

  // Reuse the receive object for a builtin snapshot (empty source). resize
  // keeps the old entry alive, so the reader must explicitly clear its text.
  snapshot.grammar_spec = nullptr;
  snapshot.token_ids.clear();
  ASSERT_TRUE(writer_manager.input_write(input));
  ASSERT_TRUE(
      reader_manager.input_read(round_trip, torch::Device(torch::kCPU)));
  ASSERT_EQ(round_trip.json_object_state_snapshots.size(), 1u);
  EXPECT_EQ(round_trip.json_object_state_snapshots[0].grammar_spec, nullptr);
}

TEST_F(ConstraintSnapshotTransportTest,
       ForwardInputPackedRoundTripKeepsEbnfGrammarSpec) {
  // The EBNF spec type must survive the same packed transport as schemas
  // and regexes: the reader's type CHECK admits it, so a PD decode worker
  // can rebuild an EBNF machine instead of crashing on the descriptor.
  ForwardInput input;
  input.json_object_state_snapshots.resize(1);
  auto& snapshot = input.json_object_state_snapshots[0];
  snapshot.enabled = true;
  snapshot.token_ids = {0, 1};
  GrammarSpec spec;
  spec.type = GrammarSpec::Type::EBNF;
  spec.source = "root ::= \"yes\"";
  snapshot.grammar_spec = std::make_shared<const GrammarSpec>(std::move(spec));

  bool is_creator = false;
  auto shm_name =
      ForwardSharedMemoryManager::create_unique_name("batch_test_ebnf_spec",
                                                     /*dp_group=*/0,
                                                     ForwardType::RAW_INPUT,
                                                     /*rank=*/0);
  ForwardSharedMemoryManager writer_manager(
      shm_name, 1 << 20, is_creator, ForwardType::RAW_INPUT);
  bool is_reader_creator = false;
  ForwardSharedMemoryManager reader_manager(
      shm_name, 1 << 20, is_reader_creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(writer_manager.input_write(input));

  ForwardInput round_trip;
  reader_manager.input_read(round_trip, torch::Device(torch::kCPU));
  ASSERT_EQ(round_trip.json_object_state_snapshots.size(), 1u);
  const auto& out = round_trip.json_object_state_snapshots[0];
  ASSERT_NE(out.grammar_spec, nullptr);
  EXPECT_EQ(out.grammar_spec->type, GrammarSpec::Type::EBNF);
  EXPECT_FALSE(out.grammar_spec->strict);
  EXPECT_EQ(out.grammar_spec->source, "root ::= \"yes\"");
}

TEST_F(ConstraintSnapshotTransportTest,
       PackedGrammarSourceCannotReadPastDescriptor) {
  ForwardInput input;
  input.json_object_state_snapshots.resize(1);
  auto& snapshot = input.json_object_state_snapshots[0];
  snapshot.enabled = true;
  GrammarSpec spec;
  spec.type = GrammarSpec::Type::REGEX;
  spec.source = "g9-descriptor-boundary-sentinel";
  snapshot.grammar_spec = std::make_shared<const GrammarSpec>(std::move(spec));
  bool is_creator = false;
  const std::string name = ForwardSharedMemoryManager::create_unique_name(
      "batch_test_bad_grammar_length",
      /*dp_group=*/0,
      ForwardType::RAW_INPUT,
      /*rank=*/0);
  ForwardSharedMemoryManager writer(
      name, 1 << 20, is_creator, ForwardType::RAW_INPUT);
  bool is_reader_creator = false;
  ForwardSharedMemoryManager reader(
      name, 1 << 20, is_reader_creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(writer.input_write(input));
  char* base = static_cast<char*>(writer.base_address());
  const uint64_t bytes = ForwardSharedMemoryManager::input_size(input);
  const std::string& marker = snapshot.grammar_spec->source;
  char* source = std::search(base, base + bytes, marker.begin(), marker.end());
  ASSERT_NE(source, base + bytes);
  ASSERT_GE(source - base, static_cast<std::ptrdiff_t>(sizeof(uint64_t)));
  // Corrupt only the source's length word: it passes the 16-MiB application
  // cap but exceeds this actual descriptor, and must fail before copying.
  constexpr uint64_t kBadSourceLength = 1ULL << 20;
  std::memcpy(source - sizeof(uint64_t), &kBadSourceLength, sizeof(uint64_t));
  EXPECT_DEATH(
      {
        ForwardInput output;
        reader.input_read(output, torch::Device(torch::kCPU));
      },
      "grammar source exceeds descriptor");
}

TEST_F(ConstraintSnapshotTransportTest,
       TransferMappingsKeepSharedPrefixAndCursorAcrossPackedAndSharedInput) {
  ForwardInput input;
  TransferKVInfo info;
  info.request_id = "shared-prefix-and-cursor";
  KVTransferMapping mapping;
  mapping.local_ids = {7, 8};
  mapping.remote_ids = {17, 18};
  mapping.remote_shared_num = 815;
  mapping.local_block_start = 203;
  info.mappings.emplace_back(std::move(mapping));
  input.transfer_kv_infos.emplace_back(std::move(info));
  input.sample_sequence_ids = {"descriptor-after-transfer-mappings"};
  auto expect_mapping = [&](const ForwardInput& restored) {
    ASSERT_EQ(restored.transfer_kv_infos.size(), 1u);
    const TransferKVInfo& out = restored.transfer_kv_infos.front();
    EXPECT_EQ(out.request_id, "shared-prefix-and-cursor");
    ASSERT_EQ(out.mappings.size(), 1u);
    const KVTransferMapping& actual = out.mappings.front();
    EXPECT_EQ(actual.local_ids, (std::vector<uint64_t>{7, 8}));
    EXPECT_EQ(actual.remote_ids, (std::vector<uint64_t>{17, 18}));
    EXPECT_EQ(actual.remote_shared_num, 815u);
    EXPECT_EQ(actual.local_block_start, 203u);
    EXPECT_EQ(restored.sample_sequence_ids, input.sample_sequence_ids);
  };

  proto::PackedForwardInput packed;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed));
  uint64_t tag = 0;
  ASSERT_GE(packed.payload().size(), sizeof(tag));
  std::memcpy(&tag, packed.payload().data(), sizeof(tag));
  EXPECT_EQ(tag, 0x584C4C4D0000000DULL);
  ForwardInput lazy = stage_packed_input_on_cpu(packed);
  ForwardInput unpacked;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(
      lazy,
      torch::Device(torch::kCPU),
      torch::kFloat32,
      unpacked,
      /*materialize_device_buffer=*/false));
  expect_mapping(unpacked);

  bool creator = false;
  const std::string name =
      ForwardSharedMemoryManager::create_unique_name("transfer_mapping_origin",
                                                     /*dp_group=*/0,
                                                     ForwardType::RAW_INPUT,
                                                     /*rank=*/0);
  ForwardSharedMemoryManager writer(
      name, 1 << 20, creator, ForwardType::RAW_INPUT);
  ForwardSharedMemoryManager reader(
      name, 1 << 20, creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(writer.input_write(input));
  ForwardInput restored;
  // Read directly with contiguous buffering disabled by the fixture. The
  // deferred worker path also allocates NPU-pinned payload storage.
  ASSERT_TRUE(reader.input_read(restored, torch::Device(torch::kCPU)));
  expect_mapping(restored);
}

TEST_F(ConstraintSnapshotTransportTest,
       RawInputWithoutDestinationOriginsIsRejected) {
  ForwardInput input;
  proto::PackedForwardInput packed;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed));
  const uint64_t legacy_tag = 0x584C4C4D00000009ULL;
  ASSERT_GE(packed.payload().size(), sizeof(legacy_tag));
  std::memcpy(
      packed.mutable_payload()->data(), &legacy_tag, sizeof(legacy_tag));
  ForwardInput lazy = stage_packed_input_on_cpu(packed);
  EXPECT_DEATH(
      {
        ForwardInput restored;
        detail::unpack_from_input_host_buffer(
            lazy,
            torch::Device(torch::kCPU),
            torch::kFloat32,
            restored,
            /*materialize_device_buffer=*/false);
      },
      "incompatible raw input format");
}

TEST_F(ConstraintSnapshotTransportTest,
       PermanentConstraintRejectionIsTerminal) {
  EXPECT_TRUE(is_permanent_rejection(kDecodeConstraintRejectedStatusCode));
  EXPECT_TRUE(is_permanent_rejection(kDecodeAddNewPromptTooLongStatusCode));
  EXPECT_FALSE(is_permanent_rejection(404));
  EXPECT_FALSE(is_permanent_rejection(500));
}

}  // namespace
}  // namespace xllm
