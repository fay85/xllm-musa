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

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/constraint/builtin_json_machine.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence.h"

namespace xllm {
namespace {

class SequenceJsonObjectRestoreTest : public ::testing::Test {
 protected:
  void initialize() {
    grammar_ = std::make_shared<JsonObjectGrammar>(
        std::vector<std::string>{"{", "}", "stop"},
        std::unordered_set<int32_t>{2});
    stopping_checker_ = StoppingChecker(/*max_generated_tokens=*/16,
                                        /*max_context_len=*/0,
                                        /*eos_token=*/-1,
                                        /*ignore_eos=*/false,
                                        /*stop_tokens=*/{},
                                        /*stop_sequences=*/{});

    SequenceParams params;
    params.seq_capacity = 16;
    params.echo = false;
    params.skip_special_tokens = true;
    params.include_stop_str_in_output = false;
    params.logprobs = false;
    params.streaming = false;
    params.enable_schedule_overlap = false;
    params.rec_type = RecType::kNone;
    params.bos_token_id = 0;
    params.request_id = "json_object_restore_test";
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;
    params.constraint_machine = wrap_builtin_machine(grammar_);
    params.json_reasoning_enabled = false;

    IncrementalDecoder decoder(
        /*prompt=*/"P",
        /*num_prompt_tokens=*/1,
        /*echo=*/params.echo,
        /*skip_special_tokens=*/params.skip_special_tokens);
    sequence_ = std::make_unique<Sequence>(/*index=*/0,
                                           std::vector<int32_t>{'P'},
                                           /*input_embedding=*/torch::Tensor(),
                                           /*mm_data=*/MMData(),
                                           decoder,
                                           params);
  }

  std::shared_ptr<const JsonObjectGrammar> grammar_;
  StoppingChecker stopping_checker_;
  RequestSamplingParam sampling_param_;
  std::unique_ptr<Sequence> sequence_;
};

TEST_F(SequenceJsonObjectRestoreTest, FailedRestoreClearsTheStaleLiveState) {
  initialize();
  ASSERT_TRUE(sequence_->json_constraint_state().initialized());

  // An enabled snapshot whose committed history violates the grammar ("}"
  // cannot open a document): the replay rejects it, the request fails, and
  // the stale live matcher must not keep leaking through the constraint
  // channel while the request is being torn down.
  JsonObjectGrammarSnapshot snapshot;
  snapshot.enabled = true;
  snapshot.token_ids = {/*"}" first*/ 1};
  EXPECT_FALSE(sequence_->restore_json_object_state(snapshot));
  EXPECT_TRUE(sequence_->error_status().has_value());
  EXPECT_FALSE(sequence_->json_constraint_state().initialized());
}

TEST_F(SequenceJsonObjectRestoreTest, ValidRestoreKeepsTheStateAttached) {
  initialize();
  ASSERT_TRUE(sequence_->json_constraint_state().initialized());

  JsonObjectGrammarSnapshot snapshot;
  snapshot.enabled = true;
  snapshot.token_ids = {/*"{" opens the object*/ 0};
  EXPECT_TRUE(sequence_->restore_json_object_state(snapshot));
  EXPECT_FALSE(sequence_->error_status().has_value());
  EXPECT_TRUE(sequence_->json_constraint_state().initialized());
}

TEST_F(SequenceJsonObjectRestoreTest, DisabledSnapshotMeansUnconstrained) {
  initialize();
  ASSERT_TRUE(sequence_->json_constraint_state().initialized());

  // A disabled snapshot is the legitimate unconstrained transition: the
  // live state is dropped on purpose, not through a failure.
  JsonObjectGrammarSnapshot snapshot;
  snapshot.enabled = false;
  EXPECT_TRUE(sequence_->restore_json_object_state(snapshot));
  EXPECT_FALSE(sequence_->error_status().has_value());
  EXPECT_FALSE(sequence_->json_constraint_state().initialized());
}

}  // namespace
}  // namespace xllm
