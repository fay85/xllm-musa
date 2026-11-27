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

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/metrics.h"
#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence.h"

namespace xllm {
namespace {

class StopAwareTokenizer final : public Tokenizer {
 public:
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override {
    std::string text;
    for (const int32_t token_id : ids) {
      if (token_id == kStopTokenId) {
        if (!skip_special_tokens) {
          text += "<|observation|>";
        }
        continue;
      }
      text.push_back(static_cast<char>(token_id));
    }
    return text;
  }

  static constexpr int32_t kStopTokenId = 1000;
};

class DelayedStopAwareTokenizer final : public Tokenizer {
 public:
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return tokenizer_.decode(ids, skip_special_tokens);
  }

 private:
  StopAwareTokenizer tokenizer_;
};

class ByteFallbackTokenizer final : public Tokenizer {
 public:
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override {
    if (ids.size() == 2 && ids[0] == kFirstByteTokenId &&
        ids[1] == kLastByteTokenId) {
      return "\xE4\xBD\xA0";
    }
    if (ids.size() == 1 &&
        (ids[0] == kFirstByteTokenId || ids[0] == kLastByteTokenId)) {
      return "\xEF\xBF\xBD";
    }
    return StopAwareTokenizer().decode(ids, skip_special_tokens);
  }

  static constexpr int32_t kFirstByteTokenId = 1001;
  static constexpr int32_t kLastByteTokenId = 1002;
};

class SequenceStopOutputTest : public ::testing::Test {
 protected:
  void initialize(size_t max_generated_tokens,
                  const std::unordered_set<int32_t>& stop_tokens,
                  const std::vector<std::vector<int32_t>>& stop_sequences = {},
                  const std::vector<int32_t>& prompt_tokens = {'P'},
                  bool include_stop_str_in_output = false,
                  int32_t eos_token = -1,
                  bool skip_special_tokens = false,
                  bool logprobs = false,
                  bool enable_schedule_overlap = false) {
    stopping_checker_ = StoppingChecker(max_generated_tokens,
                                        /*max_context_len=*/0,
                                        eos_token,
                                        /*ignore_eos=*/false,
                                        stop_tokens,
                                        stop_sequences);

    SequenceParams params;
    params.seq_capacity = 16;
    params.echo = false;
    params.skip_special_tokens = skip_special_tokens;
    params.include_stop_str_in_output = include_stop_str_in_output;
    params.logprobs = logprobs;
    params.streaming = false;
    params.enable_schedule_overlap = enable_schedule_overlap;
    params.bos_token_id = 0;
    params.request_id = "stop_output_test";
    sampling_param_.logprobs = logprobs;
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;

    IncrementalDecoder decoder(
        /*prompt=*/"P",
        /*num_prompt_tokens=*/prompt_tokens.size(),
        /*echo=*/params.echo,
        /*skip_special_tokens=*/params.skip_special_tokens);
    sequence_ = std::make_unique<Sequence>(/*index=*/0,
                                           prompt_tokens,
                                           /*input_embedding=*/torch::Tensor(),
                                           /*mm_data=*/MMData(),
                                           decoder,
                                           params);
    sequence_->kv_state().set_kv_cache_tokens_num(prompt_tokens.size());
  }

  void append_token(int32_t token_id) {
    sequence_->append_token(Token(token_id));
  }

  void append_token(int32_t token_id, float logprob) {
    Token token(token_id);
    token.logprob = logprob;
    sequence_->append_token(token);
  }

  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
  std::unique_ptr<Sequence> sequence_;
  StopAwareTokenizer tokenizer_;
};

TEST_F(SequenceStopOutputTest,
       DecodeStreamCompletesPrefillBytesWithoutRepeatingMetadata) {
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{},
             /*prompt_tokens=*/{'P'},
             /*include_stop_str_in_output=*/false,
             /*eos_token=*/-1,
             /*skip_special_tokens=*/false,
             /*logprobs=*/true);
  ByteFallbackTokenizer tokenizer;
  sequence_->enable_checking_prefill_token(/*prefill_text_withheld=*/false);
  EXPECT_FALSE(sequence_->generate_streaming_output(1, tokenizer).has_value());
  append_token(ByteFallbackTokenizer::kFirstByteTokenId, -0.1f);
  append_token(ByteFallbackTokenizer::kLastByteTokenId, -0.2f);

  auto output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer);
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->text, "\xE4\xBD\xA0");
  ASSERT_EQ(output->token_ids.size(), 1);
  EXPECT_EQ(output->token_ids[0], ByteFallbackTokenizer::kLastByteTokenId);
  ASSERT_TRUE(output->logprobs.has_value());
  ASSERT_EQ(output->logprobs->size(), 1);
  EXPECT_EQ(output->logprobs->front().token_id,
            ByteFallbackTokenizer::kLastByteTokenId);
  EXPECT_EQ(sequence_->num_committed_generated_tokens(), 2);
}

TEST_F(SequenceStopOutputTest, DecodeReleasesTextWithheldByPrefillStopWindow) {
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{{'X', 'Y'}},
             /*prompt_tokens=*/{'P'},
             /*include_stop_str_in_output=*/false,
             /*eos_token=*/-1,
             /*skip_special_tokens=*/false,
             /*logprobs=*/true,
             /*enable_schedule_overlap=*/true);
  append_token(-1);
  Token first_token('A');
  first_token.logprob = -0.1f;
  sequence_->update_last_step_token(first_token);
  const bool prefill_text_withheld = sequence_->prefill_text_withheld();
  ASSERT_TRUE(prefill_text_withheld);
  append_token(-1);
  EXPECT_GT(sequence_->num_tokens(), sequence_->num_valid_tokens());
  EXPECT_EQ(sequence_->prefill_text_withheld(), prefill_text_withheld);
  // Decode consumes P's flag even when its physical token count includes a
  // pending overlap slot. The placeholder must not release P's stop suffix.
  sequence_->enable_checking_prefill_token(prefill_text_withheld);
  EXPECT_FALSE(
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_)
          .has_value());
  Token second_token('B');
  second_token.logprob = -0.2f;
  sequence_->update_last_step_token(second_token);
  append_token(-1);
  auto output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->text, "A");
  EXPECT_EQ(output->token_ids, (std::vector<int32_t>{'B'}));
  ASSERT_TRUE(output->logprobs.has_value());
  ASSERT_EQ(output->logprobs->size(), 1u);
  EXPECT_EQ(output->logprobs->front().token_id, 'B');
  Token third_token('C');
  third_token.logprob = -0.3f;
  sequence_->update_last_step_token(third_token);
  sequence_->finish();
  auto terminal = sequence_->generate_streaming_output(4, tokenizer_);
  ASSERT_TRUE(terminal.has_value());
  EXPECT_EQ(terminal->text, "BC");
  EXPECT_EQ(terminal->token_ids, (std::vector<int32_t>{'C'}));
}

TEST_F(SequenceStopOutputTest, EmbeddingsDoNotRecordDetokenizationLatency) {
  initialize(/*max_generated_tokens=*/8, /*stop_tokens=*/{});
  sampling_param_.is_embeddings = true;
  const std::vector<float> values = {1.0f, 2.0f, 3.0f};
  sequence_->update_embeddings(torch::tensor(values));
  const double latency_before =
      COUNTER_VALUE(detokenization_latency_seconds_non_stream);

  SequenceOutput output = sequence_->generate_output(tokenizer_);
  ASSERT_TRUE(output.embeddings.has_value());
  EXPECT_EQ(output.embeddings.value(), values);
  EXPECT_EQ(output.index, 0u);
  EXPECT_FALSE(output.mm_embeddings.has_value());
  EXPECT_TRUE(output.text.empty());
  EXPECT_TRUE(output.token_ids.empty());

  EXPECT_EQ(COUNTER_VALUE(detokenization_latency_seconds_non_stream),
            latency_before);
}

TEST_F(SequenceStopOutputTest, MmEmbeddingsDoNotRecordDetokenizationLatency) {
  initialize(/*max_generated_tokens=*/8, /*stop_tokens=*/{});
  sampling_param_.is_embeddings = true;
  const torch::Tensor embedding = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
  sequence_->update_mm_embeddings({embedding});
  const double latency_before =
      COUNTER_VALUE(detokenization_latency_seconds_non_stream);

  SequenceOutput output = sequence_->generate_output(tokenizer_);
  ASSERT_TRUE(output.mm_embeddings.has_value());
  ASSERT_EQ(output.mm_embeddings->size(), 1u);
  EXPECT_TRUE(torch::equal(output.mm_embeddings->front().embedding, embedding));
  EXPECT_EQ(output.index, 0u);
  EXPECT_FALSE(output.embeddings.has_value());
  EXPECT_TRUE(output.text.empty());
  EXPECT_TRUE(output.token_ids.empty());

  EXPECT_EQ(COUNTER_VALUE(detokenization_latency_seconds_non_stream),
            latency_before);
}

TEST_F(SequenceStopOutputTest, NonStreamingTextRecordsDetokenizationLatency) {
  initialize(/*max_generated_tokens=*/2, /*stop_tokens=*/{});
  append_token('A');
  append_token('B');
  ASSERT_TRUE(sequence_->finished());
  const DelayedStopAwareTokenizer delayed_tokenizer;
  const double latency_before =
      COUNTER_VALUE(detokenization_latency_seconds_non_stream);

  SequenceOutput output = sequence_->generate_output(delayed_tokenizer);

  EXPECT_GT(COUNTER_VALUE(detokenization_latency_seconds_non_stream),
            latency_before);
  EXPECT_EQ(output.text, "AB");
  EXPECT_EQ(output.token_ids, (std::vector<int32_t>{'A', 'B'}));
  EXPECT_EQ(output.index, 0u);
  ASSERT_TRUE(output.finish_reason.has_value());
  EXPECT_EQ(output.finish_reason.value(), "length");
  EXPECT_FALSE(output.embeddings.has_value());
  EXPECT_FALSE(output.mm_embeddings.has_value());
}

TEST_F(SequenceStopOutputTest, NonStreamingExcludesStopTokenFromText) {
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{StopAwareTokenizer::kStopTokenId});
  append_token('A');
  EXPECT_FALSE(sequence_->finished());
  append_token(StopAwareTokenizer::kStopTokenId);
  ASSERT_TRUE(sequence_->finished());

  SequenceOutput output = sequence_->generate_output(tokenizer_);

  EXPECT_EQ(output.text, "A");
  ASSERT_TRUE(output.finish_reason.has_value());
  EXPECT_EQ(output.finish_reason.value(), "stop");
  ASSERT_EQ(output.token_ids.size(), 2);
  EXPECT_EQ(output.token_ids[0], 'A');
  EXPECT_EQ(output.token_ids[1], StopAwareTokenizer::kStopTokenId);
}

TEST_F(SequenceStopOutputTest, StreamingDoesNotEmitStopTokenDelta) {
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{StopAwareTokenizer::kStopTokenId});
  append_token('A');
  EXPECT_FALSE(sequence_->finished());

  auto first_output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);
  ASSERT_TRUE(first_output.has_value());
  EXPECT_EQ(first_output->text, "A");

  append_token(StopAwareTokenizer::kStopTokenId);
  ASSERT_TRUE(sequence_->finished());
  auto stop_output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);

  ASSERT_TRUE(stop_output.has_value());
  EXPECT_TRUE(stop_output->text.empty());
  ASSERT_EQ(stop_output->token_ids.size(), 1);
  EXPECT_EQ(stop_output->token_ids[0], StopAwareTokenizer::kStopTokenId);
}

TEST_F(SequenceStopOutputTest, NonStreamingIncludesStopTokenWhenRequested) {
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{StopAwareTokenizer::kStopTokenId},
             /*stop_sequences=*/{},
             /*prompt_tokens=*/{'P'},
             /*include_stop_str_in_output=*/true);
  append_token('A');
  append_token(StopAwareTokenizer::kStopTokenId);
  ASSERT_TRUE(sequence_->finished());

  SequenceOutput output = sequence_->generate_output(tokenizer_);

  EXPECT_EQ(output.text, "A<|observation|>");
  ASSERT_EQ(output.token_ids.size(), 2);
  EXPECT_EQ(output.token_ids[0], 'A');
  EXPECT_EQ(output.token_ids[1], StopAwareTokenizer::kStopTokenId);
}

TEST_F(SequenceStopOutputTest, StreamingIncludesStopTokenWhenRequested) {
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{StopAwareTokenizer::kStopTokenId},
             /*stop_sequences=*/{},
             /*prompt_tokens=*/{'P'},
             /*include_stop_str_in_output=*/true);
  append_token('A');
  EXPECT_FALSE(sequence_->finished());

  auto first_output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);
  ASSERT_TRUE(first_output.has_value());
  EXPECT_EQ(first_output->text, "A");

  append_token(StopAwareTokenizer::kStopTokenId);
  ASSERT_TRUE(sequence_->finished());
  auto stop_output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);

  ASSERT_TRUE(stop_output.has_value());
  EXPECT_EQ(stop_output->text, "<|observation|>");
}

TEST_F(SequenceStopOutputTest, EosOutputUsesIncludeStopStringSetting) {
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{},
             /*prompt_tokens=*/{'P'},
             /*include_stop_str_in_output=*/false,
             /*eos_token=*/StopAwareTokenizer::kStopTokenId);
  append_token('A');
  append_token(StopAwareTokenizer::kStopTokenId);
  ASSERT_TRUE(sequence_->finished());
  EXPECT_EQ(sequence_->generate_output(tokenizer_).text, "A");

  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{},
             /*prompt_tokens=*/{'P'},
             /*include_stop_str_in_output=*/true,
             /*eos_token=*/StopAwareTokenizer::kStopTokenId);
  append_token('A');
  append_token(StopAwareTokenizer::kStopTokenId);
  ASSERT_TRUE(sequence_->finished());
  EXPECT_EQ(sequence_->generate_output(tokenizer_).text, "A<|observation|>");
}

TEST_F(SequenceStopOutputTest, SkipSpecialTokensRemainsIndependent) {
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{StopAwareTokenizer::kStopTokenId},
             /*stop_sequences=*/{},
             /*prompt_tokens=*/{'P'},
             /*include_stop_str_in_output=*/true,
             /*eos_token=*/-1,
             /*skip_special_tokens=*/true);
  append_token('A');
  append_token(StopAwareTokenizer::kStopTokenId);
  ASSERT_TRUE(sequence_->finished());

  SequenceOutput output = sequence_->generate_output(tokenizer_);

  EXPECT_EQ(output.text, "A");
  ASSERT_EQ(output.token_ids.size(), 2);
  EXPECT_EQ(output.token_ids[0], 'A');
  EXPECT_EQ(output.token_ids[1], StopAwareTokenizer::kStopTokenId);
}

TEST_F(SequenceStopOutputTest, ExcludedStopTokenKeepsLogprob) {
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{StopAwareTokenizer::kStopTokenId},
             /*stop_sequences=*/{},
             /*prompt_tokens=*/{'P'},
             /*include_stop_str_in_output=*/false,
             /*eos_token=*/-1,
             /*skip_special_tokens=*/false,
             /*logprobs=*/true);
  append_token('A', -0.1f);
  append_token(StopAwareTokenizer::kStopTokenId, -0.2f);
  ASSERT_TRUE(sequence_->finished());

  SequenceOutput output = sequence_->generate_output(tokenizer_);

  EXPECT_EQ(output.text, "A");
  ASSERT_TRUE(output.logprobs.has_value());
  ASSERT_EQ(output.logprobs->size(), 2);
  EXPECT_EQ(output.logprobs->back().token_id, StopAwareTokenizer::kStopTokenId);
  EXPECT_FLOAT_EQ(output.logprobs->back().logprob, -0.2f);
}

TEST_F(SequenceStopOutputTest, ExcludesEntireMatchedStopSequence) {
  const std::vector<int32_t> stop_sequence = {'<', 'S', '>'};
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{stop_sequence});
  append_token('A');
  append_token('<');
  append_token('S');
  append_token('>');
  ASSERT_TRUE(sequence_->finished());

  SequenceOutput output = sequence_->generate_output(tokenizer_);

  EXPECT_EQ(output.text, "A");
  ASSERT_EQ(output.token_ids.size(), 4);
  EXPECT_EQ(output.token_ids[1], '<');
  EXPECT_EQ(output.token_ids[2], 'S');
  EXPECT_EQ(output.token_ids[3], '>');
  ASSERT_TRUE(output.finish_reason.has_value());
  EXPECT_EQ(output.finish_reason.value(), "stop");
}

TEST_F(SequenceStopOutputTest, IncludesEntireStopSequenceWhenRequested) {
  const std::vector<int32_t> stop_sequence = {'<', 'S', '>'};
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{stop_sequence},
             /*prompt_tokens=*/{'P'},
             /*include_stop_str_in_output=*/true);
  append_token('A');
  append_token('<');
  append_token('S');
  append_token('>');
  ASSERT_TRUE(sequence_->finished());

  SequenceOutput output = sequence_->generate_output(tokenizer_);

  EXPECT_EQ(output.text, "A<S>");
  ASSERT_TRUE(output.finish_reason.has_value());
  EXPECT_EQ(output.finish_reason.value(), "stop");
}

TEST_F(SequenceStopOutputTest, StreamingBuffersPotentialStopSequence) {
  const std::vector<int32_t> stop_sequence = {'<', 'S', '>'};
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{stop_sequence});
  append_token('A');
  append_token('<');
  append_token('S');
  EXPECT_FALSE(sequence_->finished());

  auto prefix_output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);
  ASSERT_TRUE(prefix_output.has_value());
  EXPECT_EQ(prefix_output->text, "A");
  ASSERT_EQ(prefix_output->token_ids.size(), 3);
  EXPECT_EQ(prefix_output->token_ids[0], 'A');
  EXPECT_EQ(prefix_output->token_ids[1], '<');
  EXPECT_EQ(prefix_output->token_ids[2], 'S');

  append_token('>');
  ASSERT_TRUE(sequence_->finished());
  auto stop_output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);

  ASSERT_TRUE(stop_output.has_value());
  EXPECT_TRUE(stop_output->text.empty());
  ASSERT_EQ(stop_output->token_ids.size(), 1);
  EXPECT_EQ(stop_output->token_ids[0], '>');
}

TEST_F(SequenceStopOutputTest, StreamingDoesNotBufferStopSequenceWhenIncluded) {
  const std::vector<int32_t> stop_sequence = {'<', 'S', '>'};
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{stop_sequence},
             /*prompt_tokens=*/{'P'},
             /*include_stop_str_in_output=*/true);
  append_token('A');
  append_token('<');
  append_token('S');
  EXPECT_FALSE(sequence_->finished());

  auto prefix_output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);
  ASSERT_TRUE(prefix_output.has_value());
  EXPECT_EQ(prefix_output->text, "A<S");

  append_token('>');
  ASSERT_TRUE(sequence_->finished());
  auto stop_output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);

  ASSERT_TRUE(stop_output.has_value());
  EXPECT_EQ(stop_output->text, ">");
}

TEST_F(SequenceStopOutputTest, StopSequenceAcrossPromptKeepsPromptTokens) {
  const std::vector<int32_t> stop_sequence = {'<', 'S', '>'};
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{stop_sequence},
             /*prompt_tokens=*/{'<', 'S'});
  append_token('>');
  ASSERT_TRUE(sequence_->finished());

  SequenceOutput output = sequence_->generate_output(tokenizer_);

  EXPECT_TRUE(output.text.empty());
  ASSERT_EQ(output.token_ids.size(), 1);
  EXPECT_EQ(output.token_ids[0], '>');
  ASSERT_TRUE(output.finish_reason.has_value());
  EXPECT_EQ(output.finish_reason.value(), "stop");
}

TEST_F(SequenceStopOutputTest, LengthFinishKeepsLastGeneratedToken) {
  initialize(/*max_generated_tokens=*/2, /*stop_tokens=*/{});
  append_token('A');
  EXPECT_FALSE(sequence_->finished());
  append_token('B');
  ASSERT_TRUE(sequence_->finished());

  SequenceOutput output = sequence_->generate_output(tokenizer_);

  EXPECT_EQ(output.text, "AB");
  ASSERT_TRUE(output.finish_reason.has_value());
  EXPECT_EQ(output.finish_reason.value(), "length");
}

TEST_F(SequenceStopOutputTest, StopTokenTakesPrecedenceAtLengthLimit) {
  initialize(/*max_generated_tokens=*/2,
             /*stop_tokens=*/{StopAwareTokenizer::kStopTokenId});
  append_token('A');
  EXPECT_FALSE(sequence_->finished());
  append_token(StopAwareTokenizer::kStopTokenId);
  ASSERT_TRUE(sequence_->finished());

  SequenceOutput output = sequence_->generate_output(tokenizer_);

  EXPECT_EQ(output.text, "A");
  ASSERT_TRUE(output.finish_reason.has_value());
  EXPECT_EQ(output.finish_reason.value(), "stop");
  ASSERT_EQ(output.token_ids.size(), 2);
  EXPECT_EQ(output.token_ids[1], StopAwareTokenizer::kStopTokenId);
}

TEST_F(SequenceStopOutputTest, StopSequenceTakesPrecedenceAtLengthLimit) {
  const std::vector<int32_t> stop_sequence = {'A', 'B'};
  initialize(/*max_generated_tokens=*/2,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{stop_sequence});
  append_token('A');
  EXPECT_FALSE(sequence_->finished());
  append_token('B');
  ASSERT_TRUE(sequence_->finished());

  SequenceOutput output = sequence_->generate_output(tokenizer_);

  EXPECT_TRUE(output.text.empty());
  ASSERT_TRUE(output.finish_reason.has_value());
  EXPECT_EQ(output.finish_reason.value(), "stop");
  ASSERT_EQ(output.token_ids.size(), 2);
  EXPECT_EQ(output.token_ids[0], 'A');
  EXPECT_EQ(output.token_ids[1], 'B');
}

TEST_F(SequenceStopOutputTest, LengthFinishFlushesStreamingStopBuffer) {
  const std::vector<int32_t> stop_sequence = {'A', 'B', 'C'};
  initialize(/*max_generated_tokens=*/2,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{stop_sequence});
  append_token('A');
  EXPECT_FALSE(sequence_->finished());
  append_token('B');
  ASSERT_TRUE(sequence_->finished());

  auto output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->text, "AB");
}

TEST_F(SequenceStopOutputTest, ManualFinishKeepsLastGeneratedToken) {
  initialize(/*max_generated_tokens=*/8, /*stop_tokens=*/{});
  append_token('A');
  sequence_->finish();

  SequenceOutput output = sequence_->generate_output(tokenizer_);

  EXPECT_EQ(output.text, "A");
  ASSERT_TRUE(output.finish_reason.has_value());
  EXPECT_EQ(output.finish_reason.value(), "stop");
}

// Regression test for the PD decode-side prefill-token check: decode() looks
// one token ahead of the slice it is given, while get_decodable_token_count()
// holds the trailing tokens back as a potential multi-token stop suffix. That
// slice can therefore end exactly at the decoder offset, which used to abort on
// slice.h:69.
TEST_F(SequenceStopOutputTest, PrefillTokenCheckToleratesWithheldStopSuffix) {
  const std::vector<int32_t> stop_sequence = {'X', 'Y'};
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{stop_sequence});
  append_token('A');
  ASSERT_FALSE(sequence_->finished());
  sequence_->enable_checking_prefill_token(/*prefill_text_withheld=*/false);

  // The only generated token is withheld, so the decoder gets a slice of length
  // num_prompt_tokens and has no token to inspect. Prefill already emitted the
  // token metadata, so there is no new output.
  auto output =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);

  EXPECT_FALSE(output.has_value());
}

// The deferred check must run once the token becomes visible. Clearing the flag
// without inspecting the token would drop the skip and re-emit the text of the
// prefill token, so only 'B' may be emitted here.
TEST_F(SequenceStopOutputTest, PrefillTokenCheckRunsOnceTokenIsVisible) {
  const std::vector<int32_t> stop_sequence = {'X', 'Y'};
  initialize(/*max_generated_tokens=*/8,
             /*stop_tokens=*/{},
             /*stop_sequences=*/{stop_sequence});
  append_token('A');
  sequence_->enable_checking_prefill_token(/*prefill_text_withheld=*/false);

  // Call 1: 'A' is withheld, so the check is deferred and nothing is emitted.
  auto first =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);
  EXPECT_FALSE(first.has_value());

  // Call 2: appending 'B' and 'C' releases 'A', so the deferred check runs and
  // skips the prefill token 'A'.
  append_token('B');
  append_token('C');
  auto second =
      sequence_->generate_streaming_output(sequence_->num_tokens(), tokenizer_);

  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->text, "B");
  EXPECT_EQ(second->token_ids, (std::vector<int32_t>{'B', 'C'}));
}

}  // namespace
}  // namespace xllm
