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

/* SeedSequence algorithm adapted from NumPy (MIT):
Copyright (c) 2015 Melissa E. O'Neill
Copyright (c) 2019 NumPy Developers
https://github.com/numpy/numpy/blob/v2.3.0/numpy/random/bit_generator.pyx

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>

#include "framework/tokenizer/tokenizers/tokenizers.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_engram.h"

namespace xllm::layer {
namespace {
// NumPy SeedSequence constants from numpy/random/bit_generator.pyx.
// Keep the upstream INIT_A/B, MULT_A/B and MIX_MULT_L/R names recognizable:
// A mixes input entropy; B expands the mixed pool into output seed words.
constexpr uint32_t kSeedInitA = 0x43b0d7e5U;
constexpr uint32_t kSeedMultA = 0x931e8875U;
constexpr uint32_t kSeedInitB = 0x8b51f9ddU;
constexpr uint32_t kSeedMultB = 0x58f38dedU;
constexpr uint32_t kSeedMixMultL = 0xca01f9ddU;
constexpr uint32_t kSeedMixMultR = 0x4973f715U;
constexpr int32_t kSeedWordBits = std::numeric_limits<uint32_t>::digits;
constexpr int32_t kSeedXorShift = kSeedWordBits / 2;
// NumPy's default entropy pool is four uint32 words (128 bits).
constexpr int32_t kSeedPoolWords = 4;
// PCG64 needs four uint64 words: a 128-bit state and a 128-bit stream seed.
constexpr int32_t kPcgSeedWords = 4;
constexpr int32_t kSeedWordsPerOutput =
    std::numeric_limits<uint64_t>::digits / kSeedWordBits;
constexpr int32_t kSeedOutputWords = kPcgSeedWords * kSeedWordsPerOutput;

// NumPy SeedSequence's single-integer seed protocol, followed by PCG64
// XSL-RR and Lemire bounded sampling. These exact draws are part of the
// trained Engram hash, not interchangeable random initialization.
std::array<uint64_t, kPcgSeedWords> seed_words(uint64_t seed) {
  uint32_t hash_constant = kSeedInitA;
  auto mix_hash = [&hash_constant](uint32_t value) {
    value ^= hash_constant;
    hash_constant *= kSeedMultA;
    value *= hash_constant;
    return value ^ (value >> kSeedXorShift);
  };
  std::array<uint32_t, kSeedPoolWords> pool;
  for (int32_t i = 0; i < kSeedPoolWords; ++i) {
    pool[i] = mix_hash(static_cast<uint32_t>(seed));
    seed >>= kSeedWordBits;
  }
  for (int32_t source = 0; source < kSeedPoolWords; ++source) {
    for (int32_t target = 0; target < kSeedPoolWords; ++target) {
      if (source == target) {
        continue;
      }
      uint32_t value =
          kSeedMixMultL * pool[target] - kSeedMixMultR * mix_hash(pool[source]);
      pool[target] = value ^ (value >> kSeedXorShift);
    }
  }
  std::array<uint64_t, kPcgSeedWords> words{};
  hash_constant = kSeedInitB;
  // Pack consecutive uint32 draws low word first, matching NumPy's uint64
  // generate_state output independently of the host's byte order.
  for (int32_t i = 0; i < kSeedOutputWords; ++i) {
    uint32_t value = pool[i % kSeedPoolWords] ^ hash_constant;
    hash_constant *= kSeedMultB;
    value *= hash_constant;
    value ^= value >> kSeedXorShift;
    words[i / kSeedWordsPerOutput] |=
        static_cast<uint64_t>(value)
        << (kSeedWordBits * (i % kSeedWordsPerOutput));
  }
  return words;
}

torch::Tensor hash_multipliers(const ModelArgs& args) {
  CHECK_GT(args.engram_compressed_vocab_size(), 0)
      << "engram_compressed_vocab_size must be positive";
  // NumPy PCG64 XSL-RR 128/64 constants (not the PCG64DXSM variant):
  // numpy/random/src/pcg64/pcg64.h, PCG_DEFAULT_MULTIPLIER_{HIGH,LOW}.
  constexpr uint64_t kPcgMultiplierHigh = 2549297995355413924ULL;
  constexpr uint64_t kPcgMultiplierLow = 4865540595714422341ULL;
  constexpr int32_t kPcgWordBits = std::numeric_limits<uint64_t>::digits;
  // The top six state bits select a rotation of the 64-bit output word.
  constexpr int32_t kPcgRotationBits = 6;
  constexpr int32_t kPcgRotationShift = 2 * kPcgWordBits - kPcgRotationBits;
  constexpr __uint128_t kPcgMultiplier =
      (static_cast<__uint128_t>(kPcgMultiplierHigh) << kPcgWordBits) |
      kPcgMultiplierLow;
  // Training seeds each layer with 10007 * layer_id, then maps bounded
  // draws to odd hash multipliers via 2 * draw + 1. Preserve both conventions
  // to address the same trained Engram embeddings.
  constexpr uint64_t kEngramLayerSeedMultiplier = 10007ULL;
  constexpr uint64_t kOddMultiplierScale = 2;
  const int64_t layers = args.engram_layer_ids().size();
  const int64_t shifts = args.engram_max_ngram_size();
  auto result = torch::empty({layers, shifts}, torch::kInt64);
  const uint64_t bound = std::numeric_limits<int64_t>::max() /
                         args.engram_compressed_vocab_size() /
                         kOddMultiplierScale;
  // The model's vocabulary uses NumPy's 64-bit bounded-integer path.
  CHECK_GT(bound, std::numeric_limits<uint32_t>::max());
  // Unsigned wraparound computes 2^64 % bound for Lemire rejection sampling.
  const uint64_t rejection_threshold = -bound % bound;
  for (int64_t layer = 0; layer < layers; ++layer) {
    const auto [state_high, state_low, sequence_high, sequence_low] =
        seed_words(kEngramLayerSeedMultiplier * args.engram_layer_ids()[layer]);
    const __uint128_t initial =
        (static_cast<__uint128_t>(state_high) << kPcgWordBits) | state_low;
    // PCG's stream increment must be odd: (sequence << 1) | 1.
    const __uint128_t increment =
        (((static_cast<__uint128_t>(sequence_high) << kPcgWordBits) |
          sequence_low)
         << 1) |
        1;
    __uint128_t state = (increment + initial) * kPcgMultiplier + increment;
    for (int64_t shift = 0; shift < shifts; ++shift) {
      __uint128_t product;
      do {
        state = state * kPcgMultiplier + increment;
        const uint64_t value = static_cast<uint64_t>(state >> kPcgWordBits) ^
                               static_cast<uint64_t>(state);
        const int32_t rotation =
            static_cast<int32_t>(state >> kPcgRotationShift);
        const uint64_t random = std::rotr(value, rotation);
        product = static_cast<__uint128_t>(random) * bound;
      } while (static_cast<uint64_t>(product) < rejection_threshold);
      result.data_ptr<int64_t>()[layer * shifts + shift] = static_cast<int64_t>(
          (product >> kPcgWordBits) * kOddMultiplierScale + 1);
    }
  }
  return result;
}

bool is_prime(int64_t value) {
  if (value < 2) {
    return false;
  }
  if (value % 2 == 0) {
    return value == 2;
  }
  for (int64_t divisor = 3; divisor <= value / divisor; divisor += 2) {
    if (value % divisor == 0) {
      return false;
    }
  }
  return true;
}

torch::Tensor token_map(const ModelArgs& args, const std::string& path) {
  std::ifstream input(path);
  CHECK(input.good()) << "Cannot read Engram tokenizer: " << path;
  auto config =
      nlohmann::json::parse(input, nullptr, /*allow_exceptions=*/false);
  CHECK(!config.is_discarded())
      << "Cannot parse Engram tokenizer JSON: " << path;
  CHECK(config.is_object())
      << "Engram tokenizer JSON must be an object: " << path;
  CHECK(config.contains("model") && config["model"].is_object())
      << "Engram tokenizer JSON requires a model object: " << path;
  const auto& model = config["model"];
  CHECK(model.contains("type") && model["type"].is_string() &&
        model.contains("vocab") &&
        (model["vocab"].is_object() || model["vocab"].is_array()))
      << "Engram tokenizer JSON requires model type and vocabulary: " << path;
  config["normalizer"] = nlohmann::json::parse(R"json({
    "type":"Sequence", "normalizers":[
      {"type":"NFKC"}, {"type":"NFD"}, {"type":"StripAccents"},
      {"type":"Lowercase"},
      {"type":"Replace", "pattern":{"Regex":"[ \t\r\n]+"}, "content":" "},
      {"type":"Replace", "pattern":{"Regex":"^ $"}, "content":"\ue000"},
      {"type":"Strip", "strip_left":true, "strip_right":true},
      {"type":"Replace", "pattern":{"String":"\ue000"}, "content":" "}
    ]})json");
  const std::string json = config.dump();
  std::unique_ptr<void, decltype(&tokenizers_free)> tokenizer(
      tokenizers_new_from_str(json.data(), json.size()), tokenizers_free);
  size_t vocab_size = 0;
  tokenizers_get_vocab_size(tokenizer.get(), &vocab_size);
  CHECK_EQ(vocab_size, args.vocab_size());
  auto result = torch::empty({static_cast<int64_t>(vocab_size)}, torch::kInt32);
  std::unordered_map<std::string, int32_t> compressed;
  compressed.reserve(vocab_size);
  for (uint32_t id = 0; id < vocab_size; ++id) {
    const char* data = nullptr;
    size_t length = 0;
    tokenizers_decode(
        tokenizer.get(), &id, 1, /*skip_special_tokens=*/0, &data, &length);
    const std::string decoded(data, length);
    std::string key;
    if (decoded.find("\xef\xbf\xbd") != std::string::npos) {
      tokenizers_id_to_token(tokenizer.get(), id, &data, &length);
      key.assign(data, length);
    } else {
      const int32_t status = tokenizers_normalize(
          tokenizer.get(), decoded.data(), decoded.size(), &data, &length);
      CHECK_EQ(status, 0) << "Engram tokenizer normalization failed for token "
                          << id << " in " << path << " (status " << status
                          << ": 1=invalid UTF-8, 2=normalizer error)";
      key = length == 0 ? decoded : std::string(data, length);
    }
    const auto [entry, inserted] = compressed.try_emplace(
        std::move(key), static_cast<int32_t>(compressed.size()));
    result.data_ptr<int32_t>()[id] = entry->second;
  }
  CHECK_EQ(compressed.size(), args.engram_compressed_vocab_size());
  return result;
}
}  // namespace

StateDict build_deepseek_v41_hash_state(const ModelArgs& args,
                                        const std::string& tokenizer_path) {
  const int64_t layers = args.engram_layer_ids().size();
  const int64_t cols =
      (args.engram_max_ngram_size() - 1) * args.engram_n_heads();
  CHECK_EQ(layers, args.engram_num_embeddings().size());
  auto primes = torch::empty({layers, cols}, torch::kInt64);
  auto offsets = torch::empty_like(primes);
  // Every head gets the next unused prime, across n-grams and layers.
  int64_t prime = args.engram_vocab_size() - 1;
  for (int64_t layer = 0; layer < layers; ++layer) {
    int64_t offset = 0;
    for (int64_t col = 0; col < cols; ++col) {
      do {
        ++prime;
      } while (!is_prime(prime));
      primes.data_ptr<int64_t>()[layer * cols + col] = prime;
      offsets.data_ptr<int64_t>()[layer * cols + col] = offset;
      offset += prime;
    }
    CHECK_LE(offset, args.engram_num_embeddings()[layer]);
  }
  return StateDict({{"token_map", token_map(args, tokenizer_path)},
                    {"multipliers", hash_multipliers(args)},
                    {"primes", primes},
                    {"offsets", offsets}});
}
}  // namespace xllm::layer
