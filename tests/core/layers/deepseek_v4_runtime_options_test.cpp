/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "core/layers/musa/deepseek_v4/deepseek_v4_runtime_options.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <future>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/block/musa/deepseek_v4_prefix_snapshot_options.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_execution_limits.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_moe_policy.h"
#include "core/layers/musa/deepseek_v4/deepseek_v4_pd_reduce_scatter_policy.h"

namespace xllm::layer {
namespace {

class ScopedEnvironment final {
 public:
  explicit ScopedEnvironment(const char* name) : name_(name) {
    const char* previous = std::getenv(name);
    if (previous != nullptr) {
      previous_ = previous;
    }
  }
  ~ScopedEnvironment() {
    if (previous_.has_value()) {
      setenv(name_.c_str(), previous_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }
  void set(const char* value) {
    if (value == nullptr) {
      ASSERT_EQ(unsetenv(name_.c_str()), 0);
    } else {
      ASSERT_EQ(setenv(name_.c_str(), value, 1), 0);
    }
  }

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

TEST(DeepseekV4RuntimeOptionsTest, KeepsMissingEmptyZeroAndOneDistinct) {
  constexpr const char* kName = "XLLM_TEST_DSV4_ENV_VALUE";
  ScopedEnvironment env(kName);
  for (const char* value :
       std::array<const char*, 7>{nullptr, "", "0", "1", "true", "10", "01"}) {
    env.set(value);
    const DeepseekV4EnvValue option(kName);
    EXPECT_EQ(option.present(), value != nullptr);
    EXPECT_EQ(option.is_one(), value != nullptr && std::string(value) == "1");
    EXPECT_EQ(option.is_zero(), value != nullptr && std::string(value) == "0");
    if (value != nullptr) {
      EXPECT_STREQ(option.c_str(), value);
    } else {
      EXPECT_EQ(option.c_str(), nullptr);
    }
  }
}

TEST(DeepseekV4RuntimeOptionsTest, OwnsEnvironmentValueAfterMutation) {
  constexpr const char* kName = "XLLM_TEST_DSV4_ENV_OWNERSHIP";
  ScopedEnvironment env(kName);
  env.set("1");
  const DeepseekV4EnvValue option(kName);
  env.set("replaced");
  env.set(nullptr);
  EXPECT_TRUE(option.is_one());
  EXPECT_STREQ(option.c_str(), "1");
}

TEST(DeepseekV4RuntimeOptionsTest, CachesProcessOptionAcrossCallsAndThreads) {
  ScopedEnvironment env("XLLM_DSV4_MATE_ROUTE_TRACE");
  env.set("1");
  const auto& first = dsv4_runtime_option<DeepseekV4Env::MATE_ROUTE_TRACE>();
  env.set("0");
  EXPECT_TRUE(dsv4_runtime_option<DeepseekV4Env::MATE_ROUTE_TRACE>().is_one());
  std::vector<std::future<const DeepseekV4EnvValue*>> threads;
  threads.reserve(8);
  for (int32_t index = 0; index < 8; ++index) {
    threads.emplace_back(std::async(std::launch::async, [] {
      return &dsv4_runtime_option<DeepseekV4Env::MATE_ROUTE_TRACE>();
    }));
  }
  for (auto& thread : threads) {
    EXPECT_EQ(thread.get(), &first);
  }
}

TEST(DeepseekV4RuntimeOptionsTest, PrefixRatioParsingAndAlignment) {
  EXPECT_EQ(musa::parse_deepseek_v4_prefix_snapshot_percent(nullptr).value(),
            80);
  for (const char* invalid : {"",
                              "0",
                              "100",
                              "-1",
                              "101",
                              "80x",
                              " 80",
                              "99999999999999999999999999"}) {
    EXPECT_FALSE(
        musa::parse_deepseek_v4_prefix_snapshot_percent(invalid).has_value());
  }
  EXPECT_EQ(musa::parse_deepseek_v4_prefix_snapshot_percent("65").value(), 65);
  EXPECT_EQ(musa::deepseek_v4_prefix_snapshot_tokens(20000, 128, 80), 16000);
  EXPECT_EQ(musa::deepseek_v4_prefix_snapshot_tokens(20000, 128, 65), 12928);
  EXPECT_EQ(musa::deepseek_v4_prefix_snapshot_tokens(2500, 128, 80), 1920);
  EXPECT_EQ(musa::deepseek_v4_prefix_snapshot_tokens(0, 128, 80), 0);
  EXPECT_EQ(musa::deepseek_v4_prefix_snapshot_tokens(100, 128, 80), 0);
  EXPECT_EQ(musa::deepseek_v4_prefix_snapshot_tokens(20000, 0, 80), 0);
  EXPECT_EQ(musa::deepseek_v4_prefix_snapshot_tokens(20000, 128, 100), 0);
  const size_t maximum = std::numeric_limits<size_t>::max();
  const size_t boundary =
      musa::deepseek_v4_prefix_snapshot_tokens(maximum, 128, 99);
  EXPECT_LT(boundary, maximum);
  EXPECT_EQ(boundary % 128, 0);
}

TEST(DeepseekV4RuntimeOptionsTest, KeepsQualifiedCommunicationAndVerifyShapes) {
  EXPECT_TRUE(dsv4_pd_reduce_scatter_eligible(4096, 512, 8, true, true, false));
  EXPECT_TRUE(
      dsv4_pd_reduce_scatter_eligible(16384, 2048, 8, true, true, false));
  EXPECT_FALSE(
      dsv4_pd_reduce_scatter_eligible(4096, 1024, 4, true, true, false));
  EXPECT_FALSE(
      dsv4_pd_reduce_scatter_eligible(32768, 4096, 8, true, true, false));
  EXPECT_TRUE(
      dsv4_grouped_fp8_verify_eligible(77, 77, 7, true, true, true, true));
  EXPECT_TRUE(
      dsv4_grouped_fp8_verify_eligible(112, 112, 7, true, true, true, true));
  EXPECT_FALSE(
      dsv4_grouped_fp8_verify_eligible(78, 84, 7, true, true, true, true));
  EXPECT_FALSE(
      dsv4_grouped_fp8_verify_eligible(77, 7, 7, true, true, true, true));
  EXPECT_FALSE(
      dsv4_grouped_fp8_verify_eligible(77, 112, 8, true, true, true, true));
}
}  // namespace
}  // namespace xllm::layer
