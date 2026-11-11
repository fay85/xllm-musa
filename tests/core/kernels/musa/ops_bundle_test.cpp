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

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "core/kernels/musa/gdn_ops.h"
#include "core/kernels/musa/musa_aot_config.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

class MusaOpsBundleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    source_ = get_musa_ops_root();
    std::string pattern =
        (std::filesystem::temp_directory_path() / "xllm-musa-ops.XXXXXX")
            .string();
    const char* directory = mkdtemp(pattern.data());
    ASSERT_NE(directory, nullptr);
    root_ = directory;
    std::filesystem::copy(
        source_, root_, std::filesystem::copy_options::recursive);
  }

  void TearDown() override {
    if (!root_.empty()) {
      std::filesystem::remove_all(root_);
    }
  }

  std::filesystem::path source_;
  std::string root_;
};

TEST_F(MusaOpsBundleTest, AcceptsMatchingBundleAndRejectsDamagedArtifacts) {
  for (const char* uri : aot::kOperatorUris) {
    EXPECT_TRUE(is_musa_operator_built(uri));
  }
  constexpr const char* kLegacyGdnUri = "mate_gdn_prefill_hq16_hv48_bf16";
  EXPECT_FALSE(is_musa_operator_built(kLegacyGdnUri));
  EXPECT_DEATH(get_module(kLegacyGdnUri),
               "operator URI is not part of this build");
  EXPECT_DEATH(
      ([] {
        CHECK_EQ(setenv("XLLM_MATE_GDN_PREFILL_SIMPLE", "1", /*overwrite=*/1),
                 0);
        MateGatedDeltaRulePrefillParams params;
        // Unsupported dispatch must fail before normalizing CPU Q/K or
        // loading a device operator, even with no MUSA devices available.
        params.q = torch::zeros({1, 1, 16, 128}, torch::kBFloat16);
        params.k = params.q.clone();
        params.v = torch::zeros({1, 1, 48, 128}, torch::kBFloat16);
        params.g = torch::zeros({1, 1, 48});
        params.beta = torch::ones({1, 1, 48});
        mate_gated_delta_rule_prefill(params);
      }()),
      "Unsupported MUSA GDN prefill configuration");

  // Validate in a child so the parent's verification cache remains empty.
  EXPECT_EXIT(
      {
        validate_musa_ops_bundle(root_);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0),
      "");

  for (const std::string_view suffix : {".so", ".o", ".mubin", ".json"}) {
    const auto artifact =
        std::find_if(aot::kArtifacts.begin(),
                     aot::kArtifacts.end(),
                     [suffix](const aot::Artifact& entry) {
                       return std::string_view(entry.path).ends_with(suffix);
                     });
    ASSERT_NE(artifact, aot::kArtifacts.end());
    SCOPED_TRACE(artifact->path);
    const auto path = std::filesystem::path(root_) / artifact->path;
    std::ofstream(path, std::ios::binary | std::ios::trunc) << "corrupted";
    EXPECT_DEATH(validate_musa_ops_bundle(root_), "artifact checksum mismatch");
    std::filesystem::remove(path);
    EXPECT_DEATH(validate_musa_ops_bundle(root_),
                 "unreadable MUSA operator artifact");
    std::filesystem::copy_file(source_ / artifact->path, path);
  }

  const auto marker = std::filesystem::path(root_) / ".bundle-id";
  std::ofstream(marker) << "stale\n";
  EXPECT_DEATH(validate_musa_ops_bundle(root_), "bundle does not match");
  std::filesystem::remove(marker);
  EXPECT_DEATH(validate_musa_ops_bundle(root_),
               "unreadable MUSA operator bundle marker");
}

}  // namespace
}  // namespace xllm::kernel::musa
