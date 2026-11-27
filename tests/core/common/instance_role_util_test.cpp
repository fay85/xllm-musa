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

#include "core/common/instance_role_util.h"

#include <gtest/gtest.h>

#include <optional>

#include "core/common/options.h"
#include "core/common/types.h"

namespace xllm {
namespace {

constexpr InstanceRole::RoleType kAllRoles[] = {InstanceRole::DEFAULT,
                                                InstanceRole::PREFILL,
                                                InstanceRole::DECODE,
                                                InstanceRole::MIX};

// Mirrors the ContinuousScheduler::Options accessor shape, whose instance_role
// is a std::optional, without pulling the scheduler target into this CPU-only
// test.
class OptionalRoleOptions final {
 public:
  OptionalRoleOptions(bool enable_disagg_pd,
                      std::optional<InstanceRole> instance_role)
      : enable_disagg_pd_(enable_disagg_pd), instance_role_(instance_role) {}

  bool enable_disagg_pd() const { return enable_disagg_pd_; }
  const std::optional<InstanceRole>& instance_role() const {
    return instance_role_;
  }

 private:
  bool enable_disagg_pd_ = false;
  std::optional<InstanceRole> instance_role_;
};

Options make_options(bool enable_disagg_pd, InstanceRole instance_role) {
  Options options;
  options.enable_disagg_pd(enable_disagg_pd);
  options.instance_role(instance_role);
  return options;
}

TEST(InstanceRoleUtilTest, PrefillRoleAloneIsNotThePdPrefillSide) {
  Options default_options;
  EXPECT_EQ(default_options.instance_role(), InstanceRole::DEFAULT);
  EXPECT_FALSE(is_pd_prefill_role(default_options));

  // Without disaggregated PD the continuous scheduler keeps decode on the same
  // instance whatever --instance_role says, so the role alone must not pass.
  for (const InstanceRole::RoleType role : kAllRoles) {
    const Options options = make_options(
        /*enable_disagg_pd=*/false, role);
    EXPECT_FALSE(is_pd_prefill_role(options)) << InstanceRole(role).to_string();
  }
}

TEST(InstanceRoleUtilTest, PdPrefillSideRequiresBothDisaggPdAndPrefillRole) {
  for (const InstanceRole::RoleType role : kAllRoles) {
    const Options options = make_options(
        /*enable_disagg_pd=*/true, role);
    const bool expected = role == InstanceRole::PREFILL;
    EXPECT_EQ(is_pd_prefill_role(options), expected)
        << InstanceRole(role).to_string();
  }
}

TEST(InstanceRoleUtilTest, OptionalRoleFollowsTheSameMatrix) {
  EXPECT_TRUE(is_pd_prefill_role(OptionalRoleOptions(
      /*enable_disagg_pd=*/true,

      InstanceRole(InstanceRole::PREFILL))));
  EXPECT_TRUE(is_pd_prefill_role(OptionalRoleOptions(
      /*enable_disagg_pd=*/true,

      InstanceRole(InstanceRole::PREFILL))));
  EXPECT_FALSE(is_pd_prefill_role(OptionalRoleOptions(
      /*enable_disagg_pd=*/false,

      InstanceRole(InstanceRole::PREFILL))));
  // An unset role (the scheduler options carry std::optional) is not prefill.
  EXPECT_FALSE(is_pd_prefill_role(OptionalRoleOptions(
      /*enable_disagg_pd=*/true, std::nullopt)));
}

}  // namespace
}  // namespace xllm
