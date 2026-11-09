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

#include "core/framework/config/parallel_config.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(KvSplitTopologyTest, FitsCpGroup) {
  EXPECT_TRUE(kv_split_fits_cp_topology(4, 2, 1));
  EXPECT_TRUE(kv_split_fits_cp_topology(4, 2, 2));
  EXPECT_TRUE(kv_split_fits_cp_topology(4, 2, 4));
}

TEST(KvSplitTopologyTest, FitsDecodeContextParallelInsideTp) {
  EXPECT_TRUE(kv_split_fits_cp_topology(1, 4, 1));
  EXPECT_TRUE(kv_split_fits_cp_topology(1, 4, 2));
  EXPECT_TRUE(kv_split_fits_cp_topology(1, 4, 4));
}

TEST(KvSplitTopologyTest, SpansWholeCpAndTpDomain) {
  EXPECT_TRUE(kv_split_spans_cp_tp(2, 4, 8));
  EXPECT_FALSE(kv_split_spans_cp_tp(2, 4, 4));
  EXPECT_TRUE(kv_split_spans_cp_tp(1, 4, 4));
}

TEST(KvSplitTopologyTest, RejectsSplitThatCannotMapOntoTopology) {
  EXPECT_FALSE(kv_split_fits_cp_topology(1, 1, 2));
  EXPECT_FALSE(kv_split_fits_cp_topology(1, 4, 3));
  EXPECT_FALSE(kv_split_fits_cp_topology(2, 2, 3));
  EXPECT_FALSE(kv_split_fits_cp_topology(2, 4, 8));
}

TEST(KvSplitTopologyTest, RejectsNonPositiveDimensions) {
  EXPECT_FALSE(kv_split_fits_cp_topology(0, 4, 2));
  EXPECT_FALSE(kv_split_fits_cp_topology(1, 0, 1));
  EXPECT_FALSE(kv_split_fits_cp_topology(1, 4, 0));
  EXPECT_FALSE(kv_split_spans_cp_tp(0, 4, 8));
  EXPECT_FALSE(kv_split_spans_cp_tp(2, 4, 0));
}

}  // namespace
}  // namespace xllm
