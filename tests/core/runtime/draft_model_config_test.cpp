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

#include "core/runtime/draft_model_config.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/model/model_args.h"
#include "core/runtime/options.h"
#include "core/util/json_reader.h"

namespace xllm {
namespace {

#if defined(USE_MLU)
TEST(DraftModelConfigTest, ShortensDFlash2BlockBeforeModelConstruction) {
  ModelArgs args;
  args.model_type("qwen3")
      .num_speculative_tokens(/*value=*/7)
      .dflash2_conv_kernel_size(4)
      .layers_to_capture({1, 3});
  runtime::Options options;
  options.speculative_algorithm(/*value=*/"DFlash2")
      .num_speculative_tokens(
          /*num_speculative_tokens=*/5);

  configure_block_diffusion_draft(args, options);

  EXPECT_EQ(args.model_type(), "DFlash2DraftModel");
  EXPECT_EQ(args.num_speculative_tokens(), 5);
  EXPECT_EQ(args.dummy_token_count(), 6);
  EXPECT_EQ(args.dflash2_conv_kernel_size(), 4);
  EXPECT_TRUE(args.layers_to_capture().empty());
  EXPECT_TRUE(args.requires_eager_execution());
}

TEST(DraftModelConfigTest, AcceptsFullBlockAndConvolutionBoundary) {
  for (const int32_t width : {8, 4}) {
    ModelArgs args;
    args.model_type(/*value=*/"qwen3")
        .num_speculative_tokens(/*value=*/7)
        .dflash2_conv_kernel_size(/*value=*/4);
    runtime::Options options;
    options.speculative_algorithm(/*value=*/"DFlash2")
        .num_speculative_tokens(width - 1);
    configure_block_diffusion_draft(args, options);
    EXPECT_EQ(args.dummy_token_count(), width);
    EXPECT_EQ(args.num_speculative_tokens(), width == 8 ? 7 : 3);
  }
}

TEST(DraftModelConfigDeathTest, RejectsInvalidRuntimeGeometry) {
  struct Geometry {
    int32_t trained;
    int32_t runtime;
    const char* error;
  };
  const std::vector<Geometry> geometries = {
      {8, 1, "requires candidate tokens"},
      {8, 9, "must not exceed"},
      {8, 3, "must cover the convolution kernel"},
      {0, 4, "positive checkpoint token budget"},
      {-1, 4, "positive checkpoint token budget"}};
  for (const Geometry& geometry : geometries) {
    SCOPED_TRACE(geometry.trained);
    SCOPED_TRACE(geometry.runtime);
    ModelArgs args;
    args.model_type("qwen3")
        .num_speculative_tokens(geometry.trained - 1)
        .dflash2_conv_kernel_size(4);
    runtime::Options options;
    options.speculative_algorithm(/*value=*/"DFlash2")
        .num_speculative_tokens(geometry.runtime - 1);
    EXPECT_DEATH(configure_block_diffusion_draft(args, options),
                 geometry.error);
  }
}

TEST(DraftModelConfigTest, ConfiguresNativeV41DSparkGeometry) {
  ModelArgs args;
  args.model_type("deepseek_v41")
      .n_layers(2)
      .dspark_num_layers(3)
      .dspark_n_routed_experts(128)
      .dspark_num_experts_per_tok(3)
      .compress_ratios({2, 1, 0, 0, 0})
      .kv_source_layer_ids({0, 1})
      .index_source_layer_ids({0})
      .engram_layer_ids({1})
      .layers_to_capture({0, 1});
  runtime::Options options;
  options.speculative_algorithm(/*value=*/"DSpark")
      .num_speculative_tokens(/*num_speculative_tokens=*/5);
  configure_block_diffusion_draft(args, options);
  EXPECT_EQ(args.model_type(), "deepseek_v41_dspark");
  EXPECT_EQ(args.n_layers(), 3);
  EXPECT_EQ(args.n_routed_experts(), 128);
  EXPECT_EQ(args.num_experts_per_tok(), 3);
  EXPECT_EQ(args.n_activated_experts(), 3);
  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{0, 0, 0}));
  EXPECT_EQ(args.num_speculative_tokens(), 5);
  EXPECT_EQ(args.dummy_token_count(), 5);
  EXPECT_FALSE(args.requires_eager_execution());
  EXPECT_TRUE(args.draft_query_uses_decode());
  EXPECT_TRUE(args.layers_to_capture().empty());
  EXPECT_TRUE(args.kv_source_layer_ids().empty());
  EXPECT_TRUE(args.index_source_layer_ids().empty());
  EXPECT_TRUE(args.engram_layer_ids().empty());
}

TEST(DraftModelConfigDeathTest, RejectsUnsupportedDraftAlgorithmsOnMlu) {
  for (const std::string& algorithm : {"DFlash", "DSpark"}) {
    ModelArgs args;
    args.model_type("qwen3");
    runtime::Options options;
    options.speculative_algorithm(algorithm);
    EXPECT_DEATH(configure_block_diffusion_draft(args, options),
                 "not supported on mlu");
  }
}
#endif

class DraftCaptureConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    char path[] = "/tmp/xllm-draft-config-XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    model_path_ = path;
  }

  void TearDown() override { std::filesystem::remove_all(model_path_); }

  void write_config(const std::string& json) {
    std::ofstream file(model_path_ + "/config.json");
    ASSERT_TRUE(file.is_open());
    file << json;
    ASSERT_TRUE(file.good());
  }

  std::string model_path_;
};

TEST_F(DraftCaptureConfigTest, ConfiguresTargetWithoutChangingModelGeometry) {
  write_config(R"({"dflash_config":{"target_layer_ids":[1,3]}})");
  ModelArgs args;
  args.model_type("glm_moe_dsa")
      .n_layers(8)
      .num_speculative_tokens(/*value=*/5);
  runtime::Options options;
  options.speculative_algorithm(/*value=*/"DFlash2")
      .draft_model_path(model_path_);
  configure_block_diffusion_target(args, options);
  EXPECT_EQ(args.layers_to_capture(), (std::vector<int32_t>{1, 3}));
  EXPECT_EQ(args.model_type(), "glm_moe_dsa");
  EXPECT_EQ(args.num_speculative_tokens(), 5);
}

TEST_F(DraftCaptureConfigTest, ConvertsBoundaryIndicesForSharedDraftCapture) {
  write_config(R"({"aux_hidden_state_layer_ids":[2,4]})");
  EXPECT_EQ(read_capture_layer_ids(model_path_), (std::vector<int32_t>{1, 3}));
}

TEST_F(DraftCaptureConfigTest, PrefersLegacyPostLayerIndices) {
  write_config(R"({"dspark_target_layer_ids":[1,3],
                    "aux_hidden_state_layer_ids":[3,5]})");
  EXPECT_EQ(read_capture_layer_ids(model_path_), (std::vector<int32_t>{1, 3}));
}

TEST_F(DraftCaptureConfigTest, AllowsEagleCaptureDefaultsForMissingConfig) {
  EXPECT_TRUE(read_capture_layer_ids(model_path_, /*required=*/false).empty());
  write_config("{}");
  EXPECT_TRUE(read_capture_layer_ids(model_path_, /*required=*/false).empty());
}

TEST_F(DraftCaptureConfigTest, RejectsMissingRequiredCaptureConfig) {
  EXPECT_DEATH(read_capture_layer_ids(model_path_), "Failed to parse");
  write_config("{}");
  EXPECT_DEATH(read_capture_layer_ids(model_path_),
               "requires dspark_target_layer_ids");
}

TEST_F(DraftCaptureConfigTest, RejectsEmbeddingCaptureBoundary) {
  write_config(R"({"aux_hidden_state_layer_ids":[0,4]})");
  EXPECT_DEATH(read_capture_layer_ids(model_path_), "Embedding capture");
}

TEST_F(DraftCaptureConfigTest, RejectsInvalidTargetCaptureLayers) {
  for (const std::string& json : {R"({"target_layer_ids":[-1,3]})",
                                  R"({"target_layer_ids":[1,8]})",
                                  R"({"target_layer_ids":[1,1]})"}) {
    write_config(json);
    ModelArgs args;
    args.n_layers(8);
    runtime::Options options;
    options.draft_model_path(model_path_);
    EXPECT_DEATH(configure_block_diffusion_target(args, options),
                 "Invalid target capture layer|out of range|Duplicate target "
                 "capture layer");
  }
}

TEST(DraftModelConfigDeathTest, RequiresDraftPathForTargetCapture) {
  ModelArgs args;
  runtime::Options options;
  EXPECT_DEATH(configure_block_diffusion_target(args, options),
               "requires --draft_model");
}

TEST_F(DraftCaptureConfigTest, MaskRootAliasWinsOverTheTextConfigValue) {
  write_config(R"({"dspark_noise_token_id": 128799,
                    "text_config": {"dspark_noise_token_id": 1}})");
  JsonReader reader;
  ASSERT_TRUE(reader.parse(model_path_ + "/config.json"));
  EXPECT_EQ(read_mask_token_id(reader), 128799);
}

TEST_F(DraftCaptureConfigTest, MaskTextConfigFallbackResolves) {
  write_config(R"({"text_config": {"mask_token_id": 128799}})");
  JsonReader reader;
  ASSERT_TRUE(reader.parse(model_path_ + "/config.json"));
  EXPECT_EQ(read_mask_token_id(reader), 128799);
}

TEST_F(DraftCaptureConfigTest, MaskMissingEverywhereReturnsInvalid) {
  write_config(R"({"text_config": {"num_hidden_layers": 4}})");
  JsonReader reader;
  ASSERT_TRUE(reader.parse(model_path_ + "/config.json"));
  EXPECT_EQ(read_mask_token_id(reader), -1);
}

TEST_F(DraftCaptureConfigTest, MaskRootAliasesKeepTheirPrecedence) {
  write_config(R"({"dflash_config": {"mask_token_id": 1},
                    "mask_token_id": 2,
                    "dspark_noise_token_id": 3})");
  JsonReader reader;
  ASSERT_TRUE(reader.parse(model_path_ + "/config.json"));
  EXPECT_EQ(read_mask_token_id(reader), 1);
  write_config(R"({"mask_token_id": 2, "dspark_noise_token_id": 3})");
  ASSERT_TRUE(reader.parse(model_path_ + "/config.json"));
  EXPECT_EQ(read_mask_token_id(reader), 2);
}

#if defined(USE_NPU)
TEST(DraftModelConfigTest,
     AllowsPythonPipelineGraphsWithoutChangingOtherDrafts) {
  auto& model_config = ModelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  for (const std::string& model_impl : {"python", "native"}) {
    model_config.model_impl(model_impl);
    for (const std::string& algorithm : {"DSpark", "DFlash", "DFlash2"}) {
      SCOPED_TRACE(model_impl);
      SCOPED_TRACE(algorithm);
      ModelArgs args;
      args.model_type(/*value=*/"qwen3").num_speculative_tokens(/*value=*/7);
      if (algorithm == "DFlash2") {
        args.dflash2_conv_kernel_size(/*value=*/4);
      }
      runtime::Options options;
      options.speculative_algorithm(algorithm).num_speculative_tokens(
          /*num_speculative_tokens=*/algorithm == "DFlash2" ? 7 : 6);

      configure_block_diffusion_draft(args, options);

      EXPECT_EQ(args.requires_eager_execution(), model_impl != "python");
      EXPECT_FALSE(args.draft_query_uses_decode());
      if (algorithm == "DSpark") {
        EXPECT_EQ(args.dummy_token_count(), model_impl == "python" ? 6 : 1);
      } else if (algorithm == "DFlash") {
        EXPECT_EQ(args.dummy_token_count(), model_impl == "python" ? 7 : 1);
      } else {
        EXPECT_EQ(args.dummy_token_count(), 8);
      }
    }
  }
  model_config.model_impl(original_model_impl);
}

TEST(DraftModelConfigTest, DFlashDummyGeometryUsesSpeculativeTokenCount) {
  auto& model_config = ModelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  model_config.model_impl("python");
  ModelArgs args;
  args.model_type("qwen3");
  runtime::Options options;
  options.speculative_algorithm(/*value=*/"DFlash")
      .num_speculative_tokens(
          /*num_speculative_tokens=*/7);
  configure_block_diffusion_draft(args, options);
  EXPECT_EQ(args.dummy_token_count(), 8);
  EXPECT_FALSE(args.requires_eager_execution());
  model_config.model_impl(original_model_impl);
}

TEST(DraftModelConfigTest, PythonV41UsesBlockRowsAndCompleteDummyQueries) {
  auto& model_config = ModelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  model_config.model_impl("python");
  ModelArgs args;
  args.model_type("deepseek_v41").n_layers(4).dspark_num_layers(2);
  runtime::Options options;
  options.speculative_algorithm(/*value=*/"DSpark")
      .num_speculative_tokens(
          /*num_speculative_tokens=*/3);
  configure_block_diffusion_draft(args, options);
  EXPECT_EQ(args.model_type(), "deepseek_v41_dspark");
  EXPECT_EQ(args.dummy_token_count(), 3);
  EXPECT_EQ(args.n_layers(), 2);
  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{0, 0}));
  EXPECT_FALSE(args.requires_eager_execution());
  EXPECT_TRUE(args.draft_query_uses_decode());
  model_config.model_impl(original_model_impl);
}

class DraftQueryInputConfigTest
    : public DraftCaptureConfigTest,
      public ::testing::WithParamInterface<
          std::tuple<std::string, std::string, bool>> {};

TEST_P(DraftQueryInputConfigTest, PreservesInputContractAcrossGraphModes) {
  const auto& [model_type, model_impl, enable_graph] = GetParam();
  auto& model_config = ModelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  model_config.model_impl(model_impl);
  for (bool graph_mode : {enable_graph, !enable_graph}) {
    ModelArgs args;
    args.model_type(model_type).n_layers(4).dspark_num_layers(2);
    runtime::Options options;
    options.speculative_algorithm(/*value=*/"DSpark")
        .num_speculative_tokens(3)
        .enable_graph(graph_mode);

    configure_block_diffusion_draft(args, options);

    EXPECT_TRUE(args.draft_query_uses_decode());
  }
  model_config.model_impl(original_model_impl);
}

INSTANTIATE_TEST_SUITE_P(
    DraftImplementations,
    DraftQueryInputConfigTest,
    ::testing::Values(std::make_tuple(std::string("deepseek_v4"),
                                      std::string("native"),
                                      false),
                      std::make_tuple(std::string("deepseek_v4"),
                                      std::string("python"),
                                      true),
                      std::make_tuple(std::string("deepseek_v41"),
                                      std::string("native"),
                                      false),
                      std::make_tuple(std::string("deepseek_v41"),
                                      std::string("python"),
                                      true)));

TEST(DraftModelConfigTest, PreservesNpuCheckpointBlockGeometry) {
  ModelArgs args;
  args.model_type("qwen3")
      .num_speculative_tokens(/*value=*/7)
      .dflash2_conv_kernel_size(/*value=*/4);
  runtime::Options options;
  options.speculative_algorithm(/*value=*/"DFlash2")
      .num_speculative_tokens(
          /*num_speculative_tokens=*/7);
  configure_block_diffusion_draft(args, options);
  EXPECT_EQ(args.num_speculative_tokens(), 7);
  EXPECT_EQ(args.dummy_token_count(), 8);
  EXPECT_EQ(args.model_type(), "DFlash2DraftModel");
}

namespace {

// Restore the public model implementation setting even after a failed test.
class ModelImplGuard final {
 public:
  explicit ModelImplGuard(const std::string& model_impl) {
    old_config_impl_ = ModelConfig::get_instance().model_impl();
    ModelConfig::get_instance().model_impl(model_impl);
  }

  ModelImplGuard(const ModelImplGuard&) = delete;
  ModelImplGuard& operator=(const ModelImplGuard&) = delete;

  ~ModelImplGuard() {
    ModelConfig::get_instance().model_impl(old_config_impl_);
  }

 private:
  std::string old_config_impl_;
};

}  // namespace

// V4 kernels use ratio 1 for the checkpoint's zero-encoded SWA tail.
TEST_F(DraftCaptureConfigTest, NormalizesV4DraftTailZeroCompressRatios) {
  ModelArgs args;
  args.model_type("deepseek_v4")
      .n_layers(4)
      .dspark_num_layers(3)
      .compress_ratios({1, 4, 128, 4, 0, 0, 0});
  runtime::Options options;
  options.speculative_algorithm(/*value=*/"DSpark")
      .num_speculative_tokens(/*num_speculative_tokens=*/5);

  configure_block_diffusion_draft(args, options);

  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{1, 1, 1}))
      << "V4.1-style 0 draft tail must normalize to pure-SWA ratio 1 for the "
         "C++ V4 draft";
  EXPECT_EQ(args.model_type(), "deepseek_v4_dspark");
}

// Python V4.1 preserves the distinct zero-encoded SWA group.
TEST_F(DraftCaptureConfigTest, PreservesV41DraftTailZeroCompressRatios) {
  const ModelImplGuard model_impl_guard("python");

  ModelArgs args;
  args.model_type("deepseek_v41")
      .n_layers(4)
      .dspark_num_layers(3)
      .compress_ratios({1, 4, 128, 4, 0, 0, 0});
  runtime::Options options;
  options.speculative_algorithm(/*value=*/"DSpark")
      .num_speculative_tokens(/*num_speculative_tokens=*/5);

  configure_block_diffusion_draft(args, options);

  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{0, 0, 0}))
      << "V4.1 draft keeps the checkpoint's 0 pure-SWA encoding verbatim";
  EXPECT_EQ(args.model_type(), "deepseek_v41_dspark");
}

// A native V4.1 checkpoint selects the V4 draft and its ratio-1 encoding.
TEST_F(DraftCaptureConfigTest, NormalizesV41DraftTailForNativeImpl) {
  const ModelImplGuard model_impl_guard("native");

  ModelArgs args;
  args.model_type("deepseek_v41")
      .n_layers(4)
      .dspark_num_layers(3)
      .compress_ratios({1, 4, 128, 4, 0, 0, 0});
  runtime::Options options;
  options.speculative_algorithm(/*value=*/"DSpark")
      .num_speculative_tokens(/*num_speculative_tokens=*/5);

  configure_block_diffusion_draft(args, options);

  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{1, 1, 1}))
      << "A V4.1 checkpoint on the native impl takes the C++ V4 draft, whose "
         "0-encoded tail must normalize to the pure-SWA ratio 1";
  EXPECT_EQ(args.model_type(), "deepseek_v4_dspark");
}
#endif

}  // namespace
}  // namespace xllm
