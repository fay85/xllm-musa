/* Copyright 2026 The xLLM Authors.
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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/framework/model/aux_hidden_capture.h"
#include "models/llm/dspark_weight_source.h"

namespace xllm {
namespace {

TEST(DSparkMetadataContract, CapturesRequestedLayerOutputsInCheckpointOrder) {
  ModelArgs args;
  args.hidden_size(2).layers_to_capture({41, 42, 43});
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  AuxHiddenCapture capture(args, options, /*max_tokens_per_batch=*/8);
  EXPECT_FALSE(capture.should_capture(40));
  EXPECT_TRUE(capture.should_capture(43));
  for (int32_t layer = 41; layer <= 43; ++layer) {
    torch::Tensor hidden = torch::full({3, 4, 2}, layer, options);
    capture.capture_layer(layer, hidden.mean(/*dim=*/1), std::nullopt);
    hidden.zero_();
  }
  const auto output = capture.finalize(torch::zeros({3, 2}, options));
  EXPECT_EQ(output.aux_hidden_states.size(0), 3);
  EXPECT_EQ(output.aux_hidden_states.size(1), 6);
  for (int64_t layer = 0; layer < 3; ++layer) {
    EXPECT_TRUE(torch::equal(
        output.aux_hidden_states.narrow(/*dim=*/1, layer * 2, /*length=*/2),
        torch::full({3, 2}, 41 + layer, options)));
  }
}

TEST(DSparkMetadataContract, DedicatedVocabularyWinsInEitherShardOrder) {
  for (bool dedicated_first : {false, true}) {
    dspark_detail::VocabularyWeightSelector selector;
    EXPECT_TRUE(selector.should_load(dedicated_first));
    selector.mark_loaded(dedicated_first);
    if (!dedicated_first) {
      EXPECT_TRUE(selector.should_load(/*dedicated=*/true));
      selector.mark_loaded(/*dedicated=*/true);
    }
    EXPECT_TRUE(selector.loaded());
    EXPECT_FALSE(selector.should_load(/*dedicated=*/false));
    EXPECT_FALSE(selector.should_load(/*dedicated=*/true));
  }
}

TEST(DSparkMetadataContract, DisabledCaptureDoesNotPublishAuxiliaryState) {
  ModelArgs args;
  args.hidden_size(2);
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  AuxHiddenCapture capture(args, options, /*max_tokens_per_batch=*/8);
  EXPECT_FALSE(capture.enabled());
  const auto output = capture.finalize(torch::zeros({3, 2}, options));
  EXPECT_FALSE(output.aux_hidden_states.defined());
}

}  // namespace
}  // namespace xllm
