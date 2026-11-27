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

#pragma once

#include <optional>
#include <vector>

#include "core/runtime/dflash_worker_impl.h"

namespace xllm {

class ProcessGroup;

// DSpark = DFlash block-diffusion draft + a low-rank Markov head that adds a
// prefix-dependent bias to each block position's draft logits. It reuses the
// entire DFlash pipeline (prefill, context-K/V injection, validate, cache
// writes).
class DSparkWorkerImpl final : public DFlashWorkerImpl {
 public:
  DSparkWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options,
                   WorkerType worker_type);

  ~DSparkWorkerImpl() override;

 protected:
  void on_draft_model_loaded(const std::string& model_weights_path) override;
  // N-wide query block: anchor (slot 0) carries the last real token and
  // predicts the first draft token; every position is a prediction. DFlash's
  // prepare_query_inputs reads this hook to pick the block layout.
  bool sample_from_anchor() const override { return true; }

  // Build the DSpark query and produce the complete proposal block.
  ForwardOutput run_decode_draft(const ForwardInput& input,
                                 ForwardInput& validate_input) override;

  std::optional<ForwardOutput> step_decode(const ForwardInput& input) override;
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;
  std::optional<ForwardOutput> step_empty(const ForwardInput& input) override;
  bool can_prepare_without_compute_stream_wait(
      const ForwardInput& input) const override;

 private:
  bool can_prelaunch_draft_for_init(bool supports_device_inputs) const;
  ForwardOutput run_decode_draft_from_pending(
      const ForwardInput& input,
      const std::vector<int64_t>& rows) override;
  ForwardOutput execute_draft_query(const ForwardInput& input,
                                    ForwardInput query,
                                    bool prepared,
                                    ForwardInput* validate_input = nullptr);

  std::pair<torch::Tensor, torch::Tensor> sample_block(
      const torch::Tensor& base_logits,
      const torch::Tensor& anchor_token_ids,
      const SamplingParameters& sampling_params) const;

  void synchronize_sampled_token_ids(
      torch::Tensor& sampled_token_ids,
      const SamplingParameters& sampling_params) const;

  ProcessGroup* sampling_process_group_ = nullptr;
};

}  // namespace xllm
