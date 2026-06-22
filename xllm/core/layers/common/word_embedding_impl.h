/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>

#include "core/framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

// Embedding parallelized in the embedding dimension.
class WordEmbeddingImpl : public torch::nn::Module {
 public:
  WordEmbeddingImpl(const ModelContext& context);
  WordEmbeddingImpl(int64_t num_embeddings,
                    int64_t embedding_dim,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options);

  // The input to the module is a list of indices, and the output is the
  // corresponding word embeddings.
  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // rank of current process
  PROPERTY(int32_t, rank) = 0;

  // world size
  PROPERTY(int32_t, world_size) = 0;

  // parameter members, must be registered
  DEFINE_WEIGHT(weight);

  // parallel args
  ParallelArgs parallel_args_;

#if defined(USE_CUDA) || defined(USE_MUSA)
  // Persistent output buffer for graph-capture-safe token embedding lookup.
  // F::embedding internally invokes weight_.index_select(0, input), and on
  // torch_musa 2.7.1 that path always allocates a fresh output via
  // EmptyMUSA -- which the MUSA stream-capture engine rejects with
  //   "operation not permitted when stream is capturing"
  // even after the warmup pass primed the caching allocator (torch_musa's
  // allocator does not honor c10::cuda::MemPoolContext set by the graph
  // executor, unlike libtorch's CUDA caching allocator).
  //
  // Mirrors the same lazily-allocated, grow-only persistent buffer pattern
  // used by ColumnParallelLinearImpl::output_buf_, Qwen3NextRMSNormImpl::
  // norm_out_buf_, and AttentionImpl::output_buf_. The buffer is sized on
  // the first forward (which happens during the pre-capture eager warmup),
  // grown only if a later call sees a larger num_tokens, then sliced via
  // narrow() for every smaller-bucket replay. Captured graphs therefore see
  // a stable storage pointer that stays valid across replays.
  //
  // Eager (non-capture) calls also use this buffer; the only overhead is
  // the initial torch::empty on the first call, which is amortised over
  // the entire process lifetime.
  mutable torch::Tensor output_buf_;
#endif
};

}  // namespace layer
}  // namespace xllm
