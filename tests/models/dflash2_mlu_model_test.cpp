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
#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <utility>

#include "core/framework/parallel_state/process_group.h"
#include "core/layers/mlu/dflash2_decoder_layer.h"
#include "core/platform/platform.h"
#include "core/runtime/mlu_graph_executor_impl.h"
#include "models/llm/mlu/qwen3_dflash2.h"

namespace xllm {
namespace {

using MluDraftModel = mlu::model::DFlash2Qwen3ModelImpl;
using DraftCausalLM = CausalLMImpl<mlu::model::DFlash2Qwen3ForCausalLM>;

class CountingDraftCausalLM final : public DraftCausalLM {
 public:
  using DraftCausalLM::DraftCausalLM;

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& caches,
                      const ModelInputParams& params) override {
    ++calls_;
    return DraftCausalLM::forward(tokens, positions, caches, params);
  }

  int32_t calls() const { return calls_; }

 private:
  int32_t calls_ = 0;
};
static_assert(
    std::is_base_of_v<QWen3ModelImplBase<layer::mlu::DFlash2DecoderLayer>,
                      MluDraftModel>);

ModelArgs make_draft_args() {
  ModelArgs args;
  args.model_type("DFlash2DraftModel")
      .n_layers(1)
      .hidden_size(128)
      .intermediate_size(256)
      .hidden_act("silu")
      .vocab_size(16)
      .head_dim(128)
      .n_heads(1)
      .n_kv_heads(1)
      .attention_bias(false)
      .max_position_embeddings(32)
      .rope_theta(10000)
      .rms_norm_eps(1e-5)
      .sliding_window(16)
      .num_speculative_tokens(/*value=*/3)
      .dflash2_conv_group_size(16)
      .dflash2_conv_kernel_size(2)
      .dflash2_selector_rank(8)
      .dflash2_selector_top_k(4);
  // One target hidden state matches the checkpoint's 128-by-128 fc weight.
  args.dflash2_target_layer_ids() = {0};
  return args;
}

std::unordered_map<std::string, torch::Tensor> make_decoder_weights(
    const torch::TensorOptions& options) {
  std::unordered_map<std::string, torch::Tensor> weights;
  for (const std::string& projection :
       {"q_proj", "k_proj", "v_proj", "o_proj"}) {
    weights.emplace("self_attn." + projection + ".weight",
                    torch::full({128, 128}, 0.125, options));
  }
  for (const std::string& norm : {"self_attn.q_norm",
                                  "self_attn.k_norm",
                                  "input_layernorm",
                                  "post_attention_layernorm"}) {
    weights.emplace(norm + ".weight", torch::ones({128}, options));
  }
  weights.emplace("mlp.gate_proj.weight",
                  torch::full({256, 128}, 0.25, options));
  weights.emplace("mlp.up_proj.weight", torch::full({256, 128}, 0.5, options));
  weights.emplace("mlp.down_proj.weight",
                  torch::full({128, 256}, 0.125, options));
  for (const std::string& conv : {"attention_conv", "mlp_conv"}) {
    weights.emplace(conv + ".base_kernel",
                    torch::full({2, 2, 128}, 0.75, options));
    weights.emplace(conv + ".kernel_projection.weight",
                    torch::full({32, 128}, 0.25, options));
  }
  return weights;
}

TEST(DFlash2MluModelTest, ActualDraftModelLoadsUnchangedLayerPrefixes) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  ProcessGroup group(0, 1, device);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  ModelContext context(parallel, make_draft_args(), QuantArgs(), options);
  MluDraftModel model(context);
  std::unordered_map<std::string, torch::Tensor> weights;
  for (const auto& [name, value] : make_decoder_weights(options)) {
    weights.emplace("layers.0." + name, value);
  }
  weights.emplace("fc.weight", torch::ones({128, 128}, options));
  weights.emplace("hidden_norm.weight", torch::ones({128}, options));
  weights.emplace("norm.weight", torch::ones({128}, options));
  weights.emplace("candidate_selector.hidden_projection.weight",
                  torch::ones({8, 128}, options));
  weights.emplace("candidate_selector.predecessor_codebook",
                  torch::ones({16, 8}, options));
  weights.emplace("candidate_selector.successor_codebook",
                  torch::ones({16, 8}, options));
  model.load_state_dict(StateDict(weights));
  model.verify_loaded_weights();
}

TEST(DFlash2MluModelTest, DraftGraphReplaysFreshBlocksForBfloat16AndInt8) {
  torch::manual_seed(42);
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto ints = options.dtype(torch::kInt32);
  ProcessGroup group(0, 1, device);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  for (const bool quantized : {false, true}) {
    SCOPED_TRACE(quantized ? "INT8" : "BF16");
    ModelArgs args = make_draft_args();
    args.max_position_embeddings(128);
    QuantArgs quant_args;
    if (quantized) {
      quant_args.quant_method("compressed-tensors")
          .is_compressed_tensors_w8a8_dynamic(true);
      CompressedQuantGroup scheme;
      scheme.targets = {
          R"(re:layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$)",
          R"(re:layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$)"};
      quant_args.compressed_groups() = {scheme};
    }
    ModelContext context(parallel, args, quant_args, options);
    mlu::model::DFlash2Qwen3ForCausalLM model(context);
    const auto draft = model->named_modules()["model"]->as<MluDraftModel>();
    ASSERT_NE(draft, nullptr);
    std::unordered_map<std::string, torch::Tensor> weights;
    for (const auto& [name, tensor] : make_decoder_weights(options)) {
      const std::string key = "layers.0." + name;
      auto weight =
          tensor.dim() == 2 ? torch::randn_like(tensor) * 0.02 : tensor;
      if (quantized &&
          quant_args.module_quant_method(key.substr(0, key.size() - 7))
              .has_value()) {
        weights.emplace(key + "_scale",
                        torch::full({weight.size(0), 1}, 1.0 / 1024, options));
        weight = (weight.to(torch::kFloat32) * 1024)
                     .round()
                     .clamp(-127, 127)
                     .to(torch::kInt8);
      }
      weights.emplace(key, std::move(weight));
    }
    weights.emplace("fc.weight", torch::ones({128, 128}, options));
    weights.emplace("hidden_norm.weight", torch::ones({128}, options));
    weights.emplace("norm.weight", torch::ones({128}, options));
    weights.emplace("candidate_selector.hidden_projection.weight",
                    torch::ones({8, 128}, options));
    weights.emplace("candidate_selector.predecessor_codebook",
                    torch::ones({16, 8}, options));
    weights.emplace("candidate_selector.successor_codebook",
                    torch::ones({16, 8}, options));
    draft->load_state_dict(StateDict(std::move(weights)));
    draft->verify_loaded_weights();
    draft->finalize_loaded_weights();
    CountingDraftCausalLM causal(std::move(model), options);
    runtime::Options runtime_options;
    runtime_options.is_draft_engine(true)
        .speculative_algorithm(/*value=*/"DFlash2")
        .num_speculative_tokens(/*num_speculative_tokens=*/3)
        .block_size(16)
        .max_seqs_per_batch(4);
    mlu::MluGraphExecutorImpl executor(&causal, args, device, runtime_options);
    const auto initial_k = torch::randn({16, 1, 16, 128}, options);
    const auto initial_v = torch::randn_like(initial_k);
    std::vector<KVCache> eager_caches;
    std::vector<KVCache> graph_caches;
    eager_caches.emplace_back(
        KVCacheTensors{initial_k.clone(), initial_v.clone()});
    graph_caches.emplace_back(
        KVCacheTensors{initial_k.clone(), initial_v.clone()});
    torch::Tensor previous;
    for (const int32_t history : {1, 7, 17, 3}) {
      const int32_t requests = history == 7 ? 4 : 3;
      ModelInputParams params;
      params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
      params.meta.num_sequences = requests;
      params.meta.q_max_seq_len = 4;
      params.meta.kv_max_seq_len = history + 4;
      params.parallel.dp_global_token_nums = {requests * 4};
      params.parallel.dp_is_decode = {0};
      params.attention.host.q_seq_lens = {0};
      params.attention.host.kv_seq_lens = {0};
      std::vector<int32_t> tables;
      std::vector<int32_t> slots;
      tables.reserve(requests * 4);
      slots.reserve(requests * 4);
      for (int32_t row = 0; row < requests; ++row) {
        params.attention.host.q_seq_lens.emplace_back((row + 1) * 4);
        params.attention.host.kv_seq_lens.emplace_back((row + 1) *
                                                       (history + 4));
        const int32_t base = ((row + history) % 4) * 4;
        for (int32_t col = 0; col < 4; ++col) {
          tables.emplace_back(base + col);
          slots.emplace_back(base * 16 + history + col);
        }
      }
      params.attention.device.q_seq_lens =
          torch::tensor(params.attention.host.q_seq_lens, ints);
      params.attention.device.kv_seq_lens =
          torch::tensor(params.attention.host.kv_seq_lens, ints);
      params.attention.device.new_cache_slots = torch::tensor(slots, ints);
      params.attention.device.block_tables =
          torch::tensor(tables, ints).view({requests, 4});
      params.embedding.input_embedding =
          torch::randn({requests * 4, 128}, options);
      const auto tokens = torch::zeros({requests * 4}, ints);
      const auto positions =
          torch::arange(history, history + 4, ints).repeat({requests});
      auto eager_params = params;
      eager_params.embedding.input_embedding =
          params.embedding.input_embedding.clone();
      const auto expected =
          causal.forward(tokens, positions, eager_caches, eager_params)
              .hidden_states;
      const int32_t before_graph = causal.calls();
      const auto actual = executor.run(tokens, positions, graph_caches, params)
                              .hidden_states.clone();
      EXPECT_EQ(causal.calls(), before_graph + (history == 1 ? 1 : 0));
      EXPECT_TRUE(torch::allclose(actual, expected, 0.002, 0.002))
          << "history=" << history
          << " max_error=" << (actual - expected).abs().max().item<float>();
      EXPECT_TRUE(torch::equal(graph_caches[0].get_k_cache(),
                               eager_caches[0].get_k_cache()));
      EXPECT_TRUE(torch::equal(graph_caches[0].get_v_cache(),
                               eager_caches[0].get_v_cache()));
      if (previous.defined() && previous.sizes() == actual.sizes()) {
        EXPECT_FALSE(torch::equal(previous, actual));
      }
      previous = actual;
    }
  }
}

}  // namespace
}  // namespace xllm
