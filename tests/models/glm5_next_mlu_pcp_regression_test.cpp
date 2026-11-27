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

// CP8 hybrid KDA/DSA with deterministic checkpoint-format weights.
// FFN, expert, vocabulary and layer counts are reduced.

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/config/eplb_config.h"
#include "framework/config/kv_cache_config.h"
#include "framework/config/scheduler_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model/model_output.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "models/llm/mlu/glm5_next.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"
#include "util/tensor_helper.h"

namespace xllm::mlu::model {
namespace {

constexpr int32_t kWorldSize = 8;
constexpr int32_t kNumLayers = 8;
constexpr int32_t kNumSpeculativeTokens = 3;
constexpr int64_t kBlockSize = 16;
constexpr int64_t kMaxTokensPerBatch = 16384;
constexpr uint64_t kWeightSeed = 0x474c4d3533464c41ULL;

constexpr int64_t kHiddenSize = 4096;
constexpr int64_t kHcMult = 4;
constexpr int64_t kLinearHeads = 64;
constexpr int64_t kLinearHeadDim = 128;
constexpr int64_t kConvKernel = 4;
constexpr int64_t kAttentionHeads = 64;
constexpr int64_t kQLoraRank = 1536;
constexpr int64_t kKvLoraRank = 512;
constexpr int64_t kQkNopeHeadDim = 256;
constexpr int64_t kVHeadDim = 256;
constexpr int64_t kIndexHeads = 32;
constexpr int64_t kIndexHeadDim = 128;
constexpr int64_t kIndexTopk = 2048;
constexpr int64_t kIndexKpool = 4;
constexpr int64_t kW4GroupSize = 128;

// Reduced dimensions; see the file comment.
constexpr int64_t kVocabSize = 512;
constexpr int64_t kIntermediateSize = 512;
constexpr int64_t kMoeIntermediateSize = 256;
constexpr int64_t kRoutedExperts = 16;

// Linear-state slot 0 and KV block 0 are reserved for padding.
constexpr int64_t kLinearStateBlocks = 3;
constexpr int64_t kKvBlocks = 192;
constexpr float kConvCheckpointSentinel = 0.375f;
constexpr float kSsmCheckpointSentinel = -1.25f;

constexpr double kDefaultSampleTolerance = 1e-3;

constexpr int32_t kExitSkip = 77;
constexpr int32_t kExitMismatch = 1;
constexpr uint32_t kChildTimeoutSeconds = 1500;

using Weights = std::unordered_map<std::string, torch::Tensor>;

uint64_t fnv1a64(const std::string& text) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (const char character : text) {
    hash ^= static_cast<uint8_t>(character);
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

uint64_t splitmix64(uint64_t& state) {
  state += 0x9e3779b97f4a7c15ULL;
  uint64_t value = state;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

// Every tensor is a pure function of its key, so each rank derives identical
// full weights on the CPU regardless of creation order or thread count.
torch::Tensor uniform_tensor(const std::string& key,
                             std::vector<int64_t> shape,
                             float low,
                             float high,
                             torch::ScalarType dtype) {
  torch::Tensor values = torch::empty(shape, torch::kFloat32);
  float* data = values.data_ptr<float>();
  const int64_t count = values.numel();
  uint64_t state = fnv1a64(key) ^ kWeightSeed;
  const float span = high - low;
  for (int64_t index = 0; index < count; ++index) {
    const float unit =
        static_cast<float>(splitmix64(state) >> 40) * (1.0f / 16777216.0f);
    data[index] = low + span * unit;
  }
  return values.to(dtype);
}

// symmetric=true yields INT8 weights in [-127, 127]; false yields raw bytes
// for the packed INT4 expert weights.
torch::Tensor int8_tensor(const std::string& key,
                          std::vector<int64_t> shape,
                          bool symmetric) {
  torch::Tensor values = torch::empty(shape, torch::kInt8);
  int8_t* data = values.data_ptr<int8_t>();
  const int64_t count = values.numel();
  uint64_t state = fnv1a64(key) ^ kWeightSeed;
  for (int64_t index = 0; index < count; ++index) {
    const uint64_t random = splitmix64(state) >> 32;
    data[index] =
        symmetric
            ? static_cast<int8_t>(static_cast<int32_t>(random % 255) - 127)
            : static_cast<int8_t>(random & 0xff);
  }
  return values;
}

class WeightBuilder final {
 public:
  explicit WeightBuilder(const torch::Device& device) : device_(device) {}

  void add_uniform(const std::string& key,
                   std::vector<int64_t> shape,
                   float low,
                   float high,
                   torch::ScalarType dtype = torch::kBFloat16) {
    add(key, uniform_tensor(key, std::move(shape), low, high, dtype));
  }

  // Unquantized BF16 projection with weights of magnitude ~0.01.
  void add_float_linear(const std::string& prefix,
                        int64_t out_features,
                        int64_t in_features) {
    add_uniform(prefix + ".weight",
                {out_features, in_features},
                /*low=*/-0.02f,
                /*high=*/0.02f);
  }

  // compressed-tensors int-quantized W8A8: INT8 weight, per-channel BF16
  // scale and an input smooth vector.
  void add_w8a8_linear(const std::string& prefix,
                       int64_t out_features,
                       int64_t in_features) {
    add_w8a8_linear(prefix, out_features, in_features, prefix + ".smooth");
  }

  // Projections that read the same input share one smooth vector, selected
  // by smooth_key.
  void add_w8a8_linear(const std::string& prefix,
                       int64_t out_features,
                       int64_t in_features,
                       const std::string& smooth_key) {
    add(prefix + ".weight",
        int8_tensor(prefix + ".weight",
                    {out_features, in_features},
                    /*symmetric=*/true));
    add_uniform(prefix + ".weight_scale",
                {out_features, 1},
                /*low=*/1.0e-4f,
                /*high=*/2.0e-4f);
    add(prefix + ".smooth",
        uniform_tensor(smooth_key,
                       {in_features},
                       /*low=*/0.5f,
                       /*high=*/1.5f,
                       torch::kBFloat16));
  }

  // SwiGLU MLP in W8A8: gate_proj and up_proj share their input smooth.
  void add_w8a8_swiglu(const std::string& prefix, int64_t intermediate_size) {
    const std::string input_smooth = prefix + "input.smooth";
    add_w8a8_linear(
        prefix + "gate_proj", intermediate_size, kHiddenSize, input_smooth);
    add_w8a8_linear(
        prefix + "up_proj", intermediate_size, kHiddenSize, input_smooth);
    add_w8a8_linear(prefix + "down_proj", kHiddenSize, intermediate_size);
  }

  // compressed-tensors int4-le-pack-quantized W4A8: two INT4 values per
  // byte, one BF16 scale per group of 128 input channels.
  void add_w4a8_linear(const std::string& prefix,
                       int64_t out_features,
                       int64_t in_features,
                       const torch::Tensor& smooth) {
    add(prefix + ".weight",
        int8_tensor(prefix + ".weight",
                    {out_features, in_features / 2},
                    /*symmetric=*/false));
    add_uniform(prefix + ".weight_scale",
                {out_features, in_features / kW4GroupSize},
                /*low=*/1.5e-3f,
                /*high=*/3.0e-3f);
    add(prefix + ".smooth", smooth);
  }

  void add(const std::string& key, const torch::Tensor& cpu_tensor) {
    const bool inserted = weights_.emplace(key, cpu_tensor.to(device_)).second;
    CHECK(inserted) << "Duplicate weight key: " << key;
  }

  Weights release() { return std::move(weights_); }

 private:
  torch::Device device_;
  Weights weights_;
};

bool is_kda_layer(int32_t layer_id) { return layer_id % 4 != 3; }

bool is_dense_layer(int32_t layer_id) { return layer_id < 3; }

void add_kda_weights(WeightBuilder& builder, const std::string& prefix) {
  const int64_t projection = kLinearHeads * kLinearHeadDim;
  for (const std::string name : {"q_proj", "k_proj", "v_proj"}) {
    builder.add_float_linear(prefix + name, projection, kHiddenSize);
  }
  builder.add_float_linear(prefix + "b_proj", kLinearHeads, kHiddenSize);
  for (const std::string name : {"f_a_proj", "g_a_proj"}) {
    builder.add_float_linear(prefix + name, kLinearHeadDim, kHiddenSize);
  }
  for (const std::string name : {"f_b_proj", "g_b_proj"}) {
    builder.add_float_linear(prefix + name, projection, kLinearHeadDim);
  }
  for (const std::string name : {"q_conv1d", "k_conv1d", "v_conv1d"}) {
    builder.add_uniform(prefix + name + ".weight",
                        {projection, 1, kConvKernel},
                        /*low=*/-0.2f,
                        /*high=*/0.2f,
                        torch::kFloat32);
  }
  builder.add_uniform(prefix + "A_log",
                      {kLinearHeads},
                      /*low=*/-0.2f,
                      /*high=*/0.2f,
                      torch::kFloat32);
  builder.add_uniform(prefix + "dt_bias",
                      {projection},
                      /*low=*/-0.2f,
                      /*high=*/0.2f,
                      torch::kFloat32);
  builder.add_uniform(
      prefix + "o_norm.weight", {kLinearHeadDim}, /*low=*/0.9f, /*high=*/1.1f);
  builder.add_float_linear(prefix + "o_proj", kHiddenSize, projection);
}

void add_dsa_weights(WeightBuilder& builder, const std::string& prefix) {
  builder.add_w8a8_linear(prefix + "q_a_proj", kQLoraRank, kHiddenSize);
  builder.add_uniform(prefix + "q_a_layernorm.weight",
                      {kQLoraRank},
                      /*low=*/0.9f,
                      /*high=*/1.1f);
  builder.add_w8a8_linear(
      prefix + "q_b_proj", kAttentionHeads * kQkNopeHeadDim, kQLoraRank);
  builder.add_w8a8_linear(
      prefix + "kv_a_proj_with_mqa", kKvLoraRank, kHiddenSize);
  builder.add_uniform(prefix + "kv_a_layernorm.weight",
                      {kKvLoraRank},
                      /*low=*/0.9f,
                      /*high=*/1.1f);
  builder.add_float_linear(prefix + "kv_b_proj",
                           kAttentionHeads * (kQkNopeHeadDim + kVHeadDim),
                           kKvLoraRank);
  builder.add_w8a8_linear(
      prefix + "o_proj", kHiddenSize, kAttentionHeads * kVHeadDim);

  const std::string indexer = prefix + "indexer.";
  builder.add_float_linear(
      indexer + "wq_b", kIndexHeads * kIndexHeadDim, kQLoraRank);
  builder.add_float_linear(indexer + "wk", kIndexHeadDim, kHiddenSize);
  builder.add_float_linear(indexer + "weights_proj", kIndexHeads, kHiddenSize);
  builder.add_uniform(indexer + "k_norm.weight",
                      {kIndexHeadDim},
                      /*low=*/0.9f,
                      /*high=*/1.1f);
  builder.add_uniform(indexer + "k_norm.bias",
                      {kIndexHeadDim},
                      /*low=*/-0.05f,
                      /*high=*/0.05f);
  builder.add_uniform(indexer + "index_kpool_compress_gate",
                      {kIndexHeadDim, kHiddenSize},
                      /*low=*/-0.02f,
                      /*high=*/0.02f);
  builder.add_uniform(indexer + "index_kpool_compress_ape",
                      {kIndexKpool, kIndexHeadDim},
                      /*low=*/-0.05f,
                      /*high=*/0.05f);
}

void add_sparse_moe_weights(WeightBuilder& builder, const std::string& prefix) {
  builder.add_float_linear(prefix + "gate", kRoutedExperts, kHiddenSize);
  builder.add_uniform(prefix + "gate.e_score_correction_bias",
                      {kRoutedExperts},
                      /*low=*/-0.1f,
                      /*high=*/0.1f,
                      torch::kFloat32);
  for (int64_t expert = 0; expert < kRoutedExperts; ++expert) {
    const std::string expert_prefix =
        prefix + "experts." + std::to_string(expert) + ".";
    // gate_proj and up_proj share one input smooth vector per expert.
    const torch::Tensor input_smooth = uniform_tensor(expert_prefix + "smooth",
                                                      {kHiddenSize},
                                                      /*low=*/0.5f,
                                                      /*high=*/1.5f,
                                                      torch::kBFloat16);
    builder.add_w4a8_linear(expert_prefix + "gate_proj",
                            kMoeIntermediateSize,
                            kHiddenSize,
                            input_smooth);
    builder.add_w4a8_linear(expert_prefix + "up_proj",
                            kMoeIntermediateSize,
                            kHiddenSize,
                            input_smooth);
    builder.add_w4a8_linear(expert_prefix + "down_proj",
                            kHiddenSize,
                            kMoeIntermediateSize,
                            uniform_tensor(expert_prefix + "down_proj.smooth",
                                           {kMoeIntermediateSize},
                                           /*low=*/0.5f,
                                           /*high=*/1.5f,
                                           torch::kBFloat16));
  }
  builder.add_w8a8_swiglu(prefix + "shared_experts.", kMoeIntermediateSize);
}

// Full checkpoint-format weights. Each rank builds all of them; the
// production loader keeps the EP shard [rank * E / 8, (rank + 1) * E / 8) of
// the routed experts and replicates everything else.
Weights make_weights(const torch::Device& device) {
  WeightBuilder builder(device);
  builder.add_uniform("embed_tokens.weight",
                      {kVocabSize, kHiddenSize},
                      /*low=*/-1.5f,
                      /*high=*/1.5f);
  builder.add_uniform(
      "norm.weight", {kHiddenSize}, /*low=*/0.9f, /*high=*/1.1f);
  const int64_t hc_rows = (2 + kHcMult) * kHcMult;
  for (int32_t layer_id = 0; layer_id < kNumLayers; ++layer_id) {
    const std::string layer = "layers." + std::to_string(layer_id) + ".";
    for (const std::string name :
         {"input_layernorm", "post_attention_layernorm"}) {
      builder.add_uniform(layer + name + ".weight",
                          {kHiddenSize},
                          /*low=*/0.9f,
                          /*high=*/1.1f);
    }
    for (const std::string name : {"hc_attn_", "hc_ffn_"}) {
      builder.add_uniform(layer + name + "fn",
                          {hc_rows, kHcMult * kHiddenSize},
                          /*low=*/-0.02f,
                          /*high=*/0.02f);
      builder.add_uniform(
          layer + name + "base", {hc_rows}, /*low=*/-0.1f, /*high=*/0.1f);
      builder.add_uniform(
          layer + name + "scale", {3}, /*low=*/0.05f, /*high=*/0.15f);
    }
    if (is_kda_layer(layer_id)) {
      add_kda_weights(builder, layer + "self_attn.");
    } else {
      add_dsa_weights(builder, layer + "self_attn.");
    }
    if (is_dense_layer(layer_id)) {
      builder.add_w8a8_swiglu(layer + "mlp.", kIntermediateSize);
    } else {
      add_sparse_moe_weights(builder, layer + "mlp.");
    }
  }
  return builder.release();
}

ModelArgs make_model_args(const std::vector<int32_t>& layers_to_capture) {
  ModelArgs args;
  args.model_type() = "glm5_next";
  args.dtype() = "bfloat16";
  args.vocab_size() = kVocabSize;
  args.hidden_size() = kHiddenSize;
  args.n_layers() = kNumLayers;
  args.n_heads() = kAttentionHeads;
  args.n_kv_heads() = kAttentionHeads;
  args.intermediate_size() = kIntermediateSize;
  args.max_position_embeddings() = 1048576;
  args.rms_norm_eps() = 1e-5f;
  args.tie_word_embeddings() = false;
  args.first_k_dense_replace() = 3;
  args.hidden_act() = "silu";
  args.n_routed_experts() = kRoutedExperts;
  args.n_shared_experts() = 1;
  args.num_experts_per_tok() = 8;
  args.moe_intermediate_size() = kMoeIntermediateSize;
  args.routed_scaling_factor() = 2.5f;
  args.norm_topk_prob() = true;
  args.n_group() = 1;
  args.topk_group() = 1;
  args.scoring_func() = "sigmoid";
  args.topk_method() = "noaux_tc";
  args.swiglu_limit() = 10.0f;
  args.hc_mult() = kHcMult;
  args.hc_sinkhorn_iters() = 20;
  args.hc_eps() = 1e-6f;
  args.qk_nope_head_dim() = kQkNopeHeadDim;
  args.qk_rope_head_dim() = 0;
  args.v_head_dim() = kVHeadDim;
  args.head_dim() = kQkNopeHeadDim;
  args.q_lora_rank() = kQLoraRank;
  args.kv_lora_rank() = kKvLoraRank;
  args.enable_mla() = true;
  args.rope_scaling_rope_type() = "default";
  args.index_head_dim() = kIndexHeadDim;
  args.index_n_heads() = kIndexHeads;
  args.index_topk() = kIndexTopk;
  args.index_kpool() = kIndexKpool;
  args.index_kpool_compress() = true;
  args.index_kpool_always_select_tail() = true;
  args.index_share_for_mtp_iteration() = true;
  args.linear_num_key_heads() = kLinearHeads;
  args.linear_num_value_heads() = kLinearHeads;
  args.linear_key_head_dim() = kLinearHeadDim;
  args.linear_value_head_dim() = kLinearHeadDim;
  args.linear_conv_kernel_dim() = kConvKernel;
  args.linear_lower_bound() = -5.0f;
  args.mamba_ssm_dtype() = "float32";
  std::vector<std::string> layer_types;
  std::vector<std::string> mlp_layer_types;
  layer_types.reserve(kNumLayers);
  mlp_layer_types.reserve(kNumLayers);
  for (int32_t layer_id = 0; layer_id < kNumLayers; ++layer_id) {
    layer_types.emplace_back(is_kda_layer(layer_id)
                                 ? "linear_attention"
                                 : "deepseek_sparse_attention");
    mlp_layer_types.emplace_back(is_dense_layer(layer_id) ? "dense" : "sparse");
  }
  args.layer_types() = std::move(layer_types);
  args.mlp_layer_types() = std::move(mlp_layer_types);
  args.layers_to_capture() = layers_to_capture;
  return args;
}

// Mirrors load_ct_quant_config for the production quantization_config; the
// target and ignore patterns are the checkpoint's own.
QuantArgs make_quant_args() {
  const std::string layers =
      "(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|"
      "26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45)";
  QuantArgs args;
  args.quant_method() = "compressed-tensors";
  args.bits() = 8;
  args.is_sym() = true;
  args.activation_dynamic() = true;
  args.is_compressed_tensors_w8a8_dynamic() = true;
  args.ignored_modules() = {"re:.*lm_head",
                            "re:.*embed_tokens",
                            R"(re:model\.visual)",
                            R"(re:.*\.norm$)",
                            R"(re:.*mlp\.gate$)",
                            "re:.*router$",
                            "re:.*attn_mha$",
                            "re:.*attn_mqa$",
                            "re:.*hyper_connection",
                            "re:.*mapping_proj$",
                            R"(re:.*shared_head\.norm$)",
                            "re:.*eh_proj$",
                            "re:.*downsample$"};
  args.compressed_groups() = {
      CompressedQuantGroup{
          .targets =
              {R"(re:.*mlp\.(gate_proj|up_proj|down_proj)$)",
               R"(re:.*layers\.)" + layers +
                   R"(\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)$)",
               R"(re:.*layers\.(3|7|11|15|19|23|27|31|35|39|43|45)\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)$)"},
          .bits = 8,
          .group_size = 0,
          .preserve_smooth = true},
      CompressedQuantGroup{
          .targets =
              {R"(re:.*layers\.)" + layers +
               R"(\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$)"},
          .bits = 4,
          .group_size = kW4GroupSize,
          .preserve_smooth = true}};
  return args;
}

struct LayerCacheTensors {
  torch::Tensor conv;
  torch::Tensor ssm;
  torch::Tensor key;
  torch::Tensor index;
  torch::Tensor kpool_tail;
};

// Cache tensors laid out as KVCacheShape lays them out for a GLM5-Next MTP
// target with num_speculative_tokens=3: the convolution state keeps
// kernel-1 committed columns followed by one column per speculative token,
// and every linear-state slot owns num_speculative_tokens+1 SSM rows whose
// first row is the committed state.
std::vector<LayerCacheTensors> make_cache_tensors(const torch::Device& device) {
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const int64_t conv_state_len = kConvKernel - 1 + kNumSpeculativeTokens;
  const int64_t checkpoint_stride = kNumSpeculativeTokens + 1;
  const int64_t kpool_tail_len = kIndexKpool + 2 * (kNumSpeculativeTokens + 1);
  std::vector<LayerCacheTensors> tensors;
  tensors.reserve(kNumLayers);
  for (int32_t layer_id = 0; layer_id < kNumLayers; ++layer_id) {
    LayerCacheTensors layer;
    if (is_kda_layer(layer_id)) {
      layer.conv =
          torch::zeros({kLinearStateBlocks,
                        conv_state_len,
                        3 * (kLinearHeads / kWorldSize) * kLinearHeadDim},
                       options);
      layer.ssm = torch::zeros(
          {kLinearStateBlocks * checkpoint_stride,
           kLinearHeads / kWorldSize,
           kLinearHeadDim,
           kLinearHeadDim},
          options.dtype(resolve_ssm_dtype(
              /*mamba_ssm_dtype_str=*/"float32", torch::kBFloat16)));
      layer.conv.narrow(/*dim=*/1, kConvKernel - 1, kNumSpeculativeTokens)
          .fill_(kConvCheckpointSentinel);
      layer.ssm.view({kLinearStateBlocks, checkpoint_stride, -1})
          .narrow(/*dim=*/1, /*start=*/1, kNumSpeculativeTokens)
          .fill_(kSsmCheckpointSentinel);
    } else {
      layer.key =
          torch::zeros({kKvBlocks, 1, kBlockSize, kKvLoraRank}, options);
      layer.index = torch::zeros(
          {kKvBlocks, 1, kBlockSize / kIndexKpool, kIndexHeadDim}, options);
      layer.kpool_tail = torch::zeros(
          {kLinearStateBlocks, 2, kpool_tail_len, kIndexHeadDim}, options);
    }
    tensors.emplace_back(std::move(layer));
  }
  return tensors;
}

std::vector<KVCache> make_caches(
    const std::vector<LayerCacheTensors>& tensors) {
  std::vector<KVCache> caches;
  caches.reserve(tensors.size());
  for (const LayerCacheTensors& layer : tensors) {
    if (layer.conv.defined()) {
      caches.emplace_back(
          KVCache(LinearAttentionKVCacheTensors{layer.conv, layer.ssm}));
      continue;
    }
    IndexedKVCacheTensors indexed;
    indexed.kv_cache_tensors.key_cache = layer.key;
    indexed.index_cache = layer.index;
    indexed.kpool_tail = layer.kpool_tail;
    caches.emplace_back(KVCache(indexed));
  }
  return caches;
}

// Speculative checkpoints are owned by speculative verification; ordinary
// prefill and decode must leave them untouched.
bool checkpoints_untouched(const std::vector<LayerCacheTensors>& tensors) {
  bool untouched = true;
  for (size_t layer_id = 0; layer_id < tensors.size(); ++layer_id) {
    const LayerCacheTensors& layer = tensors[layer_id];
    if (!layer.conv.defined()) {
      continue;
    }
    const bool conv_untouched =
        layer.conv.narrow(/*dim=*/1, kConvKernel - 1, kNumSpeculativeTokens)
            .eq(kConvCheckpointSentinel)
            .all()
            .item<bool>();
    const bool ssm_untouched =
        layer.ssm.view({kLinearStateBlocks, kNumSpeculativeTokens + 1, -1})
            .narrow(/*dim=*/1, /*start=*/1, kNumSpeculativeTokens)
            .eq(kSsmCheckpointSentinel)
            .all()
            .item<bool>();
    LOG_IF(ERROR, !conv_untouched)
        << "Layer " << layer_id << " rewrote speculative conv checkpoints";
    LOG_IF(ERROR, !ssm_untouched)
        << "Layer " << layer_id << " rewrote speculative SSM checkpoints";
    untouched = untouched && conv_untouched && ssm_untouched;
  }
  return untouched;
}

struct SequenceState {
  int32_t linear_state_id = 0;
  int32_t first_block = 0;
  int32_t num_blocks = 0;
  int32_t length = 0;
};

struct HiddenSamples {
  float values[2][4];
};

struct CacheSamples {
  // [sequence][key, compressed index]
  float dsa[2][2];
  // [CP rank][sequence][conv, SSM] for the first logical head on each rank.
  float kda[kWorldSize][2][2];
};

// CP=8 / EP=8 samples from e414171ef on MLU590-M9DK, using kWeightSeed.
constexpr HiddenSamples kPrefill = {
    {{0.330078125f, 1.1015625f, -1.78125f, 0.85546875f},
     {-0.125f, 1.328125f, 1.328125f, -1.515625f}}};
constexpr HiddenSamples kChunk2 = {
    {{1.234375f, -0.80078125f, 0.53515625f, -0.14453125f},
     {-0.296875f, 0.83984375f, -1.8359375f, 1.3046875f}}};
constexpr HiddenSamples kDecode2 = {
    {{0.9609375f, -0.01177978516f, -1.09375f, -1.8046875f},
     {0.0048828125f, -0.447265625f, 0.6953125f, 0.1206054688f}}};
constexpr HiddenSamples kTurn2Decode = {
    {{-0.6015625f, -0.185546875f, -0.703125f, -0.94921875f},
     {0.359375f, -1.3359375f, -1.1875f, 0.1298828125f}}};
constexpr CacheSamples kChunk2Caches = {
    .dsa = {{4.0625f, -0.03247070312f}, {-0.1137695312f, 0.447265625f}},
    .kda = {
        {{2.234375f, 0.0001716940023f}, {-0.6640625f, 0.0007297327975f}},
        {{-0.359375f, 0.007880941033f}, {-0.58203125f, 0.001331661362f}},
        {{-0.00373840332f, 0.002957318211f}, {0.1416015625f, 0.001119454275f}},
        {{0.57421875f, 0.002118875273f}, {0.365234375f, 0.001077918801f}},
        {{1.7734375f, 0.0005943361903f}, {0.2890625f, -0.0008219594019f}},
        {{-0.01165771484f, -0.004967074841f},
         {-0.2099609375f, 0.002834114246f}},
        {{-0.40625f, 0.0007942903903f}, {-0.412109375f, -0.001497511286f}},
        {{-0.7578125f, -0.002814526903f}, {-0.1245117188f, -0.001458446961f}}}};
constexpr CacheSamples kDecode2Caches = {
    .dsa = {{-0.2451171875f, 0.30078125f}, {0.435546875f, 0.3828125f}},
    .kda = {
        {{0.45703125f, 0.002365184948f}, {-0.2451171875f, 0.004235941451f}},
        {{0.8125f, 0.002829528879f}, {-1.5703125f, -0.0009964628844f}},
        {{0.4609375f, 2.004753333e-05f}, {0.3515625f, -0.002220587106f}},
        {{-0.416015625f, -0.01119114645f}, {-0.1513671875f, -0.000190579085f}},
        {{-0.1015625f, -0.001932092477f}, {0.271484375f, -0.00103383814f}},
        {{0.1030273438f, 0.004069101065f}, {0.0908203125f, -0.0003122166381f}},
        {{-0.85546875f, -8.80745938e-05f}, {-1.3828125f, 1.320186675e-05f}},
        {{-0.095703125f, 0.01216405164f}, {0.25390625f, -0.003447745461f}}}};
constexpr float kTurn2Aux[2][3][2] = {{{-0.443359375f, -0.37890625f},
                                       {-0.392578125f, -0.4296875f},
                                       {-0.2470703125f, -0.6015625f}},
                                      {{-1.203125f, -0.9453125f},
                                       {-1.203125f, -0.8671875f},
                                       {-1.1484375f, -0.9296875f}}};

struct Step {
  std::string name;
  BatchForwardType::Value type = BatchForwardType::PREFILL;
  std::vector<int32_t> new_tokens;
  const HiddenSamples* hidden = nullptr;
  const CacheSamples* caches = nullptr;
};

// Every prefill segment gives sequence 0 at least 449 tokens, i.e. eight
// 64-token KDA chunks, so all eight ranks own tokens and PCP engages.
std::vector<Step> make_steps() {
  return {
      {"prefill", BatchForwardType::PREFILL, {512, 200}, &kPrefill},
      {"chunk1", BatchForwardType::CHUNKED_PREFILL, {576, 96}},
      {"chunk2",
       BatchForwardType::CHUNKED_PREFILL,
       {449, 1},
       &kChunk2,
       &kChunk2Caches},
      {"decode0", BatchForwardType::DECODE, {1, 1}},
      {"decode1", BatchForwardType::DECODE, {1, 1}},
      {"decode2", BatchForwardType::DECODE, {1, 1}, &kDecode2, &kDecode2Caches},
      {"turn2_chunk", BatchForwardType::CHUNKED_PREFILL, {512, 130}},
      {"turn2_decode", BatchForwardType::DECODE, {1, 1}, &kTurn2Decode},
  };
}

int32_t token_id_at(size_t sequence, int32_t position) {
  uint64_t state = kWeightSeed ^ (static_cast<uint64_t>(sequence) << 32) ^
                   static_cast<uint64_t>(position);
  return 1 + static_cast<int32_t>(splitmix64(state) % (kVocabSize - 1));
}

struct StepInput {
  torch::Tensor tokens;
  torch::Tensor positions;
  ModelInputParams params;
};

// Fills the fields BatchInputBuilder::state_to_forward_input and
// prepare_input_params_for_linear_attention fill on MLU, where sequence
// lengths are cumulative with a leading zero.
StepInput make_step_input(const Step& step,
                          const std::vector<SequenceState>& sequences,
                          const torch::Device& device) {
  const auto ints = torch::TensorOptions().dtype(torch::kInt32).device(device);
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<int32_t> slots;
  std::vector<int32_t> q_cumulative = {0};
  std::vector<int32_t> kv_cumulative = {0};
  std::vector<int32_t> cached_tokens;
  std::vector<int32_t> linear_state_ids;
  std::vector<int32_t> block_table;
  cached_tokens.reserve(sequences.size());
  linear_state_ids.reserve(sequences.size());
  int32_t table_width = 0;
  for (const SequenceState& state : sequences) {
    table_width = std::max(table_width, state.num_blocks);
  }
  block_table.reserve(sequences.size() * static_cast<size_t>(table_width));
  StepInput input;
  int32_t max_query = 0;
  int32_t max_kv = 0;
  for (size_t sequence = 0; sequence < sequences.size(); ++sequence) {
    const SequenceState& state = sequences[sequence];
    const int32_t count = step.new_tokens[sequence];
    for (int32_t offset = 0; offset < count; ++offset) {
      const int32_t position = state.length + offset;
      tokens.emplace_back(token_id_at(sequence, position));
      positions.emplace_back(position);
      slots.emplace_back(
          (state.first_block + position / static_cast<int32_t>(kBlockSize)) *
              static_cast<int32_t>(kBlockSize) +
          position % static_cast<int32_t>(kBlockSize));
    }
    // Unused entries point at the reserved padding block 0.
    for (int32_t block = 0; block < table_width; ++block) {
      block_table.emplace_back(
          block < state.num_blocks ? state.first_block + block : 0);
    }
    q_cumulative.emplace_back(q_cumulative.back() + count);
    kv_cumulative.emplace_back(kv_cumulative.back() + state.length + count);
    cached_tokens.emplace_back(state.length);
    linear_state_ids.emplace_back(state.linear_state_id);
    input.params.linear_state_validity_mask.emplace_back(state.length > 0 ? 1
                                                                          : 0);
    max_query = std::max(max_query, count);
    max_kv = std::max(max_kv, state.length + count);
  }

  input.tokens = torch::tensor(tokens, ints);
  input.positions = torch::tensor(positions, ints);
  ModelInputParams& params = input.params;
  params.meta.batch_forward_type = BatchForwardType(step.type);
  params.meta.num_sequences = static_cast<int32_t>(sequences.size());
  params.meta.actual_num_sequences = params.meta.num_sequences;
  params.meta.q_max_seq_len = max_query;
  params.meta.kv_max_seq_len = max_kv;
  params.attention.device.q_seq_lens = torch::tensor(q_cumulative, ints);
  params.attention.device.q_cu_seq_lens = torch::tensor(q_cumulative, ints);
  params.attention.device.kv_seq_lens = torch::tensor(kv_cumulative, ints);
  params.attention.device.kv_cache_tokens_nums =
      torch::tensor(cached_tokens, ints);
  params.attention.device.new_cache_slots = torch::tensor(slots, ints);
  params.attention.device.block_tables =
      torch::tensor(block_table, ints)
          .view({static_cast<int64_t>(sequences.size()), table_width});
  params.attention.host.q_seq_lens = q_cumulative;
  params.attention.host.q_cu_seq_lens = std::move(q_cumulative);
  params.attention.host.kv_seq_lens = std::move(kv_cumulative);
  params.attention.host.kv_cache_tokens_nums = std::move(cached_tokens);
  params.attention.host.block_tables =
      params.attention.device.block_tables.cpu();
  params.embedding.linear_state_indices = torch::tensor(linear_state_ids, ints);
  params.embedding.linear_state_ids = std::move(linear_state_ids);
  params.parallel.dp_global_token_nums = {static_cast<int32_t>(tokens.size())};
  return input;
}

void expect_sample(const torch::Tensor& tensor,
                   std::initializer_list<int64_t> coordinates,
                   float expected,
                   const char* name,
                   double tolerance = kDefaultSampleTolerance) {
  torch::Tensor value = tensor;
  for (const int64_t index : coordinates) {
    value = value.select(0, index);
  }
  const float actual = value.item<float>();
  ASSERT_TRUE(std::isfinite(actual))
      << name << " at " << ::testing::PrintToString(coordinates);
  EXPECT_NEAR(actual, expected, tolerance + tolerance * std::abs(expected))
      << name << " at " << ::testing::PrintToString(coordinates);
}

void check_hidden(const Step& step,
                  const torch::Tensor& hidden,
                  const HiddenSamples& expected) {
  const int64_t rows = step.new_tokens[0] + step.new_tokens[1];
  ASSERT_EQ(hidden.sizes(), torch::IntArrayRef({rows, kHiddenSize}));
  ASSERT_EQ(hidden.scalar_type(), torch::kBFloat16);
  constexpr int64_t kChannels[] = {0, 127, 1024, kHiddenSize - 1};
  int64_t row = -1;
  for (int32_t seq = 0; seq < 2; ++seq) {
    SCOPED_TRACE(::testing::Message() << "sequence " << seq);
    row += step.new_tokens[seq];
    for (int32_t i = 0; i < 4; ++i) {
      expect_sample(hidden,
                    {row, kChannels[i]},
                    expected.values[seq][i],
                    "hidden",
                    /*tolerance=*/8e-3);
    }
  }
}

void check_aux(const torch::Tensor& aux, const std::vector<int32_t>& layers) {
  ASSERT_EQ(layers.size(), 3);
  ASSERT_EQ(aux.sizes(), torch::IntArrayRef({2, 3 * kHiddenSize}));
  ASSERT_EQ(aux.scalar_type(), torch::kBFloat16);
  constexpr int64_t kChannels[] = {127, 1024};
  for (int32_t seq = 0; seq < 2; ++seq) {
    for (int32_t slot = 0; slot < 3; ++slot) {
      SCOPED_TRACE(::testing::Message()
                   << "sequence " << seq << " layer " << layers[slot]);
      for (int32_t i = 0; i < 2; ++i) {
        expect_sample(aux,
                      {seq, slot * kHiddenSize + kChannels[i]},
                      kTurn2Aux[seq][slot][i],
                      "aux_hidden",
                      /*tolerance=*/4e-3);
      }
    }
  }
}

void check_caches(const std::vector<LayerCacheTensors>& tensors,
                  const std::vector<SequenceState>& sequences,
                  const CacheSamples& expected,
                  int32_t rank) {
  const LayerCacheTensors& kda = tensors[6];
  const LayerCacheTensors& dsa = tensors[3];
  constexpr int64_t kLocalHeads = kLinearHeads / kWorldSize;
  ASSERT_EQ(kda.conv.sizes(),
            torch::IntArrayRef({kLinearStateBlocks,
                                kConvKernel - 1 + kNumSpeculativeTokens,
                                3 * kLocalHeads * kLinearHeadDim}));
  ASSERT_EQ(
      kda.ssm.sizes(),
      torch::IntArrayRef({kLinearStateBlocks * (1 + kNumSpeculativeTokens),
                          kLocalHeads,
                          kLinearHeadDim,
                          kLinearHeadDim}));
  ASSERT_EQ(dsa.key.sizes(),
            torch::IntArrayRef({kKvBlocks, 1, kBlockSize, kKvLoraRank}));
  ASSERT_EQ(dsa.index.sizes(),
            torch::IntArrayRef(
                {kKvBlocks, 1, kBlockSize / kIndexKpool, kIndexHeadDim}));
  ASSERT_EQ(kda.conv.scalar_type(), torch::kBFloat16);
  ASSERT_EQ(kda.ssm.scalar_type(), torch::kFloat32);
  ASSERT_EQ(dsa.key.scalar_type(), torch::kBFloat16);
  ASSERT_EQ(dsa.index.scalar_type(), torch::kBFloat16);
  for (int32_t seq = 0; seq < 2; ++seq) {
    SCOPED_TRACE(::testing::Message() << "sequence " << seq);
    const auto& state = sequences[seq];
    // Last committed token and last complete pool of four tokens.
    const int64_t pos = state.length - 1;
    const int64_t pool = state.length / kIndexKpool - 1;
    expect_sample(
        dsa.key,
        {state.first_block + pos / kBlockSize, 0, pos % kBlockSize, 127},
        expected.dsa[seq][0],
        "layer3.key");
    const int64_t pools_per_block = kBlockSize / kIndexKpool;
    expect_sample(dsa.index,
                  {state.first_block + pool / pools_per_block,
                   0,
                   pool % pools_per_block,
                   63},
                  expected.dsa[seq][1],
                  "layer3.index");
    expect_sample(kda.conv,
                  {state.linear_state_id, kConvKernel - 2, 17},
                  expected.kda[rank][seq][0],
                  "layer6.conv",
                  /*tolerance=*/4e-3);
    expect_sample(
        kda.ssm,
        {state.linear_state_id * (1 + kNumSpeculativeTokens), 0, 17, 31},
        expected.kda[rank][seq][1],
        "layer6.ssm");
  }
}

// One pass over the scenario. The production pass captures nothing, as the
// MTP deployment does; the capture pass materializes the hidden states after
// the last dense KDA layer, the first DSA+MoE layer and a KDA+MoE layer.
bool run_scenario(const std::string& pass,
                  const std::vector<int32_t>& layers_to_capture,
                  const ParallelArgs& parallel_args,
                  const Weights& weights,
                  Device& device) {
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  const ModelArgs model_args = make_model_args(layers_to_capture);
  const QuantArgs quant_args = make_quant_args();
  const ModelContext context(parallel_args, model_args, quant_args, options);
  Glm5NextModel model(context);
  model->load_state_dict(StateDict(weights));
  model->verify_loaded_weights();

  const std::vector<Step> steps = make_steps();
  const std::vector<LayerCacheTensors> cache_tensors =
      make_cache_tensors(device.unwrap());
  std::vector<KVCache> caches = make_caches(cache_tensors);
  // Sequence 0 reaches 2053 tokens and sequence 1 reaches 431.
  std::vector<SequenceState> sequences = {
      {.linear_state_id = 1, .first_block = 1, .num_blocks = 144, .length = 0},
      {.linear_state_id = 2,
       .first_block = 145,
       .num_blocks = 40,
       .length = 0}};
  CHECK_LE(sequences.back().first_block + sequences.back().num_blocks,
           kKvBlocks);

  bool checkpoints_ok = true;
  for (const Step& step : steps) {
    if (step.type != BatchForwardType::DECODE) {
      // Guards the scenario itself: a rank without tokens would silently
      // route this segment through the non-PCP path.
      const layer::glm5_next_pcp::Geometry geometry =
          layer::glm5_next_pcp::build_geometry(
              step.new_tokens,
              kWorldSize,
              static_cast<int32_t>(kernel::mlu::kda_prefill_chunk_size(
                  kLinearHeads, /*use_qk_l2norm=*/true)));
      CHECK(std::all_of(geometry.tokens_per_rank.begin(),
                        geometry.tokens_per_rank.end(),
                        [](int32_t count) { return count > 0; }))
          << step.name << " does not give every CP rank tokens";
    }
    StepInput input = make_step_input(step, sequences, device.unwrap());
    const ModelOutput output =
        model->forward(input.tokens, input.positions, caches, input.params);
    device.synchronize_default_stream();
    CHECK(output.hidden_states.defined())
        << pass << "." << step.name << " produced no hidden states";
    for (size_t sequence = 0; sequence < sequences.size(); ++sequence) {
      sequences[sequence].length += step.new_tokens[sequence];
      CHECK_LE(
          sequences[sequence].length,
          sequences[sequence].num_blocks * static_cast<int32_t>(kBlockSize));
    }
    const std::string name = pass + "." + step.name;
    SCOPED_TRACE(::testing::Message()
                 << "rank " << parallel_args.rank() << " " << name);
    if (step.hidden != nullptr && layers_to_capture.empty()) {
      check_hidden(step, output.hidden_states, *step.hidden);
    }
    if (!layers_to_capture.empty() && step.name == "turn2_decode") {
      check_aux(output.aux_hidden_states, layers_to_capture);
    }
    if (step.caches != nullptr && layers_to_capture.empty()) {
      check_caches(
          cache_tensors, sequences, *step.caches, parallel_args.rank());
    }
    const bool untouched = checkpoints_untouched(cache_tensors);
    LOG_IF(ERROR, !untouched)
        << name << " rewrote uncommitted checkpoints on rank "
        << parallel_args.rank();
    checkpoints_ok = checkpoints_ok && untouched;
  }
  return checkpoints_ok;
}

std::unique_ptr<ProcessGroup> create_rank_group(int32_t rank,
                                                int32_t port,
                                                bool trans,
                                                const std::string& name,
                                                const torch::Device& device) {
  std::unique_ptr<ProcessGroup> group = create_process_group(
      rank, kWorldSize, kWorldSize, port, trans, "127.0.0.1", name, device);
  // TCPStore construction does not wait for every peer; one collective keeps
  // the server alive until all ranks have joined.
  torch::Tensor rendezvous = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  group->allreduce(rendezvous);
  CHECK_EQ(rendezvous.item<float>(), static_cast<float>(kWorldSize))
      << "Process group " << name << " did not reach every rank";
  return group;
}

int32_t run_rank(int32_t rank, int32_t cp_port, int32_t ep_port) {
  // Bounds a stuck collective, including a peer that died during setup.
  ::alarm(kChildTimeoutSeconds);
  if (Platform::device_count() < kWorldSize) {
    return kExitSkip;
  }
  torch::set_num_threads(1);
  torch::InferenceMode inference_mode;
  Device device(rank);
  device.set_device();
  KVCacheConfig::get_instance().block_size(kBlockSize);
  SchedulerConfig::get_instance().max_tokens_per_batch(kMaxTokensPerBatch);
  EPLBConfig::get_instance().expert_parallel_degree(1);

  // Production wiring for dp=1, cp=8, tp=1, ep=8: separate eight-rank CP and
  // MoE EP groups, single-rank TP groups, and kv_split_size=1 so every rank
  // keeps the full KV cache.
  std::unique_ptr<ProcessGroup> cp_group =
      create_rank_group(rank,
                        cp_port,
                        /*trans=*/false,
                        "glm5_next_pcp_cp_group",
                        device.unwrap());
  std::unique_ptr<ProcessGroup> ep_group = create_rank_group(
      rank, ep_port, /*trans=*/true, "glm5_next_pcp_ep_group", device.unwrap());
  ProcessGroup single_rank(/*rank=*/0, /*world_size=*/1, device.unwrap());
  ParallelArgs parallel_args(rank,
                             kWorldSize,
                             /*dp_size=*/1,
                             /*cp_size=*/kWorldSize,
                             cp_group.get(),
                             /*ep_size=*/kWorldSize);
  parallel_args.kv_split_size() = 1;
  parallel_args.tp_size() = 1;
  parallel_args.cp_group_ = cp_group.get();
  parallel_args.moe_ep_group_ = ep_group.get();
  parallel_args.tp_group_ = &single_rank;
  parallel_args.single_rank_group_ = &single_rank;
  parallel_args.moe_tp_group_ = &single_rank;
  parallel_args.dcp_group_ = &single_rank;
  CHECK_EQ(parallel_args.cp_rank(), rank);
  CHECK_EQ(parallel_args.kv_split_size_effective(), 1);

  const Weights weights = make_weights(device.unwrap());
  const bool production_ok = run_scenario("production",
                                          /*layers_to_capture=*/{},
                                          parallel_args,
                                          weights,
                                          device);
  const bool capture_ok = run_scenario("capture",
                                       /*layers_to_capture=*/{2, 3, 6},
                                       parallel_args,
                                       weights,
                                       device);
  return production_ok && capture_ok && !::testing::Test::HasFailure()
             ? 0
             : kExitMismatch;
}

TEST(Glm5NextMluPcpRegressionTest, EightRanksMatchOutputSamples) {
  const int32_t cp_port = net::get_local_free_port();
  const int32_t ep_port = net::get_local_free_port();
  ASSERT_GT(cp_port, 0);
  ASSERT_GT(ep_port, 0);
  ASSERT_NE(cp_port, ep_port);

  // Fork before any MLU initialization in this process.
  std::vector<pid_t> children;
  children.reserve(kWorldSize);
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    const pid_t child = ::fork();
    ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
    if (child == 0) {
      ::_exit(run_rank(rank, cp_port, ep_port));
    }
    children.emplace_back(child);
  }

  int32_t skipped = 0;
  int32_t failed = 0;
  for (size_t rank = 0; rank < children.size(); ++rank) {
    int32_t status = 0;
    pid_t waited = -1;
    do {
      waited = ::waitpid(children[rank], &status, /*options=*/0);
    } while (waited < 0 && errno == EINTR);
    ASSERT_EQ(waited, children[rank]);
    const int32_t exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    skipped += exit_code == kExitSkip ? 1 : 0;
    failed += exit_code != 0 && exit_code != kExitSkip ? 1 : 0;
    if (!WIFEXITED(status)) {
      ADD_FAILURE() << "Rank " << rank << " terminated by signal "
                    << (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    } else if (exit_code != 0 && exit_code != kExitSkip) {
      ADD_FAILURE() << "Rank " << rank << " exited with code " << exit_code;
    }
  }
  if (skipped == kWorldSize) {
    GTEST_SKIP() << "Requires " << kWorldSize << " MLU devices.";
  }
  ASSERT_EQ(skipped, 0);
  ASSERT_EQ(failed, 0);
}

}  // namespace
}  // namespace xllm::mlu::model
