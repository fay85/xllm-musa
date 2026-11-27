/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <algorithm>
#include <cstdint>
#include <memory>

#include "core/framework/config/kernel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/dit_cache/dit_cache.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "models/dit/autoencoders/autoencoder_kl_flux2.h"
#include "models/dit/pipelines/pipeline_flux2_base.h"
#include "models/dit/processors/flux2_image_processor.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/transformers/transformer_flux2.h"
#include "models/dit/utils/dit_parallel_mixin.h"
#include "models/llm/qwen3.h"

namespace xllm {

using Flux2KleinTextEncoder = QWen3ForCausalLM;

class Flux2KleinPipelineImpl final : public Flux2PipelineBaseImpl,
                                     public dit::CFGParallelMixin {
 public:
  explicit Flux2KleinPipelineImpl(const DiTModelContext& context)
      : dit::CFGParallelMixin(context),
        parallel_args_(context.get_parallel_args()) {
    const auto& model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();
    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);

    tokenizer_max_length_ = 512;
    is_distilled_ = true;

    flux2_image_processor_ = Flux2ImageProcessor(
        context.get_model_context("vae"), vae_scale_factor_ * 2);
    vae_ = AutoencoderKLFlux2(context.get_model_context("vae"));
    pos_embed_ = register_module(
        "pos_embed",
        Flux2PosEmbed(context.get_model_args("transformer").rope_theta(),
                      context.get_model_args("transformer").axes_dims_rope()));

    transformer_ = Flux2DiTModel(context.get_model_context("transformer"),
                                 context.get_parallel_args());
    num_layers_ = context.get_model_args("transformer").num_layers();
    num_single_layers_ =
        context.get_model_args("transformer").num_single_layers();
    scheduler_ =
        FlowMatchEulerDiscreteScheduler(context.get_model_context("scheduler"));

    if (context.has_component("text_encoder")) {
#if defined(USE_NPU)
      CHECK_EQ(KernelConfig::get_instance().npu_kernel_backend(), "TORCH")
          << "Flux2Klein in-process Qwen3 text encoding requires "
             "--npu_kernel_backend=TORCH.";
#endif
      ProcessGroup* tp_group = parallel_args_.dit_text_encoder_tp_group_;
      CHECK(tp_group != nullptr)
          << "Flux2Klein requires an injected text encoder TP group.";
      ParallelArgs te_parallel_args(
          tp_group->rank(), tp_group->world_size(), tp_group);
      te_parallel_args.tp_size(tp_group->world_size());
      te_parallel_args.tp_group_ = tp_group;
      text_encoder_empty_kv_caches_.resize(static_cast<size_t>(
          context.get_model_args("text_encoder").n_layers()));
      const ModelContext& source_te_context =
          context.get_model_context("text_encoder");

      ModelArgs te_model_args = source_te_context.get_model_args();
      te_model_args.embedding_mode(true);
      CHECK_GT(te_model_args.n_layers(), 26)
          << "Flux2Klein requires Qwen3 decoder layers 9, 18, and 27 for "
             "auxiliary hidden-state capture.";
      // AuxHiddenCapture uses zero-based post-layer indices. Keep the layer
      // order aligned with diffusers' 9th, 18th, and 27th hidden states.
      te_model_args.layers_to_capture({8, 17, 26});
      ModelContext te_context(te_parallel_args,
                              te_model_args,
                              source_te_context.get_quant_args(),
                              source_te_context.get_tensor_options());
      te_context.set_model_id(context.model_type());
      text_encoder_ = Flux2KleinTextEncoder(te_context);
      CHECK(!text_encoder_.is_empty())
          << "Failed to create Flux2Klein Qwen3 text encoder";
    }

    register_module("vae", vae_);
    register_module("flux2_image_processor", flux2_image_processor_);
    register_module("transformer", transformer_);
    register_module("scheduler", scheduler_);
    if (!text_encoder_.is_empty()) {
      register_module("text_encoder", text_encoder_);
    }
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;

    int64_t seed = generation_params.seed >= 0 ? generation_params.seed : 42;
    auto prompts = std::make_optional(input.prompts);
    std::optional<torch::Tensor> latents = input.tensor_sources.get("latent");
    std::optional<torch::Tensor> prompt_embeds =
        input.tensor_sources.get("prompt_embed");
    std::optional<torch::Tensor> negative_prompt_embeds =
        input.tensor_sources.get("negative_prompt_embed");
    std::vector<torch::Tensor> image_sources =
        input.media_sources.get({"image"});
    std::optional<torch::Tensor> images =
        image_sources.empty() ? std::nullopt
                              : std::make_optional(image_sources[0]);

    // Encode prompts in-process when the Qwen3 text encoder is loaded and no
    // pre-computed embeddings are provided.
    if (!prompt_embeds.has_value() && !input.prompts.empty() &&
        !text_encoder_.is_empty()) {
      prompt_embeds =
          encode_prompts(input.prompts, generation_params.max_sequence_length);
    }
    if (!negative_prompt_embeds.has_value() &&
        !input.negative_prompts.empty() && !text_encoder_.is_empty()) {
      CHECK_EQ(input.negative_prompts.size(), input.prompts.size())
          << "Flux2Klein negative prompt batch must match prompt batch";
      negative_prompt_embeds = encode_prompts(
          input.negative_prompts, generation_params.max_sequence_length);
    }

    auto output = forward_impl(
        prompts,                                  // prompt
        generation_params.height,                 // height
        generation_params.width,                  // width
        generation_params.num_inference_steps,    // num_inference_steps
        generation_params.true_cfg_scale,         // guidance_scale
        generation_params.num_images_per_prompt,  // num_images_per_prompt
        seed,                                     // seed
        latents,                                  // latents
        prompt_embeds,                            // prompt_embeds
        negative_prompt_embeds,                   // negative_prompt_embeds
        images,                                   // images
        generation_params.max_sequence_length     // max_sequence_length
    );

    DiTForwardOutput out;
    out.tensors = torch::chunk(output, input.batch_size);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    std::string model_path = loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    std::unique_ptr<DiTFolderLoader> text_encoder_loader;
    if (!text_encoder_.is_empty()) {
      text_encoder_loader = loader->take_component_loader("text_encoder");
      // In the diffusers layout the tokenizer files live in the separate
      // "tokenizer" component folder (Qwen2TokenizerFast); the text_encoder
      // folder only holds weights and config.
      auto tokenizer_loader = loader->take_component_loader("tokenizer");
      std::unique_ptr<Tokenizer> tokenizer = tokenizer_loader->tokenizer();
      CHECK(tokenizer != nullptr)
          << "Failed to load Flux2Klein Qwen3 tokenizer";
      tokenizer_ = std::move(tokenizer);
    }
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device());
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device());
    if (text_encoder_loader != nullptr) {
      text_encoder_->load_model(std::move(text_encoder_loader));
      text_encoder_->to(options_.device());
    }
  }

 private:
  // Qwen3 chat-template prompt (diffusers Flux2Klein wraps the user prompt
  // with add_generation_prompt=True and enable_thinking=False). With
  // enable_thinking=False the official Qwen3 template still appends the
  // empty-think marker "<think>\n\n</think>\n\n" (4 tokens);
  std::string build_qwen3_chat_prompt(const std::string& prompt) const {
    return "<|im_start|>user\n" + prompt +
           "<|im_end|>\n<|im_start|>assistant\n" + "<think>\n\n</think>\n\n";
  }

  // Causal + padding-aware attention mask for the in-process Qwen3 text
  // encoder. FIA int8 convention: 1 = block, 0 = attend.
  torch::Tensor build_text_attention_mask(int64_t seq_len,
                                          int64_t prompt_length,
                                          const torch::Device& device) const {
    auto mask_opts = torch::TensorOptions().dtype(torch::kInt8).device(device);
    torch::Tensor mask = torch::zeros({seq_len, seq_len}, mask_opts);
    torch::Tensor upper = torch::ones({seq_len, seq_len}, mask_opts);
    upper.triu_(1);
    mask.add_(upper);

    // Block only the columns added for right padding. Token values in the real
    // prompt do not affect the attention mask.
    if (prompt_length < seq_len) {
      mask.slice(/*dim=*/1, prompt_length, seq_len).fill_(kAttnMaskBlock);
    }

    // FIA allMask requires mask dims of at least 512; extra rows/cols are
    // bounded by the sequence lengths and set to block as a safe default.
    constexpr int64_t kMinMaskDim = 512;
    if (seq_len < kMinMaskDim) {
      torch::Tensor block_cols =
          torch::ones({seq_len, kMinMaskDim - seq_len}, mask_opts);
      mask = torch::cat({mask, block_cols}, /*dim=*/1);
      torch::Tensor block_rows =
          torch::ones({kMinMaskDim - seq_len, mask.size(1)}, mask_opts);
      mask = torch::cat({mask, block_rows}, /*dim=*/0);
    }
    return mask;
  }

  // Build ModelInputParams and FIA metadata for the in-process Qwen3 text
  // encoder. Qwen3 reuses the supplied metadata instead of rebuilding it.
  ModelInputParams build_text_encoder_input(
      const torch::Tensor& tokens,
      const torch::Tensor& attention_mask) const {
    const int32_t sequence_length = static_cast<int32_t>(tokens.numel());

    ModelInputParams params;
    params.meta.num_sequences = 1;
    params.meta.actual_num_sequences = 1;
    params.meta.q_max_seq_len = sequence_length;
    params.meta.kv_max_seq_len = sequence_length;
    params.meta.batch_forward_type = BatchForwardType::PREFILL;
    params.prefill_without_cache = true;
    params.attention.host.q_seq_lens = {sequence_length};
    params.attention.host.kv_seq_lens = {sequence_length};
#if defined(USE_NPU)
    params.attention.host.q_cu_seq_lens = {sequence_length};
#else
    params.attention.host.q_cu_seq_lens = {0, sequence_length};
#endif
    params.attention.device.q_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
    params.attention.device.kv_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
#if defined(USE_NPU)
    params.attention.device.q_cu_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
#else
    params.attention.device.q_cu_seq_lens =
        torch::tensor({0, sequence_length}, torch::kInt).to(tokens.device());
#endif
    params.graph.attn_mask = attention_mask;
    params.attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(params, /*enable_mla=*/false));
#if defined(USE_NPU)
    // aclnnFusedInferAttentionScoreV3 with TND layout and sparse_mode=1
    // (allMask) requires a 4D [1, 1, q, kv] int8 mask (1 = block, 0 = attend).
    torch::Tensor fia_attn_mask = params.graph.attn_mask.to(torch::kInt8);
    if (fia_attn_mask.dim() == 2) {
      fia_attn_mask = fia_attn_mask.unsqueeze(0).unsqueeze(0);
    }
    params.attn_metadata->fia_attn_mask = fia_attn_mask.contiguous();
    params.attn_metadata->fia_sparse_mode = 1;
#endif
    return params;
  }

  torch::Tensor encode_prompts(const std::vector<std::string>& prompts,
                               int64_t max_sequence_length) {
    CHECK(!text_encoder_.is_empty()) << "Qwen3 text encoder is not loaded";
    CHECK(tokenizer_ != nullptr) << "Qwen3 tokenizer is not loaded";

    const int64_t pad_length =
        max_sequence_length > 0 ? max_sequence_length : tokenizer_max_length_;
    const int64_t capture_capacity =
        SchedulerConfig::get_instance().max_tokens_per_batch();
    CHECK_LE(pad_length, capture_capacity)
        << "Flux2Klein prompt sequence length " << pad_length
        << " exceeds the text encoder capture capacity " << capture_capacity
        << "; reduce --max_sequence_length or increase "
           "--max_tokens_per_batch.";
    const torch::Device device = options_.device();
    std::vector<torch::Tensor> prompt_embeds_list;
    prompt_embeds_list.reserve(prompts.size());
    for (const std::string& prompt : prompts) {
      std::vector<int32_t> token_ids;
      CHECK(tokenizer_->encode(build_qwen3_chat_prompt(prompt), &token_ids))
          << "Failed to tokenize Flux2Klein prompt";
      const int64_t prompt_length =
          std::min(static_cast<int64_t>(token_ids.size()), pad_length);
      // Right-pad with the Qwen3 pad token (truncates when oversized,
      // matching diffusers padding="max_length", truncation=True).
      token_ids.resize(static_cast<size_t>(pad_length), kQwen3PadTokenId);

      torch::Tensor tokens =
          torch::tensor(token_ids, torch::TensorOptions().dtype(torch::kInt32))
              .to(device);
      torch::Tensor positions =
          torch::arange(
              0, pad_length, torch::TensorOptions().dtype(torch::kInt32))
              .to(device);
      torch::Tensor attention_mask =
          build_text_attention_mask(pad_length, prompt_length, device);
      ModelInputParams input_params =
          build_text_encoder_input(tokens, attention_mask);
      ModelOutput output = text_encoder_->forward(
          tokens, positions, text_encoder_empty_kv_caches_, input_params);
      const torch::Tensor& aux_hidden_states = output.aux_hidden_states;
      CHECK(aux_hidden_states.defined())
          << "Flux2Klein Qwen3 text encoder did not return aux hidden states";
      CHECK_EQ(aux_hidden_states.dim(), 2)
          << "Flux2Klein Qwen3 aux hidden states must be [tokens, 3 * hidden]";
      CHECK_EQ(aux_hidden_states.size(0), pad_length)
          << "Flux2Klein Qwen3 aux hidden token count mismatch";
      const int64_t hidden_size = output.hidden_states.size(-1);
      CHECK_EQ(aux_hidden_states.size(1), kQwen3AuxLayerCount * hidden_size)
          << "Flux2Klein Qwen3 text encoder aux hidden width mismatch";

      // Materialize the permuted view because AuxHiddenCapture reuses its
      // internal buffer on the next prompt's forward pass.
      torch::Tensor prompt_embeds =
          aux_hidden_states.view({pad_length, kQwen3AuxLayerCount, hidden_size})
              .permute({1, 0, 2})
              .contiguous();
      prompt_embeds_list.push_back(std::move(prompt_embeds));
    }
    return torch::stack(prompt_embeds_list, 0);  // [B, 3, seq_len, hidden_size]
  }

  // Prepare noise latents and their 4-D position IDs.
  std::pair<torch::Tensor, torch::Tensor> prepare_latents(
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      std::optional<torch::Tensor> latents = std::nullopt) {
    int64_t adjusted_height = 2 * (height / (vae_scale_factor_ * 2));
    int64_t adjusted_width = 2 * (width / (vae_scale_factor_ * 2));
    std::vector<int64_t> shape = {batch_size,
                                  num_channels_latents * 4,
                                  adjusted_height / 2,
                                  adjusted_width / 2};
    if (latents.has_value()) {
      torch::Tensor latent_image_ids =
          prepare_latent_image_ids(latents.value());
      return {latents.value(), latent_image_ids};
    }
    torch::Tensor latents_tensor =
        xllm::dit::randn_tensor(shape, seed, options_);
    torch::Tensor packed_latents = pack_latents(latents_tensor);
    torch::Tensor latent_image_ids = prepare_latent_image_ids(latents_tensor);
    return {packed_latents, latent_image_ids};
  }

  std::pair<torch::Tensor, torch::Tensor> prepare_image_latents(
      const std::vector<torch::Tensor>& images,
      int64_t batch_size,
      int64_t seed) {
    std::vector<torch::Tensor> image_latents;
    image_latents.reserve(images.size());
    for (const torch::Tensor& image : images) {
      torch::Tensor image_latent =
          vae_->encode(image.to(options_.dtype()), seed);
      torch::Tensor patched_latent = patchify_latents(image_latent);
      torch::Tensor latents_bn_mean =
          vae_->get_bn_running_mean()
              .view({1, -1, 1, 1})
              .to(image_latent.device(), image_latent.dtype());
      torch::Tensor latents_bn_std =
          torch::sqrt(vae_->get_bn_running_var().view({1, -1, 1, 1}) +
                      vae_->get_batch_norm_eps());
      image_latent = (patched_latent - latents_bn_mean) / latents_bn_std;
      image_latents.emplace_back(image_latent);
    }
    torch::Tensor concatenated_latents = pack_latents_for_images(image_latents);
    torch::Tensor image_latent_ids = _prepare_image_ids(image_latents);

    concatenated_latents = concatenated_latents.unsqueeze(0);
    torch::Tensor repeated_latents =
        concatenated_latents.repeat({batch_size, 1, 1});
    torch::Tensor repeated_ids = image_latent_ids.repeat({batch_size, 1, 1});
    return {repeated_latents, repeated_ids};
  }

  torch::Tensor pack_latents_for_images(
      const std::vector<torch::Tensor>& image_latents) {
    std::vector<torch::Tensor> packed_latents;
    packed_latents.reserve(image_latents.size());
    for (const torch::Tensor& latent : image_latents) {
      torch::Tensor packed = pack_latents(latent);
      packed = packed.squeeze(0);
      packed_latents.emplace_back(packed);
    }
    return torch::cat(packed_latents, 0);
  }

  torch::Tensor unpack_latents_with_ids(const torch::Tensor& latents,
                                        const torch::Tensor& latent_ids,
                                        int64_t height,
                                        int64_t width) {
    int64_t batch_size = latents.size(0);
    int64_t channels = latents.size(2);

    std::vector<torch::Tensor> x_list;
    x_list.reserve(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      torch::Tensor data = latents[i];
      torch::Tensor pos = latent_ids[i];

      torch::Tensor h_ids = pos.select(1, 1).to(torch::kInt64);
      torch::Tensor w_ids = pos.select(1, 2).to(torch::kInt64);

      int64_t h = height;
      int64_t w = width;

      torch::Tensor flat_ids = h_ids * w + w_ids;

      torch::Tensor out = torch::zeros({h * w, channels}, data.options());
      out.scatter_(0, flat_ids.unsqueeze(1).expand({-1, channels}), data);

      out = out.view({h, w, channels}).permute({2, 0, 1});
      x_list.push_back(out);
    }

    return torch::stack(x_list, 0);
  }

  // Main inference logic.
  torch::Tensor forward_impl(
      std::optional<std::vector<std::string>> prompt,
      int64_t height = 512,
      int64_t width = 512,
      int64_t num_inference_steps = 4,
      float true_cfg_scale = 1.0f,
      int64_t num_images_per_prompt = 1,
      std::optional<int64_t> seed = std::nullopt,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> images = std::nullopt,
      int64_t max_sequence_length = 512) {
    torch::NoGradGuard no_grad;

    int64_t batch_size = prompt_embeds.value().size(0);
    int64_t total_batch_size = batch_size * num_images_per_prompt;

    int64_t num_channels = prompt_embeds.value().size(1);
    int64_t seq_len = prompt_embeds.value().size(2);
    int64_t hidden_dim = prompt_embeds.value().size(3);

    // (B, num_channels, seq_len, hidden_dim) → (B, seq_len, num_channels *
    // hidden_dim)
    auto prompt_embeds_value =
        prompt_embeds.value()
            .permute({0, 2, 1, 3})
            .reshape({batch_size, seq_len, num_channels * hidden_dim});

    prompt_embeds_value =
        prompt_embeds_value.repeat({1, num_images_per_prompt, 1});
    prompt_embeds_value = prompt_embeds_value.view(
        {batch_size * num_images_per_prompt, seq_len, -1});

    device_ = options_.device();

    torch::Tensor encoded_prompt_embeds = prompt_embeds_value;
    torch::Tensor text_ids = prepare_text_ids(prompt_embeds_value);
    encoded_prompt_embeds =
        encoded_prompt_embeds.to(device_).to(torch::kBFloat16);
    text_ids = text_ids.to(device_).to(torch::kLong);

    // Classifier-free guidance: prepare negative prompt embeddings.
    bool do_cfg = (true_cfg_scale > 1.0f) && !is_distilled_;
    torch::Tensor neg_prompt_embeds;
    torch::Tensor neg_text_ids;
    if (do_cfg) {
      CHECK(negative_prompt_embeds.has_value())
          << "Flux2Klein CFG requires negative prompt embeddings";
      int64_t neg_num_ch = negative_prompt_embeds.value().size(1);
      int64_t neg_seq_len = negative_prompt_embeds.value().size(2);
      int64_t neg_hidden_dim = negative_prompt_embeds.value().size(3);

      neg_prompt_embeds =
          negative_prompt_embeds.value()
              .permute({0, 2, 1, 3})
              .reshape({batch_size, neg_seq_len, neg_num_ch * neg_hidden_dim});
      neg_prompt_embeds =
          neg_prompt_embeds.repeat({1, num_images_per_prompt, 1});
      neg_prompt_embeds = neg_prompt_embeds.view(
          {batch_size * num_images_per_prompt, neg_seq_len, -1});
      neg_text_ids = prepare_text_ids(neg_prompt_embeds);
      neg_prompt_embeds = neg_prompt_embeds.to(device_).to(torch::kBFloat16);
      neg_text_ids = neg_text_ids.to(device_).to(torch::kLong);
    }

    std::vector<torch::Tensor> condition_images_list;
    if (images.has_value()) {
      auto input_images = images.value();
      if (input_images.dim() == 3) {
        input_images = input_images.unsqueeze(0);
      }

      for (int64_t i = 0; i < input_images.size(0); ++i) {
        auto img = input_images[i];
        flux2_image_processor_->check_image_input(img);
        int64_t image_width = img.size(-1);
        int64_t image_height = img.size(-2);

        if (image_width * image_height > 1024 * 1024) {
          img = flux2_image_processor_->resize_to_target_area(img, 1024 * 1024);
          image_width = img.size(-1);
          image_height = img.size(-2);
        }
        int64_t multiple_of = vae_scale_factor_ * 2;
        image_width = (image_width / multiple_of) * multiple_of;
        image_height = (image_height / multiple_of) * multiple_of;
        img =
            flux2_image_processor_->preprocess(img, image_height, image_width);
        condition_images_list.push_back(img);
      }
    }

    int64_t num_channels_latents = transformer_->in_channels() / 4;
    auto [prepared_latents, latent_image_ids] =
        prepare_latents(total_batch_size,
                        num_channels_latents,
                        height,
                        width,
                        seed.has_value() ? seed.value() : 42,
                        latents);
    prepared_latents = prepared_latents.to(torch::kBFloat16);
    latent_image_ids = latent_image_ids.to(torch::kLong);

    torch::Tensor image_latents;
    torch::Tensor image_latent_ids;
    if (!condition_images_list.empty()) {
      std::tie(image_latents, image_latent_ids) =
          prepare_image_latents(condition_images_list,
                                total_batch_size,
                                seed.has_value() ? seed.value() : 42);
    }

    // Klein uses linspace(1.0, 1/N, N) as default sigmas.
    std::vector<float> new_sigmas;
    new_sigmas.reserve(num_inference_steps);
    if (num_inference_steps == 1) {
      new_sigmas.emplace_back(1.0f);
    } else {
      for (int64_t i = 0; i < num_inference_steps; ++i) {
        new_sigmas.emplace_back(1.0f - static_cast<float>(i) /
                                           (num_inference_steps - 1) *
                                           (1.0f - 1.0f / num_inference_steps));
      }
    }

    int64_t image_seq_len = prepared_latents.size(1);
    float mu = compute_empirical_mu(image_seq_len, num_inference_steps);
    auto [timesteps, num_inference_steps_actual] = flux2_retrieve_timesteps(
        scheduler_, num_inference_steps, options_.device(), new_sigmas, mu);

    auto [rot_emb1, rot_emb2] =
        pos_embed_->forward_cache(text_ids,
                                  latent_image_ids,
                                  height / (vae_scale_factor_ * 2),
                                  width / (vae_scale_factor_ * 2));

    torch::Tensor image_rotary_emb = torch::stack({rot_emb1, rot_emb2}, 0);

    int64_t latent_height = 2 * (height / (vae_scale_factor_ * 2));
    int64_t latent_width = 2 * (width / (vae_scale_factor_ * 2));

    DiTCache::get_instance().set_context(
        {/*infer_steps=*/num_inference_steps,
         /*num_blocks=*/num_layers_ + num_single_layers_});
    scheduler_->set_begin_index(0);
    torch::Tensor timestep =
        torch::empty({prepared_latents.size(0)}, prepared_latents.options());

    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i].unsqueeze(0);
      timestep.fill_(t.item<float>())
          .to(prepared_latents.dtype())
          .div_(1000.0f);

      // Build model input: noise latents [+ optional image condition tokens].
      torch::Tensor latent_model_input = prepared_latents.to(options_.dtype());
      torch::Tensor latent_image_ids_input = latent_image_ids;
      if (image_latents.defined()) {
        latent_model_input = torch::cat({prepared_latents, image_latents}, 1)
                                 .to(options_.dtype());
        latent_image_ids_input =
            torch::cat({latent_image_ids, image_latent_ids}, 1);
        auto [rot_emb1_img, rot_emb2_img] =
            pos_embed_->forward_cache(text_ids,
                                      latent_image_ids_input,
                                      height / (vae_scale_factor_ * 2),
                                      width / (vae_scale_factor_ * 2));
        image_rotary_emb = torch::stack({rot_emb1_img, rot_emb2_img}, 0);
      }

      torch::Tensor noise_pred;
      if (do_cfg) {
        // CFG parallel: exec_with_cfg dispatches positive (rank 0) and
        // negative (rank 1) forward passes, then allgathers results.
        // When cfg_size == 1, both passes run sequentially on the same rank.
        auto [pos_noise, neg_noise] = exec_with_cfg([&](bool is_positive) {
          auto& embeds =
              is_positive ? encoded_prompt_embeds : neg_prompt_embeds;
          torch::Tensor pred =
              transformer_->forward(latent_model_input,
                                    embeds,
                                    timestep,
                                    /*guidance=*/torch::Tensor(),
                                    image_rotary_emb,
                                    /*step_idx=*/i + 1);
          // Truncate to noise-latent tokens (exclude image condition tokens).
          if (image_latents.defined()) {
            pred = pred.narrow(1, 0, prepared_latents.size(1));
          }
          return pred;
        });

        // CFG formula in float32 for precision:
        noise_pred =
            neg_noise.to(torch::kFloat32) +
            static_cast<float>(true_cfg_scale) *
                (pos_noise.to(torch::kFloat32) - neg_noise.to(torch::kFloat32));
      } else {
        // No CFG: single forward pass with positive embeddings.
        noise_pred = transformer_->forward(latent_model_input,
                                           encoded_prompt_embeds,
                                           timestep,
                                           /*guidance=*/torch::Tensor(),
                                           image_rotary_emb,
                                           /*step_idx=*/i + 1);

        // Truncate to noise-latent tokens (exclude image condition tokens).
        if (image_latents.defined()) {
          noise_pred = noise_pred.narrow(1, 0, prepared_latents.size(1));
        }
      }

      noise_pred = noise_pred.to(prepared_latents.dtype());

      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);

      prepared_latents = prev_latents.detach();

      noise_pred.reset();
      prev_latents = torch::Tensor();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }

    torch::Tensor unpacked_latents = unpack_latents_with_ids(prepared_latents,
                                                             latent_image_ids,
                                                             latent_height / 2,
                                                             latent_width / 2);
    auto latents_bn_mean =
        vae_->get_bn_running_mean()
            .view({1, -1, 1, 1})
            .to(unpacked_latents.device(), unpacked_latents.dtype());
    auto latents_bn_std =
        torch::sqrt(vae_->get_bn_running_var().view({1, -1, 1, 1}) +
                    vae_->get_batch_norm_eps())
            .to(unpacked_latents.device(), unpacked_latents.dtype());
    unpacked_latents = unpacked_latents * latents_bn_std + latents_bn_mean;
    unpacked_latents = unpatchify_latents(unpacked_latents);

    torch::Tensor image = vae_->decode(unpacked_latents.to(options_.dtype()));

    image = flux2_image_processor_->postprocess(image);

    return image;
  }

 private:
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  AutoencoderKLFlux2 vae_{nullptr};
  Flux2ImageProcessor flux2_image_processor_{nullptr};
  Flux2DiTModel transformer_{nullptr};
  int32_t tokenizer_max_length_;
  int32_t vae_scale_factor_;
  int64_t num_layers_;
  int64_t num_single_layers_;
  bool is_distilled_;
  Flux2PosEmbed pos_embed_{nullptr};

  Flux2KleinTextEncoder text_encoder_{nullptr};
  std::vector<KVCache> text_encoder_empty_kv_caches_;
  const ParallelArgs parallel_args_;

  static constexpr int32_t kQwen3PadTokenId = 151643;
  static constexpr int8_t kAttnMaskBlock = 1;
  static constexpr int64_t kQwen3AuxLayerCount = 3;
};
TORCH_MODULE(Flux2KleinPipeline);

REGISTER_DIT_MODEL(Flux2KleinPipeline, Flux2KleinPipeline);
};  // namespace xllm
