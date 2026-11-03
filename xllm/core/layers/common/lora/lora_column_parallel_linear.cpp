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

#include "layers/common/lora/lora_column_parallel_linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "framework/lora/lora_context.h"
#include "framework/lora/lora_runtime.h"
#if defined(USE_NPU)
#include "framework/lora/ascendc/ascendc_lora_storage.h"
#include "framework/lora/ascendc/xllm_lora_ascendc_binding.h"
#endif  // USE_NPU

namespace xllm {
namespace layer {

LoRAColumnParallelLinearImpl::LoRAColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const std::string& proj_name,
    const LinearExtraArgs& linear_extra_args)
    : proj_name_(proj_name),
      in_features_(in_features),
      out_features_(out_features),
      is_fused_gate_up_(proj_name == "gate_up_proj") {
  base_ = ColumnParallelLinear(in_features,
                               out_features,
                               bias,
                               gather_output,
                               quant_args,
                               process_group,
                               options,
                               linear_extra_args);

  // out_features here is the total; per-rank it is out_features / world_size.
  tp_rank_ = process_group ? process_group->rank() : 0;
  tp_world_size_ = process_group ? process_group->world_size() : 1;
  const int64_t world_size = tp_world_size_;
  out_size_local_ = out_features / std::max<int64_t>(1, world_size);
  inter_size_local_ = is_fused_gate_up_ ? out_size_local_ / 2 : out_size_local_;
}

torch::Tensor LoRAColumnParallelLinearImpl::forward(torch::Tensor input) {
  auto y = base_->forward(input);

  const auto* ctx = current_lora_context();
  if (ctx == nullptr || ctx->adapter_ids == nullptr ||
      ctx->q_seq_lens_vec == nullptr || ctx->layer_index < 0) {
    return y;
  }
  const auto& adapter_ids = *ctx->adapter_ids;
  const auto& q_seq_lens = *ctx->q_seq_lens_vec;
  if (adapter_ids.empty() || adapter_ids.size() != q_seq_lens.size()) {
    return y;
  }
  bool any_nonzero = false;
  for (auto id : adapter_ids) {
    if (id != 0) {
      any_nonzero = true;
      break;
    }
  }
  if (!any_nonzero) return y;

  // P1a Phase 0 fast path: single-adapter batch (99% prod). Skip per-seq
  // slice/matmul loop; do 2 matmuls per proj on full [T, hidden].
  {
    uint64_t sole_aid = 0;
    bool single_adapter = true;
    for (auto id : adapter_ids) {
      if (id == 0) {
        single_adapter = false;  // base-only seq disqualifies fast path
        break;
      }
      if (sole_aid == 0) {
        sole_aid = id;
      } else if (id != sole_aid) {
        single_adapter = false;
        break;
      }
    }
    if (single_adapter && sole_aid != 0) {
      auto& runtime = LoRARuntime::instance();
      if (is_fused_gate_up_) {
        const std::optional<LoRARuntime::ProjDelta> gate_pd =
            runtime.get_per_proj_delta(sole_aid, ctx->layer_index, "gate_proj");
        const std::optional<LoRARuntime::ProjDelta> up_pd =
            runtime.get_per_proj_delta(sole_aid, ctx->layer_index, "up_proj");
        if (!gate_pd.has_value() && !up_pd.has_value()) return y;
        auto shrink_expand =
            [&](const LoRARuntime::ProjDelta* pd) -> torch::Tensor {
          if (pd == nullptr) {
            return torch::zeros({input.size(0), inter_size_local_},
                                input.options());
          }
          auto tmp = torch::matmul(input, pd->A.transpose(0, 1));
          torch::Tensor B_local = pd->B;
          if (tp_world_size_ > 1 && pd->B.size(0) > inter_size_local_) {
            const int64_t start = tp_rank_ * inter_size_local_;
            B_local = pd->B.slice(0, start, start + inter_size_local_);
          }
          auto d = torch::matmul(tmp, B_local.transpose(0, 1));
          return (d * pd->scaling).to(input.dtype());
        };
        auto gate_delta =
            shrink_expand(gate_pd.has_value() ? &*gate_pd : nullptr);
        auto up_delta = shrink_expand(up_pd.has_value() ? &*up_pd : nullptr);
        auto fused_delta = torch::cat({gate_delta, up_delta}, /*dim=*/-1);
        y = y + fused_delta;
        return y;
      } else {
        // Single-proj branch (reserved). Not exercised in Qwen family
        // currently, but mirror the fast path for symmetry.
        const std::optional<LoRARuntime::ProjDelta> pd =
            runtime.get_per_proj_delta(sole_aid, ctx->layer_index, proj_name_);
        if (!pd.has_value()) return y;
        auto tmp = torch::matmul(input, pd->A.transpose(0, 1));
        torch::Tensor B_local = pd->B;
        if (tp_world_size_ > 1 && pd->B.size(0) > out_size_local_) {
          const int64_t start = tp_rank_ * out_size_local_;
          B_local = pd->B.slice(0, start, start + out_size_local_);
        }
        auto delta = torch::matmul(tmp, B_local.transpose(0, 1));
        delta = (delta * pd->scaling).to(y.dtype());
        y = y + delta;
        return y;
      }
    }
  }

#if defined(USE_NPU)
  // AscendC LoRA slow_path (multi-rank multi-adapter) -- optional kernel
  // acceleration for the fused gate_up column-parallel MoE case. Buckets
  // tokens by rank and launches gate + up bgmv_shrink/expand per bucket,
  // scattering into y's two inter-size halves via index_add.
  //
  // Envelope pre-flight: the kernel only serves rank <= 64 (the
  // vllm-ascend bgmv_expand ceiling). If any active bucket is unsupported,
  // control breaks out so R > 64 adapters fall through to the rank-agnostic
  // ATen per-seq path instead of silently returning zero delta.
  do {
    if (std::getenv("DISABLE_ASCENDC_SPRINT_GAMMA") != nullptr) break;
    {
      static bool first_attempt_col = true;
      if (first_attempt_col) {
        first_attempt_col = false;
        LOG(INFO) << "[AscendC Column multishape] first attempt";
      }
    }
    if (std::getenv("USE_ASCENDC_LORA") == nullptr) break;
    if (adapter_ids.size() <= 1) break;
    if (!input.device().is_privateuseone()) break;
    if (!is_fused_gate_up_) break;

    auto& storage = AscendCLoRAStorage::instance();
    auto gate_b = storage.get_bucketed(ctx->layer_index, "gate_proj");
    auto up_b = storage.get_bucketed(ctx->layer_index, "up_proj");
    if (gate_b.num_buckets() == 0 && up_b.num_buckets() == 0) break;
    if (!storage.slot_lookup_dev().defined() ||
        !storage.bucket_lookup_dev().defined()) {
      break;
    }
    if (ctx->adapter_ids_per_token == nullptr ||
        !ctx->adapter_ids_per_token->defined()) {
      break;
    }
    if (ctx->adapter_ids_per_token->numel() != input.size(0)) break;

    // Reject the whole kernel path before modifying y if any bucket is
    // unsupported, including mixed small/large-rank adapter batches.
    bool has_valid_bucket = false;
    auto all_buckets_in_envelope =
        [&](const AscendCLoRAStorage::BucketedStackedView& bv) {
          for (int b = 0; b < bv.num_buckets(); ++b) {
            const auto& view = bv.buckets[b];
            if (!view.valid) continue;
            has_valid_bucket = true;
            if (view.A_stacked.size(1) > 64 ||
                view.A_stacked.size(2) != input.size(1)) {
              return false;
            }
          }
          return true;
        };
    if (!all_buckets_in_envelope(gate_b) || !all_buckets_in_envelope(up_b) ||
        !has_valid_bucket) {
      LOG_FIRST_N(WARNING, 20)
          << "[AscendC LoRA slow_path] unsupported bucket at layer="
          << ctx->layer_index
          << " (gate_up); falling back to per-seq ATen apply (kernel rank "
             "ceiling 64).";
      break;
    }

    auto per_tok_slot =
        storage.slot_lookup_dev().index_select(0, *ctx->adapter_ids_per_token);
    auto per_tok_bucket = storage.bucket_lookup_dev().index_select(
        0, *ctx->adapter_ids_per_token);
    // The per-token slot vector stays on the device; instead of syncing it
    // back, reject the whole kernel path when any non-base adapter in this
    // batch has no AscendC storage slot. Such an adapter would otherwise look
    // like a base token and silently lose its delta.
    bool all_adapters_registered = true;
    for (auto id : adapter_ids) {
      if (id != 0 && !storage.has_adapter(id)) {
        all_adapters_registered = false;
        break;
      }
    }
    if (!all_adapters_registered) break;

    LOG_EVERY_N(INFO, 100) << "[AscendC Column multishape] enter aids="
                           << adapter_ids.size()
                           << " layer=" << ctx->layer_index
                           << " gate_buckets=" << gate_b.num_buckets()
                           << " up_buckets=" << up_b.num_buckets();

    auto shard_B_col = [&](const torch::Tensor& B_full) {
      if (tp_world_size_ <= 1 || B_full.size(1) <= inter_size_local_)
        return B_full;
      const int64_t num_shards = B_full.size(1) / inter_size_local_;
      if (num_shards <= 0) return B_full;
      const int64_t shard_idx = tp_rank_ * num_shards / tp_world_size_;
      const int64_t start = shard_idx * inter_size_local_;
      return B_full.slice(1, start, start + inter_size_local_);
    };

    const auto buf_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device());

    // Iterate buckets present in EITHER gate or up (same adapter registers
    // both under a shared bucket_id). Every valid bucket passed the
    // envelope pre-flight, so no adapter is omitted by the kernel path.
    const int max_buckets = std::max(gate_b.num_buckets(), up_b.num_buckets());
    for (int b = 0; b < max_buckets; ++b) {
      bool gate_valid = (b < gate_b.num_buckets()) && gate_b.buckets[b].valid;
      bool up_valid = (b < up_b.num_buckets()) && up_b.buckets[b].valid;
      if (!gate_valid && !up_valid) continue;

      const int64_t R = gate_valid ? gate_b.buckets[b].A_stacked.size(1)
                                   : up_b.buckets[b].A_stacked.size(1);
      if (R > 64) continue;  // kernel rank ceiling
      if (gate_valid && gate_b.buckets[b].A_stacked.size(2) != input.size(1))
        continue;
      if (up_valid && up_b.buckets[b].A_stacked.size(2) != input.size(1))
        continue;

      auto mask = (per_tok_bucket == static_cast<int64_t>(b));
      auto sub_token_ids = torch::nonzero(mask).squeeze(-1);
      if (sub_token_ids.numel() == 0) continue;

      auto sub_input = input.index_select(0, sub_token_ids);
      auto sub_slot = per_tok_slot.index_select(0, sub_token_ids);
      const int64_t sub_B = sub_input.size(0);

      if (gate_valid) {
        const auto& gate_view = gate_b.buckets[b];
        auto sub_buf = torch::zeros({sub_B, R}, buf_options);
        auto gateB = shard_B_col(gate_view.B_stacked);
        auto sub_y = torch::zeros({sub_B, inter_size_local_}, input.options());
        xllm::bgmv_shrink(sub_input,
                          gate_view.A_stacked,
                          sub_slot,
                          sub_buf,
                          static_cast<double>(gate_view.scaling));
        xllm::bgmv_expand(
            sub_buf, gateB, sub_slot, sub_y, 0, inter_size_local_);
        y.slice(1, 0, inter_size_local_).index_add_(0, sub_token_ids, sub_y);
      }
      if (up_valid) {
        const auto& up_view = up_b.buckets[b];
        auto sub_buf = torch::zeros({sub_B, R}, buf_options);
        auto upB = shard_B_col(up_view.B_stacked);
        auto sub_y = torch::zeros({sub_B, inter_size_local_}, input.options());
        xllm::bgmv_shrink(sub_input,
                          up_view.A_stacked,
                          sub_slot,
                          sub_buf,
                          static_cast<double>(up_view.scaling));
        xllm::bgmv_expand(sub_buf, upB, sub_slot, sub_y, 0, inter_size_local_);
        y.slice(1, inter_size_local_, 2 * inter_size_local_)
            .index_add_(0, sub_token_ids, sub_y);
      }
    }
    return y;
  } while (false);
#endif  // USE_NPU

  auto& runtime = LoRARuntime::instance();
  int64_t tok_off = 0;

  for (size_t seq_idx = 0; seq_idx < adapter_ids.size(); ++seq_idx) {
    const int32_t seq_len = q_seq_lens[seq_idx];
    if (seq_len <= 0) continue;
    const uint64_t aid = adapter_ids[seq_idx];
    if (aid == 0) {
      tok_off += seq_len;
      continue;
    }

    auto x_seq = input.slice(0, tok_off, tok_off + seq_len);

    if (is_fused_gate_up_) {
      // Fused gate_up: lookup gate_proj + up_proj deltas, concat.
      const std::optional<LoRARuntime::ProjDelta> gate_pd =
          runtime.get_per_proj_delta(aid, ctx->layer_index, "gate_proj");
      const std::optional<LoRARuntime::ProjDelta> up_pd =
          runtime.get_per_proj_delta(aid, ctx->layer_index, "up_proj");
      if (!gate_pd.has_value() && !up_pd.has_value()) {
        tok_off += seq_len;
        continue;
      }
      auto make_delta = [&](const LoRARuntime::ProjDelta* pd) -> torch::Tensor {
        if (pd == nullptr) {
          return torch::zeros({x_seq.size(0), inter_size_local_},
                              x_seq.options());
        }
        auto tmp = torch::matmul(x_seq, pd->A.transpose(0, 1));
        // TP shard: adapter's B is [inter_full, rank]. This rank owns
        // rows [tp_rank * inter_size_local_, +inter_size_local_).
        torch::Tensor B_local = pd->B;
        if (tp_world_size_ > 1 && pd->B.size(0) > inter_size_local_) {
          const int64_t start = tp_rank_ * inter_size_local_;
          B_local = pd->B.slice(0, start, start + inter_size_local_);
        }
        auto d = torch::matmul(tmp, B_local.transpose(0, 1));
        return (d * pd->scaling).to(x_seq.dtype());
      };
      auto gate_delta = make_delta(gate_pd.has_value() ? &*gate_pd : nullptr);
      auto up_delta = make_delta(up_pd.has_value() ? &*up_pd : nullptr);
      auto fused_delta = torch::cat({gate_delta, up_delta}, /*dim=*/-1);
      auto y_view = y.slice(0, tok_off, tok_off + seq_len);
      y_view.copy_(y_view + fused_delta);
    } else {
      // Single proj (future use — reserved). Not applicable to Qwen for now.
      const std::optional<LoRARuntime::ProjDelta> pd =
          runtime.get_per_proj_delta(aid, ctx->layer_index, proj_name_);
      if (!pd.has_value()) {
        tok_off += seq_len;
        continue;
      }
      auto tmp = torch::matmul(x_seq, pd->A.transpose(0, 1));
      // TP shard: adapter's B is [out_full, rank]. This rank owns
      // rows [tp_rank * out_size_local_, +out_size_local_).
      torch::Tensor B_local = pd->B;
      if (tp_world_size_ > 1 && pd->B.size(0) > out_size_local_) {
        const int64_t start = tp_rank_ * out_size_local_;
        B_local = pd->B.slice(0, start, start + out_size_local_);
      }
      auto delta = torch::matmul(tmp, B_local.transpose(0, 1));
      delta = (delta * pd->scaling).to(y.dtype());
      auto y_view = y.slice(0, tok_off, tok_off + seq_len);
      y_view.copy_(y_view + delta);
    }
    tok_off += seq_len;
  }
  return y;
}

void LoRAColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict) {
  base_->load_state_dict(state_dict);
}

void LoRAColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  base_->load_state_dict(state_dict, prefixes);
}

void LoRAColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    int32_t shard_tensor_count,
    const std::vector<int64_t>& shard_sizes) {
  base_->load_state_dict(state_dict, shard_tensor_count, shard_sizes);
}

bool LoRAColumnParallelLinearImpl::uses_w8a8_dynamic_quant() const {
  // The w8a8 quantized-allgather fast path in DenseMLPImpl bypasses
  // forward(), so it must not be taken while a LoRA delta could apply to
  // the current batch - the delta would be silently dropped. Report the
  // base's answer only when no adapter participates.
  const auto* ctx = current_lora_context();
  if (ctx != nullptr && ctx->adapter_ids != nullptr) {
    for (auto id : *ctx->adapter_ids) {
      if (id != 0) {
        return false;
      }
    }
  }
  return base_->uses_w8a8_dynamic_quant();
}

}  // namespace layer
}  // namespace xllm
