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

#include "models/cp_capability.h"

#include <unordered_set>

#include "core/common/instance_role_util.h"
#include "core/common/options.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/platform/mlu/mlu_model_cp.h"
#include "core/platform/platform.h"
#include "core/util/utils.h"
#include "models/model_registry.h"

namespace xllm {

std::optional<std::string> validate_model_cp(const Options& options,
                                             EngineType engine_type,
                                             const std::string& model_type,
                                             int32_t global_world_size) {
  if (options.cp_size() < 1) {
    return "cp_size must be greater than or equal to 1";
  }

  if (options.cp_size() == 1) {
    return std::nullopt;
  }
  if (options.dp_size() < 1) {
    return "CP requires dp_size >= 1";
  }

  if (Platform::is_mlu()) {
    const bool is_vlm =
        engine_type == EngineType::VLM || engine_type == EngineType::VLMSSM;
    if (is_vlm && !mlu::supports_vlm_cp(model_type)) {
      return "MLU VLM CP does not support model_type=" + model_type +
             " with engine_type=" + engine_type.to_string();
    }
    if (!is_vlm && engine_type != EngineType::LLM &&
        engine_type != EngineType::SSM) {
      return "MLU CP supports only LLM or supported VLM generation";
    }
    if (options.task_type() != "generate") {
      return "MLU CP supports only the generate task";
    }
    if (!is_mlu_model_cp_capable(model_type)) {
      return "MLU CP does not support model_type=" + model_type;
    }

    if (options.dp_size() != 1) {
      return "MLU CP requires dp_size == 1";
    }

    if (global_world_size % (options.dp_size() * options.cp_size()) != 0) {
      return "MLU CP requires world_size divisible by dp_size * cp_size "
             "(orthogonal PCP x TP layout)";
    }

    const int32_t kv_split_size =
        ParallelConfig::get_instance().kv_split_size_effective();
    if (kv_split_size > 1 && model_type != "glm_moe_dsa") {
      return "MLU CP with kv_split_size > 1 supports only glm_moe_dsa";
    }
    const int32_t tp_size =
        global_world_size / (options.dp_size() * options.cp_size());
    if (!kv_split_fits_cp_topology(options.cp_size(), tp_size, kv_split_size) &&
        !kv_split_spans_cp_tp(options.cp_size(), tp_size, kv_split_size)) {
      return "MLU CP requires kv_split_size to divide cp_size or equal "
             "cp_size * tp_size";
    }

    if (options.ep_size() != global_world_size) {
      return "MLU CP requires ep_size == global world size";
    }
    return std::nullopt;
  }

  if (Platform::is_npu()) {
    const bool is_python_qwen3_vl =
        model_type == "qwen3_vl" &&
        ModelConfig::is_python_model_impl(
            ModelConfig::get_instance().model_impl());
    if (is_python_qwen3_vl) {
      if (engine_type != EngineType::VLM) {
        return "Python Qwen3-VL CP supports only non-speculative VLM "
               "generation";
      }
      // Dense CP gathers the current prefill only. Cached prefixes and media
      // spanning chunks need a separate absolute-position/KV contract.
      if (options.enable_chunked_prefill() ||
          SchedulerConfig::get_instance().enable_chunked_prefill()) {
        return "Python Qwen3-VL CP requires enable_chunked_prefill=false";
      }
      if (options.priority_strategy() == "multi_slo_and_prio") {
        return "Python Qwen3-VL CP does not support "
               "priority_strategy=multi_slo_and_prio (requires chunked "
               "prefill)";
      }
      if (options.enable_prefix_cache() ||
          KVCacheConfig::get_instance().enable_prefix_cache()) {
        return "Python Qwen3-VL CP requires enable_prefix_cache=false";
      }
      if (options.enable_schedule_overlap() ||
          SchedulerConfig::get_instance().enable_schedule_overlap()) {
        return "Python Qwen3-VL CP requires enable_schedule_overlap=false";
      }
      if (options.enable_graph() || options.enable_prefill_piecewise_graph()) {
        return "Python Qwen3-VL CP requires eager execution";
      }
      if (options.enable_disagg_pd() || options.enable_pd_ooc() ||
          options.instance_role() != InstanceRole::DEFAULT) {
        return "Python Qwen3-VL CP requires standalone DEFAULT execution";
      }
      if (options.num_speculative_tokens() != 0) {
        return "Python Qwen3-VL CP does not support speculative decoding";
      }
      if (ParallelConfig::get_instance().kv_split_size_effective() != 1) {
        return "Python Qwen3-VL CP requires kv_split_size == 1";
      }
    }
    if (engine_type != EngineType::LLM && engine_type != EngineType::SSM &&
        !(is_python_qwen3_vl && engine_type == EngineType::VLM)) {
      return "Model-side CP supports only LLM text generation";
    }
    if (options.task_type() != "generate") {
      return "Model-side CP supports only the generate task";
    }
    const bool is_dsv4_model = util::is_deepseek_v4_model_type(model_type);
    // glm5_next is exempted alongside deepseek_v4 only for DFlash2
    // specifically: DFlash2's aux-hidden-buffer restore under CP is proven
    // correct (glm5_next.py's cp_merge_rows on aux_hidden_buffer, mirroring
    // DeepseekV4CpContext.gather_restore) and unit-tested
    // (test_glm5_next_cp.py's
    // test_layer_capture_restores_global_rows_with_context_parallelism).
    // glm_moe_dsa is exempted for DFlash2 by the same mechanism, python
    // model impl only (the native/ATB glm_moe_dsa has no aux capture):
    // glm5_2.py merges the target output and the aux-hidden buffer with
    // one cp_merge_rows permutation (unit-tested in
    // test_glm5_2_cp_ragged.py's
    // test_cp_aux_hidden_is_merged_with_target_in_capture_order).
    // Eagle3/DFlash(1)/DSpark are not exempted here: their model-specific
    // dispatch paths were not individually verified (design.md §3.1 Gate B1).
    const bool is_glm5_next_dflash2 = SpeculativeConfig::is_glm5_next_dflash2(
        model_type, options.speculative_algorithm());
    const bool is_glm_moe_dsa_dflash2 =
        SpeculativeConfig::is_glm_moe_dsa_dflash2(
            model_type,
            options.speculative_algorithm(),
            ModelConfig::get_instance().model_impl());
    if (engine_type == EngineType::SSM &&
        SpeculativeConfig::requires_aux_hidden_capture(
            options.speculative_algorithm()) &&
        !is_dsv4_model && !is_glm5_next_dflash2 && !is_glm_moe_dsa_dflash2) {
      return "Current model-side CP does not support aux-hidden-capture "
             "speculative algorithms (Eagle3/DFlash/DSpark); run speculative "
             "decoding on a cp_size=1 Decode instance.";
    }
    // Native model-side CP is compatible with graph mode because the two are
    // phase-disjoint: CP only engages on batch_forward_type.no_decode(), while
    // ACL graph captures/replays pure decode. The Python CP path uses the
    // same phase split with decode-only ACLGraph.
    //
    // The one batch that satisfies both gates is spec-verify chunked prefill.
    // The graph executor only takes it for hybrid-linear-attention models, and
    // no CP-capable model is one, so it falls back to eager today; the guard in
    // AclGraphExecutorImpl::run() keeps that true if a future model is both.
    if (options.instance_role() != InstanceRole::DEFAULT &&
        options.instance_role() != InstanceRole::PREFILL) {
      return "Model-side CP supports only DEFAULT or PREFILL roles";
    }

    // Python model executor runs a standalone torch CP path (all-gather KV,
    // eager prefill only) that does not go through the ATB fused-attention op,
    // so it bypasses the ATB-backend requirement and the ATB CP capability
    // allowlist below. The safety constraints above (LLM/generate,
    // DEFAULT/PREFILL) still apply. Orthogonal TP x CP is supported (both may
    // be > 1, sharing world = cp * tp); the collective communicator builds the
    // narrowed TP group and the strided CP group as separate torch subgroups
    // off the shared world rendezvous endpoint. GLM-5 Next also restores
    // DP-local CP rows before routed MoE, supporting the dp * cp * tp layout.
    // Other Python CP models remain restricted to DP1.
    if (ModelConfig::is_python_model_impl(
            ModelConfig::get_instance().model_impl())) {
      // Only models whose Python forward actually shards the sequence (via
      // cp_shard_rows / cp_merge_rows) may enable CP. Other Python models keep
      // a full-sequence forward, so a cp_context would be built but never
      // consumed: qwen3_5 would reach _prefill_cp with unsharded rows (garbled
      // or out-of-bounds output) and unsupported MLA models would silently
      // recompute the whole sequence on every rank. Mirror the implementations
      // that explicitly consume cp_context rather than admitting every Python
      // model type.
      static const std::unordered_set<std::string> kPythonCpCapableModels = {
          "qwen3",
          "qwen3_vl",
          "glm_moe_dsa",
          "glm5_next",
      };
      if (kPythonCpCapableModels.find(model_type) ==
          kPythonCpCapableModels.end()) {
        return "Python model-side CP does not support model_type=" +
               model_type +
               "; supported models are qwen3, qwen3_vl, glm_moe_dsa, and "
               "glm5_next.";
      }
      // Consume the shared glm5_next+DFlash2 pairing predicate (like the
      // aux-capture gate above) so the algorithm whitelist cannot drift
      // when further algorithms are cleared. The explicit model_type
      // qualifier stays: this gate scopes a glm5_next-specific refusal,
      // and dropping the qualifier would newly refuse qwen3/glm_moe_dsa
      // SSM shapes that pass today with a glm5_next-specific message.
      if (model_type == "glm5_next" && engine_type == EngineType::SSM &&
          !is_glm5_next_dflash2) {
        return "Python GLM-5 Next CP does not support this target-side "
               "speculative algorithm; run speculation on a cp_size=1 "
               "Decode instance";
      }
      if (model_type == "glm_moe_dsa" && engine_type == EngineType::SSM &&
          SpeculativeConfig::is_mtp_algorithm(
              options.speculative_algorithm())) {
        return "Python model-side CP does not support MTP speculative "
               "verification; run MTP on a cp_size=1 Decode instance";
      }
      if (options.dp_size() != 1 && model_type != "glm5_next") {
        return "Python CP requires dp_size == 1";
      }
      const uint64_t parallel_width = static_cast<uint64_t>(options.dp_size()) *
                                      static_cast<uint64_t>(options.cp_size());
      if (global_world_size <= 0 ||
          static_cast<uint64_t>(global_world_size) % parallel_width != 0) {
        return "Python CP requires world_size divisible by dp_size * cp_size";
      }
      const int32_t attn_tp_size = static_cast<int32_t>(
          static_cast<uint64_t>(global_world_size) / parallel_width);
      const int32_t kv_split =
          ParallelConfig::get_instance().kv_split_size_effective();
      if (model_type == "glm5_next") {
        if (options.ep_size() == 0 ||
            global_world_size % options.ep_size() != 0) {
          return "Python GLM-5 Next CP requires ep_size to be a positive "
                 "divisor of world_size";
        }
        // EPLv1 may use any positive divisor. EPLv2 owns one expert shard
        // per rank, so PCP still requires ep_size == world_size.
        if (options.expert_parallel_degree().value_or(0) == 2 &&
            options.ep_size() != global_world_size) {
          return "Python GLM-5 Next EPLv2 with PCP requires ep_size "
                 "equal to world_size";
        }
        // Owner-sharded KV takes the same constraint glm_moe_dsa uses below,
        // because the constraint is a property of the sharded-KV mechanism
        // rather than of that model: PyExecutorImpl builds
        // KVShardBatchMetadata only for prefill/chunked-prefill batches, so
        // the owner-local write plus the CP gather that reconstructs the
        // logical cache -- the only path that has been validated -- exists on
        // the prefill side alone. On decode the shards would have to be
        // combined by the dcp group through SfaDcpAttentionBackend, which is
        // a different mechanism this gate does not enable. A prefill-only
        // instance is the exact condition, not "role == PREFILL": with
        // enable_disagg_pd=false the continuous scheduler runs prefill and
        // decode on the same instance whatever the role says
        // (is_pd_prefill_role spells out that condition).
        if (kv_split > 1 && !is_pd_prefill_role(options)) {
          return "Python GLM-5 Next CP with kv_split_size > 1 requires "
                 "disaggregated PD with the PREFILL role; set "
                 "enable_disagg_pd=true and instance_role=PREFILL";
        }
        // No mix-batch gate: resolve_batch_mode() force-disables mixed
        // batches whenever cp_size > 1 (and again for every PD prefill
        // instance), so the raw enable_mix_batch flag -- default true -- is
        // inert for every shape this block validates. Gating on it rejected
        // default-flag CP launches over behavior the scheduler never runs.
        // Prefix caching is admitted on the same prefill-only PD shape the
        // kv_split gate above admits. A prefix-cache hit reaches the model
        // exactly like a chunked prefill's cached suffix: a batch whose
        // kv_seq_lens exceed q_seq_lens over blocks the block manager
        // already allocated, which is the prefix-extended CP forward the
        // chunked-prefill admission proved (build_cp_context's
        // kv_gather_index covers each segment's causal prefix). The
        // prefill-only scope keeps the DEFAULT role's decode-on-CP ranks
        // outside the admission: decode there reads matched blocks with no
        // CP plan, and that combination stays untested. This admission is
        // also what makes host offload configurable for glm5_next CP:
        // validate_host_cache_options requires prefix caching, so without
        // this shape a CP prefill could never set --host_blocks_factor > 1.
        if (KVCacheConfig::get_instance().enable_prefix_cache() &&
            !is_pd_prefill_role(options)) {
          return "Python GLM-5 Next CP with enable_prefix_cache=true "
                 "requires disaggregated PD with the PREFILL role; a "
                 "standalone CP instance must set enable_prefix_cache=false";
        }
        // Schedule overlap is admitted for glm5_next CP: the overlapped
        // decode token replacement is byte-identical across CP ranks
        // (greedy argmax over CP-replicated logits, plus the CP sample-token
        // broadcast for stochastic batches in LLMWorkerImpl), the deferred
        // KDA linear-state restore stays ordered on the compute stream, and
        // the single-threaded worker task pool keeps the block lifecycle
        // FIFO. The M11.5 overlap-invariant tests are the authority.
        if (options.enable_disagg_pd() &&
            options.instance_role() != InstanceRole::PREFILL) {
          return "Python GLM-5 Next CP with disaggregated PD requires the "
                 "PREFILL role";
        }
        if (options.enable_pd_ooc()) {
          return "Python GLM-5 Next CP initially requires enable_pd_ooc=false";
        }
      }
      if (!kv_split_fits_cp_topology(
              options.cp_size(), attn_tp_size, kv_split)) {
        return "Python CP requires kv_split_size effective value to be a "
               "positive divisor of cp_size";
      }
      if (model_type == "glm_moe_dsa" && kv_split > 1 &&
          !is_pd_prefill_role(options)) {
        return "Python GLM CP with kv_split_size > 1 requires disaggregated "
               "PD with the PREFILL role; set enable_disagg_pd=true and "
               "instance_role=PREFILL";
      }
      return std::nullopt;
    }

    // Require registered NPU model-side CP capability. The backend is not
    // constrained: ATB models drive CP through NpuCpPlan, while TORCH models
    // (deepseek_v4) own their CP split inside the model. Both rely on the
    // orthogonal dp * cp * attn_tp == world layout validated below.
    std::string effective_backend;
    std::string resolved_name;
    std::string resolve_error;
    // Runtime platform branches are type-checked in every platform build.
    const std::string requested_backend = options.npu_kernel_backend();
    if (!resolve_model_registration(model_type,
                                    requested_backend,
                                    &effective_backend,
                                    &resolved_name,
                                    &resolve_error)) {
      return "Model-side CP rejected model_type=" + model_type + ": " +
             resolve_error;
    }
    if (!is_npu_model_cp_capable(resolved_name)) {
      return "NPU model-side CP does not support model_type=" + model_type +
             " (resolved=" + resolved_name +
             "); only deepseek_v32, deepseek_v32_mtp, deepseek_v4, "
             "deepseek_v4_mtp, glm_moe_dsa, glm_moe_dsa_mtp are registered as "
             "CP-capable.";
    }
    if (global_world_size % (options.dp_size() * options.cp_size()) != 0) {
      return "NPU CP requires world_size divisible by dp_size * cp_size "
             "(orthogonal CP x TP layout)";
    }
    const int32_t attn_tp_size =
        global_world_size / (options.dp_size() * options.cp_size());
    if (attn_tp_size < 1) {
      return "NPU CP requires attn_tp_size >= 1";
    }
    const int32_t kv_split =
        ParallelConfig::get_instance().kv_split_size_effective();
    if (!kv_split_fits_cp_topology(options.cp_size(), attn_tp_size, kv_split)) {
      return "NPU CP requires kv_split_size effective value to be a positive "
             "divisor of cp_size";
    }
    return std::nullopt;
  }

  if (Platform::is_musa()) {
    if (engine_type != EngineType::LLM && engine_type != EngineType::SSM) {
      return "MUSA CP supports only LLM/SSM engine generation";
    }
    if (options.task_type() != "generate") {
      return "MUSA CP supports only the generate task";
    }
    if (!is_musa_model_cp_capable(model_type)) {
      return "MUSA CP supports only Qwen3.5-family targets, got model_type=" +
             model_type;
    }
    if (options.cp_size() != 2 || options.dp_size() < 1 ||
        global_world_size != options.dp_size() * options.cp_size()) {
      return "MUSA CP requires cp_size == 2 and world_size == dp_size * "
             "cp_size (no tensor parallelism)";
    }
    if (ParallelConfig::get_instance().kv_split_size_effective() != 1) {
      return "MUSA CP keeps the full KV cache on every CP rank; set "
             "kv_split_size=1";
    }
    if (options.instance_role() != InstanceRole::DEFAULT &&
        options.instance_role() != InstanceRole::PREFILL) {
      return "MUSA CP supports only DEFAULT or PREFILL roles";
    }
    // PUSH is not yet adapted for MUSA context parallelism: the complete KV
    // cache lives only on the last CP rank, so a per-rank PUSH would ship the
    // earlier ranks' incomplete halves. PULL serves from the last-rank owner
    // (see LLMEngine::get_cache_info) and is supported; require it until PUSH
    // is adapted.
    if (options.enable_disagg_pd() &&
        options.kv_cache_transfer_mode() == "PUSH") {
      return "MUSA CP disaggregated PD supports only "
             "kv_cache_transfer_mode=PULL; PUSH is not yet adapted for MUSA "
             "context parallelism (the complete KV cache lives only on the "
             "last CP rank)";
    }
    return std::nullopt;
  }

  return "cp_size > 1 is only supported on platforms with model-side CP "
         "(MLU/NPU/MUSA); disable CP (cp_size=1) or use MLU/NPU/MUSA.";
}

}  // namespace xllm
