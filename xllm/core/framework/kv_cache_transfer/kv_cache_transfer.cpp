/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/kv_cache_transfer/kv_cache_transfer.h"

#include <folly/ScopeGuard.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include "absl/strings/ascii.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/request/sequence.h"
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU) || \
    defined(USE_MUSA)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif
#include "util/env_var.h"

namespace xllm {

namespace {

std::vector<KVTransferTaskResult> failed_tasks_for_requests(
    const std::vector<TransferKVInfo>& transfer_kv_infos) {
  std::vector<std::string> request_ids =
      unique_transfer_request_ids(transfer_kv_infos);
  if (request_ids.empty()) {
    return {};
  }
  return {KVTransferTaskResult{std::move(request_ids),
                               KVTransferErrorCode::FAILED}};
}

// Mirrors cp_utils.py: only an explicit, recognized value selects a mode, and
// only "sharded" differs from the default. An empty value is deliberately not
// a mode (an unset shell variable), so it keeps the replicated default.
constexpr char kCpIndexWriteModeEnv[] = "XLLM_CP_INDEX_WRITE_MODE";

}  // namespace

int32_t declared_cp_index_write_mode() {
  // Mirror the python reader (cp_utils.py: strip, then lowercase, then exact
  // match): absl::StripAsciiWhitespace strips the full ASCII whitespace set
  // (\t\n\v\f\r and space) just like python's str.strip(), so a value such
  // as "  SHARDED\v" resolves identically on both sides.
  const std::string lowered = absl::AsciiStrToLower(
      util::get_optional_string_env(kCpIndexWriteModeEnv).value_or(""));
  const absl::string_view trimmed = absl::StripAsciiWhitespace(lowered);
  return trimmed == "sharded" ? kCpIndexWriteModeSharded
                              : kCpIndexWriteModeReplicated;
}

const char* cp_index_write_mode_label(int32_t cp_index_write_mode) {
  switch (cp_index_write_mode) {
    case kCpIndexWriteModeReplicated:
      return "replicated";
    case kCpIndexWriteModeSharded:
      return "sharded";
    default:
      return "unspecified";
  }
}

KvSplitWidthPlan plan_kv_split_widths(int32_t src_kv_split_size,
                                      int32_t dst_kv_split_size,
                                      int32_t src_cp_index_write_mode,
                                      int32_t dst_cp_index_write_mode,
                                      std::string* reason) {
  const auto reject = [reason](const std::string& message) {
    if (reason != nullptr) {
      *reason = message;
    }
    return KvSplitWidthPlan::UNRECONCILABLE;
  };
  if (src_kv_split_size < 1) {
    // 0 is proto3's "field not present" and means the source never declared
    // a width -- nothing can be validated about how its blocks are laid out.
    return reject(
        "KV split width not declared, source=" +
        std::to_string(src_kv_split_size) +
        ", destination=" + std::to_string(dst_kv_split_size) +
        ": the source must declare kv_split_size before a PUSH can be mapped "
        "(0 means the peer's registration record, or the service that served "
        "it, carried no width)");
  }
  if (dst_kv_split_size < 1) {
    // An undeclared DESTINATION width only matters when the source actually
    // splits KV: a split source's stride remapping depends on the
    // destination's declared width, and refusing is what once kept a decode
    // intending dcp=4 (behind a service that never carried the field) from
    // silently taking the 1:1 mapping path. A width-1 source never splits,
    // skips filter_kv_split_infos entirely, and maps onto the destination's
    // table one-to-one whatever the destination declares or fails to
    // declare -- refusing it would fail every request of a legacy pair
    // (non-CP prefill, old-service-registered decode) that transferred fine
    // with stride=1 before the field existed.
    if (src_kv_split_size == 1) {
      return KvSplitWidthPlan::RANK_LOCAL;
    }
    return reject(
        "KV split width not declared, source=" +
        std::to_string(src_kv_split_size) +
        ", destination=" + std::to_string(dst_kv_split_size) +
        ": a split source requires the destination to declare kv_split_size "
        "before a PUSH can be mapped (0 means the peer's registration record, "
        "or the service that served it, carried no width)");
  }
  if (src_kv_split_size == dst_kv_split_size) {
    // Sharded INDEX writes keep only page 0 of each logical block's
    // index-page group valid on any rank, so the 1:1 page moves of a
    // RANK_LOCAL plan would deliver stale peer pages. Only the replicated
    // mode (the XLLM_CP_INDEX_WRITE_MODE default) makes every page of every
    // resource transferable; an unspecified mode is a legacy instance and
    // keeps the pre-field behavior. The check is symmetric: a sharded
    // declaration on EITHER side of an equal-width kv_split_size > 1 pair
    // poisons the 1:1 moves, because the sharded side's non-zero pages are
    // stale no matter which endpoint wrote them.
    const bool src_sharded =
        src_cp_index_write_mode == kCpIndexWriteModeSharded;
    const bool dst_sharded =
        dst_cp_index_write_mode == kCpIndexWriteModeSharded;
    if (src_kv_split_size > 1 && (src_sharded || dst_sharded)) {
      const char* sharded_side =
          src_sharded && dst_sharded
              ? "both sides"
              : (src_sharded ? "the source" : "the destination");
      return reject(
          "CP index write mode mismatch, source=" +
          std::to_string(src_kv_split_size) + ", destination=" +
          std::to_string(dst_kv_split_size) + ", source write mode=" +
          cp_index_write_mode_label(src_cp_index_write_mode) +
          ", destination write mode=" +
          cp_index_write_mode_label(dst_cp_index_write_mode) + ": " +
          std::string(sharded_side) +
          " declare(s) the sharded index write mode "
          "(XLLM_CP_INDEX_WRITE_MODE=sharded), whose pages stay valid only "
          "for a kv_split_size == 1 destination -- pair an equal-width "
          "kv_split destination only with the replicated write mode on both "
          "sides (the default)");
    }
    return KvSplitWidthPlan::RANK_LOCAL;
  }
  if (dst_kv_split_size == 1) {
    return KvSplitWidthPlan::STRIDED;
  }
  return reject(
      "incompatible KV split widths, source=" +
      std::to_string(src_kv_split_size) +
      ", destination=" + std::to_string(dst_kv_split_size) +
      ": only equal widths or an unsplit destination (width 1) are supported");
}

bool KVCacheTransfer::validate_transfer_mappings(
    const std::vector<KVTransferMapping>& mappings,
    const std::string& request_id,
    int32_t kv_split_size,
    bool rank_local_mapping) {
  if (kv_split_size < 1) {
    LOG(ERROR) << "KV cache transfer requires kv_split_size >= 1, request_id="
               << request_id << ", kv_split_size=" << kv_split_size;
    return false;
  }

  std::unordered_set<int32_t> group_ids;
  group_ids.reserve(mappings.size());
  for (const KVTransferMapping& mapping : mappings) {
    if (!group_ids.emplace(mapping.group_id).second) {
      LOG(ERROR) << "Duplicate KV cache transfer mapping, request_id="
                 << request_id << ", group_id=" << mapping.group_id;
      return false;
    }

    const std::optional<BlockType> block_type =
        block_type_from_cache_group_id(mapping.group_id);
    const bool validate_full_kv_split_coverage =
        kv_split_size > 1 && !rank_local_mapping && block_type.has_value() &&
        is_kv_split_cache_block_type(block_type.value());
    if (!validate_full_kv_split_coverage) {
      if (mapping.local_ids.size() != mapping.remote_ids.size()) {
        LOG(ERROR) << "KV cache transfer mapping size mismatch, request_id="
                   << request_id << ", group_id=" << mapping.group_id
                   << ", local=" << mapping.local_ids.size()
                   << ", remote=" << mapping.remote_ids.size();
        return false;
      }
      continue;
    }

    const size_t local_count = mapping.local_ids.size();
    const size_t remote_count = mapping.remote_ids.size();
    if (local_count == 0) {
      if (remote_count != 0) {
        LOG(ERROR) << "KV-split mapping has remote ids without local ids, "
                   << "request_id=" << request_id
                   << ", group_id=" << mapping.group_id
                   << ", remote=" << remote_count;
        return false;
      }
      continue;
    }

    const size_t split_size = static_cast<size_t>(kv_split_size);
    if (local_count > std::numeric_limits<size_t>::max() / split_size) {
      LOG(ERROR) << "KV-split mapping coverage size overflow, request_id="
                 << request_id << ", group_id=" << mapping.group_id
                 << ", local=" << local_count
                 << ", kv_split_size=" << kv_split_size;
      return false;
    }
    const size_t max_remote_count = local_count * split_size;
    // A fully covered source block spans split_size destination blocks, but the
    // last source block of a prompt is usually partial: only the destination
    // blocks that hold real tokens exist, and the builder stops at the end of
    // the destination list. The honest lower bound is one destination block per
    // source block -- a block that holds tokens always owns at least one -- not
    // the tight `max - split + 1`, which rejected valid transfers (a 246-token
    // tail across two source blocks asked for 2 destination blocks and was
    // refused with expected_remote_range=[5, 8]). Width correctness is enforced
    // by validate_kv_split_width_plan and the post-filter 1:1 check covers the
    // per-rank shape; this gate rejects a mapping that claims more destination
    // blocks than the stride can cover, or one that leaves a source block
    // without any destination block.
    const size_t min_remote_count = local_count;
    if (remote_count < min_remote_count || remote_count > max_remote_count) {
      LOG(ERROR) << "KV-split mapping remote coverage mismatch, request_id="
                 << request_id << ", group_id=" << mapping.group_id
                 << ", local=" << local_count << ", remote=" << remote_count
                 << ", kv_split_size=" << kv_split_size
                 << ", expected_remote_range=[" << min_remote_count << ", "
                 << max_remote_count << "]";
      return false;
    }
  }
  return true;
}

bool KVCacheTransfer::validate_transfer_mappings(
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    int32_t kv_split_size) {
  for (const TransferKVInfo& info : transfer_kv_infos) {
    if (!validate_transfer_mappings(info.mappings,
                                    info.request_id,
                                    kv_split_size,
                                    info.rank_local_mapping)) {
      return false;
    }
  }
  return true;
}

bool KVCacheTransfer::validate_kv_split_width_plan(
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    int32_t src_kv_split_size) {
  // The push path re-reads the process's declared write mode: it is the same
  // value register_instance_info put on the registration record, so this is
  // defense in depth for the scheduler-side dispatch gate.
  const int32_t src_cp_index_write_mode = declared_cp_index_write_mode();
  for (const TransferKVInfo& info : transfer_kv_infos) {
    // The push-side defense must classify BOTH mapping kinds: a strided
    // info is only legitimate for the pairs plan_kv_split_widths calls
    // STRIDED, and an inexpressible pair (e.g. a width-1 source feeding a
    // width-2 destination) reaches this function with
    // rank_local_mapping=false -- rank_local_mapping is set by the
    // scheduler-side resolve, so a regressed scheduler gate would build
    // exactly that info, and a plain "skip non-rank-local" would let it
    // through to the unfiltered 1:1 remote_ids consumption.
    std::string reason;
    const KvSplitWidthPlan plan =
        plan_kv_split_widths(src_kv_split_size,
                             info.remote_instance_info.kv_split_size,
                             src_cp_index_write_mode,
                             info.remote_instance_info.cp_index_write_mode,
                             &reason);
    if (plan == KvSplitWidthPlan::UNRECONCILABLE) {
      LOG(ERROR) << "Refusing KV cache transfer, request_id=" << info.request_id
                 << ": " << reason;
      return false;
    }
    // This branch is the scheduler-gate regression catch: rank_local_mapping
    // is set by the dispatch-side resolve, and a mapper claiming 1:1 against
    // a plan the widths do not license (STRIDED here -- UNRECONCILABLE is
    // handled above and RANK_LOCAL cannot contradict itself) would have both
    // owners write the same destination blocks. plan_kv_split_widths only
    // fills `reason` on the UNRECONCILABLE path, so name the contradiction
    // and the actual pair here instead of logging an empty tail.
    if (info.rank_local_mapping && plan != KvSplitWidthPlan::RANK_LOCAL) {
      LOG(ERROR) << "Refusing KV cache transfer, request_id=" << info.request_id
                 << ": rank-local mapping contradicts the width plan, source="
                 << src_kv_split_size
                 << ", destination=" << info.remote_instance_info.kv_split_size
                 << " (the pair plans to STRIDED; the scheduler gate must "
                    "regress for a mapper to claim rank-local here)";
      return false;
    }
  }
  return true;
}

folly::SemiFuture<bool> KVCacheTransfer::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  if (!validate_transfer_mappings(
          mappings, /*request_id=*/"PULL", /*kv_split_size=*/1)) {
    promise.setValue(false);
    return future;
  }
  threadpool_.schedule([this,
                        src_cluster_id,
                        src_addr,
                        mappings,
                        promise = std::move(promise)]() mutable {
    const bool success = pull_kv_blocks(src_cluster_id, src_addr, mappings);
    promise.setValue(success);
  });
  return future;
}

// In KV-split mode, each block-scoped mapping's local_ids already contains
// only this rank's physical blocks. remote_ids holds the D-side block entries
// for every rank, expanded by the source stride and starting at the first
// destination block this mapping may write; this rank keeps the stripe it owns
// of each block, i.e. the entry at
// (local_block_start + k) * kv_split_size + kv_split_rank - remote_base, where
// remote_base is the first destination block the step may write. The function
// rebuilds remote_ids accordingly and drops infos with no mappings.
std::vector<TransferKVInfo> filter_kv_split_infos(
    int32_t kv_split_rank,
    int32_t kv_split_size,
    const std::vector<TransferKVInfo>& kv_infos) {
  std::vector<TransferKVInfo> filtered_kv_infos;
  for (const TransferKVInfo& kv_info : kv_infos) {
    if (kv_info.rank_local_mapping) {
      filtered_kv_infos.emplace_back(kv_info);
      continue;
    }
    TransferKVInfo filtered = kv_info;
    for (KVTransferMapping& mapping : filtered.mappings) {
      const std::optional<BlockType> block_type =
          block_type_from_cache_group_id(mapping.group_id);
      if (!block_type.has_value() ||
          !is_kv_split_cache_block_type(block_type.value())) {
        continue;
      }
      const std::vector<uint64_t>& remote_ids = mapping.remote_ids;
      const size_t stride = static_cast<size_t>(kv_split_size);
      // Positions alone do not identify a block: a per-step mapping starts at
      // the transfer cursor and may have dropped whole blocks the shared
      // prefix already covers, so the builder records the block index of
      // local_ids[0]. A later dispatch step starts well past
      // floor(shared / stride), and deriving the start from the shared count
      // would shift every stripe by that difference. The emitted destination
      // slice starts at this block's first stripe or after the shared prefix,
      // whichever comes later.
      const size_t block_start = static_cast<size_t>(mapping.local_block_start);
      const size_t remote_base = std::max(
          block_start * stride, static_cast<size_t>(mapping.remote_shared_num));
      std::vector<uint64_t> kept_local;
      std::vector<uint64_t> kept_remote;
      kept_local.reserve(mapping.local_ids.size());
      kept_remote.reserve(mapping.local_ids.size());
      for (size_t k = 0; k < mapping.local_ids.size(); ++k) {
        const size_t remote_block =
            (block_start + k) * stride + static_cast<size_t>(kv_split_rank);
        if (remote_block < remote_base) {
          // This rank's stripe of the straddling block is already on the
          // destination, so dropping the pair keeps both sides the same size.
          continue;
        }
        const size_t remote_idx = remote_block - remote_base;
        if (remote_idx >= remote_ids.size()) {
          break;
        }
        if (kept_local.empty()) {
          mapping.local_block_start = static_cast<uint64_t>(block_start + k);
        }
        kept_local.emplace_back(mapping.local_ids[k]);
        kept_remote.emplace_back(remote_ids[remote_idx]);
      }
      mapping.local_ids = std::move(kept_local);
      mapping.remote_ids = std::move(kept_remote);
    }
    // local_ids[k] maps to the destination stripe kv_split_rank of block
    // local_block_start + k. When the strided remote index runs past the
    // D-side block list (the prompt
    // spans multiple logical blocks and the last one is not full, which only
    // happens for kv_split_rank > 0), the loop above stops early. local_ids
    // must then be truncated to the blocks that actually got a remote target;
    // otherwise the two sides differ in size and PushKvBlocks rejects the whole
    // transfer. The dropped tail blocks correspond to tokens beyond the prompt
    // length, so the truncation is loss-free.
    const bool has_mapping = std::any_of(filtered.mappings.begin(),
                                         filtered.mappings.end(),
                                         [](const KVTransferMapping& mapping) {
                                           return !mapping.local_ids.empty() &&
                                                  !mapping.remote_ids.empty();
                                         });
    if (has_mapping) {
      filtered_kv_infos.push_back(std::move(filtered));
    }
  }
  return filtered_kv_infos;
}

void advance_transfer_cursor_to_shared_prefix(Sequence* sequence,
                                              uint32_t kv_split_size) {
  if (sequence == nullptr) {
    return;
  }
  const std::optional<TransferKVInfo>& transfer_info =
      sequence->kv_state().transfer_kv_info();
  if (!transfer_info.has_value()) {
    return;
  }
  for (const KVTransferMapping& mapping : transfer_info->mappings) {
    if (mapping.remote_shared_num == 0) {
      // No shared prefix for this group: nothing to skip, and a no-op advance.
      continue;
    }
    const std::optional<BlockType> block_type =
        block_type_from_cache_group_id(mapping.group_id);
    if (!block_type.has_value()) {
      LOG(ERROR) << "Unknown KV cache transfer group_id: " << mapping.group_id;
      continue;
    }
    const size_t source_shared = source_blocks_in_remote_shared(
        mapping.remote_shared_num,
        transfer_remote_stride(block_type.value(),
                               transfer_info->rank_local_mapping,
                               kv_split_size));
    if (source_shared == 0) {
      continue;
    }
    if (block_type.value() == BlockType::KV) {
      sequence->kv_state().advance_transfer_block_idx(source_shared);
    } else {
      sequence->kv_state().advance_group_transfer_block_idx(block_type.value(),
                                                            source_shared);
    }
  }
}

std::vector<std::string> KVCacheTransfer::rotate_dst_rank(
    const std::vector<std::string>& keys,
    int32_t kv_split_rank) {
  int32_t offset = kv_split_rank;
  std::vector<std::string> rotated_keys;
  auto sorted_keys = keys;
  std::sort(sorted_keys.begin(), sorted_keys.end());
  for (int32_t i = 0; i < keys.size(); i++) {
    rotated_keys.emplace_back(sorted_keys[(i + offset) % sorted_keys.size()]);
  }
  return rotated_keys;
}

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU) || \
    defined(USE_MUSA)
folly::SemiFuture<std::vector<KVTransferTaskResult>>
KVCacheTransfer::push_kv_blocks_async(
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args,
    std::shared_ptr<KVPushSynchronizerImpl> layer_synchronizer,
    bool is_spec_draft) {
  folly::Promise<std::vector<KVTransferTaskResult>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        transfer_kv_infos = transfer_kv_infos,
                        parallel_args,
                        layer_synchronizer,
                        is_spec_draft,
                        promise = std::move(promise)]() mutable {
    bool push_safe = false;
    auto completion = folly::makeGuard([this, &push_safe, is_spec_draft]() {
      if (!is_spec_draft) {
        notify_push_completion(std::nullopt, push_safe);
      }
    });
    if (is_spec_draft && !prepare_draft_transfer_infos(transfer_kv_infos)) {
      // Structured request failure: no source consumer remains for these
      // mappings, so shared prefill slots do not need to stay quarantined.
      push_safe = true;
      promise.setValue(failed_tasks_for_requests(transfer_kv_infos));
      return;
    }
    std::unordered_map<std::string, KVCacheInfo> merged_kv_infos;
    std::vector<TransferKVInfo> filtered_kv_infos;
    const std::vector<TransferKVInfo>* kv_infos = &transfer_kv_infos;
    // Filter when KV is actually sharded across ranks. When
    // kv_split_size==1 (each CP rank holds a full KV replica) the filter
    // degenerates to a copy, so we skip it and let each rank consume
    // remote_ids 1:1.
    const int32_t kv_split_size = parallel_args.kv_split_size_effective();
    // The mapping decision is checked here, where `kv_split_size` really is the
    // source's declared width. It deliberately does NOT live inside
    // validate_transfer_mappings(): the post-filter call below passes the
    // literal 1, which means "each rank's mapping is 1:1 now", not "the source
    // declares width 1", and reading it as a width refused every push to a
    // destination that declares kv_split_size > 1.
    if (!validate_kv_split_width_plan(*kv_infos, kv_split_size)) {
      push_safe = true;
      promise.setValue(failed_tasks_for_requests(transfer_kv_infos));
      return;
    }
    if (!validate_transfer_mappings(*kv_infos, kv_split_size)) {
      push_safe = true;
      promise.setValue(failed_tasks_for_requests(transfer_kv_infos));
      return;
    }
    if (kv_split_size > 1) {
      filtered_kv_infos = filter_kv_split_infos(
          parallel_args.kv_split_rank(), kv_split_size, *kv_infos);
      kv_infos = &filtered_kv_infos;
      if (kv_infos->empty()) {
        push_safe = true;
        promise.setValue(std::vector<KVTransferTaskResult>{});
        return;
      }
    }
    // Post-filter identity: every rank's local_ids and remote_ids must line up
    // 1:1. The literal 1 below is that expectation, not a declared width.
    if (!validate_transfer_mappings(*kv_infos, /*kv_split_size=*/1)) {
      push_safe = true;
      promise.setValue(failed_tasks_for_requests(transfer_kv_infos));
      return;
    }
    merge_kv_blocks(merged_kv_infos, *kv_infos, parallel_args);
    std::vector<KVTransferTaskResult> results;
    if (!merged_kv_infos.empty()) {
      results = this->push_kv_blocks(merged_kv_infos,
                                     layer_synchronizer,
                                     is_spec_draft,
                                     parallel_args.kv_split_rank(),
                                     parallel_args.kv_split_size_effective());
    }
    // Unsafe layer completions were already reported by the sender. Structured
    // request failures (e.g. rejected bindings) leave no live source consumers.
    push_safe = true;
    promise.setValue(std::move(results));
  });
  return future;
}

#endif

void KVCacheTransfer::merge_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args) {
  // Obtain the parallel parameters of the source instance.
  // When CP is enabled on the P side, the per-DP worker count is
  // cp_size * tp_size. We need the *actual* TP size (excluding CP) so that
  // src_dp_local_tp_rank correctly reflects only the TP dimension.
  // Using cp_size * tp_size here would make CP rank > 0 workers appear to
  // have a tp_rank >= dst_world_size, causing the linked_dp_ranks filter to
  // skip all requests for those workers.
  int32_t src_rank = parallel_args.rank();
  int32_t src_dp_size = parallel_args.dp_size();
  int32_t src_kv_split_size = parallel_args.kv_split_size_effective();
  int32_t src_world_size = parallel_args.world_size();
  int32_t src_tp_size = src_world_size / src_dp_size / src_kv_split_size;
  int32_t src_dp_local_tp_rank = src_rank % src_tp_size;
  auto append_mappings = [](std::vector<KVTransferMapping>& dst,
                            const std::vector<KVTransferMapping>& src) {
    for (const KVTransferMapping& src_mapping : src) {
      auto it = std::find_if(dst.begin(),
                             dst.end(),
                             [&src_mapping](const KVTransferMapping& mapping) {
                               return mapping.group_id == src_mapping.group_id;
                             });
      if (it == dst.end()) {
        dst.emplace_back(src_mapping);
        continue;
      }
      it->local_ids.insert(it->local_ids.end(),
                           src_mapping.local_ids.begin(),
                           src_mapping.local_ids.end());
      it->remote_ids.insert(it->remote_ids.end(),
                            src_mapping.remote_ids.begin(),
                            src_mapping.remote_ids.end());
    }
  };
  std::unordered_map<std::string, std::unordered_set<std::string>>
      seen_request_ids;
  for (auto& info : transfer_kv_infos) {
    // Obtain the parallel parameters of the destination instance.
    int32_t dst_dp_rank = info.dp_rank;
    int32_t dst_dp_size = info.remote_instance_info.dp_size;
    int32_t dst_world_size = info.remote_instance_info.cluster_ids.size();
    int32_t dst_tp_size = dst_world_size / dst_dp_size;
    // Get the DP groups of the destination instance connected to the current
    // worker.
    std::unordered_set<int32_t> linked_dp_ranks;
    for (int32_t i = src_dp_local_tp_rank; i < dst_world_size;
         i += src_tp_size) {
      int32_t linked_dp_rank = i / dst_tp_size;
      linked_dp_ranks.emplace(linked_dp_rank);
    }
    // If the target DP rank of the request is not linked to the current worker,
    // skip the request.
    if (linked_dp_ranks.find(dst_dp_rank) == linked_dp_ranks.end()) {
      continue;
    }
    // The current worker needs to push the KV Cache to all workers in the
    // destination DP group it is connected to.
    for (int32_t i =
             src_dp_local_tp_rank % dst_tp_size + dst_tp_size * dst_dp_rank;
         i < dst_tp_size * (dst_dp_rank + 1);
         i += src_tp_size) {
      uint64_t dst_cluster_id = info.remote_instance_info.cluster_ids[i];
      auto& dst_addr = info.remote_instance_info.addrs[i];
      std::string key = std::to_string(dst_cluster_id) + "_" + dst_addr;
      // Merge all kv blocks with the same destination worker into a single
      // vector.
      if (merged_kv_infos.find(key) == merged_kv_infos.end()) {
        KVCacheInfo kv_info;
        kv_info.dst_cluster_id = dst_cluster_id;
        kv_info.dst_addr = dst_addr;
        append_mappings(kv_info.mappings, info.mappings);
        append_unique_request_id(
            seen_request_ids, key, kv_info.request_ids, info.request_id);

        merged_kv_infos[key] = std::move(kv_info);
      } else {
        append_mappings(merged_kv_infos[key].mappings, info.mappings);
        append_unique_request_id(seen_request_ids,
                                 key,
                                 merged_kv_infos[key].request_ids,
                                 info.request_id);
      }
    }
  }
}

std::shared_ptr<KVCacheTransfer> KVCacheTransferFactory::create(
    uint16_t transfer_listen_port,
    const Device& device,
    const std::string& model_type,
    const std::string& model_id) {
  std::shared_ptr<KVCacheTransfer> transfer;

  int32_t device_id = device.index();

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU) || \
    defined(USE_MUSA)
  LOG(INFO) << "Create Mooncake KVCacheTransfer.";
  transfer = std::make_shared<MooncakeKVCacheTransferDefault>(
      device_id, transfer_listen_port, device, model_type);
#endif

  return transfer;
}

}  // namespace xllm
