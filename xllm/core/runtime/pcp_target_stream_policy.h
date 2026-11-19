/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#pragma once
#include <cstdint>

namespace xllm::speculative_worker {

struct PcpTargetStreamPolicy {
  bool enabled = false;
  bool deepseek_v4 = false;
  bool pd_prefill = false;
  bool prefill_like = false;
  bool spec_verify = false;
  bool graph_warmup = false;
  bool schedule_overlap = false;
  int64_t cp_size = 1;
  int64_t dp_size = 1;
  int64_t num_sequences = 1;
};

inline bool pcp_batch_target_pool_allowed(const PcpTargetStreamPolicy& policy) {
  return policy.enabled && policy.deepseek_v4 && policy.pd_prefill &&
         policy.prefill_like && !policy.spec_verify && !policy.graph_warmup &&
         !policy.schedule_overlap && policy.cp_size == 2 && policy.dp_size == 1 &&
         policy.num_sequences == 2;
}

}  // namespace xllm::speculative_worker
