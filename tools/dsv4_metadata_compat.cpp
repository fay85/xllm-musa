/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "core/kernels/musa/musa_dsv4_ops_api.h"

namespace xllm::kernel::musa {

// The base kernel archive predates the metadata-key setter. The current
// Attention object references it even when the isolated legacy q6 forward is
// selected; no batched-MATE current path is entered in this A/B.
void deepseek_v4_mate_sparse_attention_set_metadata_key(int64_t) {}

}  // namespace xllm::kernel::musa
