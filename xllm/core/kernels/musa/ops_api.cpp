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

#include "core/kernels/musa/ops_api.h"

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {

torch::Tensor matmul(MatmulParams& params) {
  return xllm::kernel::musa::matmul(
      params.a, params.b, params.bias, params.output);
}

torch::Tensor fp8_block_matmul(Fp8BlockMatmulParams& params) {
  return xllm::kernel::musa::gemm_fp8_nt_groupwise(params.a,
                                                   params.b,
                                                   params.a_scale,
                                                   params.b_scale,
                                                   params.output_dtype,
                                                   params.output);
}

}  // namespace xllm::kernel::musa
