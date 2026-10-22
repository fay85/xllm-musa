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

namespace xllm::runtime::musa {

// The MUSA graph executor currently provides only the native graph pool.  Keep
// the backend on that pool even when the cross-platform VMM option defaults to
// true; entering the VMM path would call the deliberately unimplemented stubs.
inline constexpr bool supports_graph_vmm_pool() { return false; }

}  // namespace xllm::runtime::musa
