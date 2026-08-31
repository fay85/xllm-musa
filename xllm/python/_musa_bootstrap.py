# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-shot bootstrap for MUSA embedded-interpreter environment.

The C++ worker already loads libtorch_musa. This module imports ``torch_musa``
so ``torch.musa`` is available before ``xllm.python.initialize_runtime()``.
It must be imported exactly once by the C++ host (py_model_helper.cpp) before
any other xllm.python module — never by user code.
"""

import torch_musa  # noqa: F401
