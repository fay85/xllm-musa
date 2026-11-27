# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parallelism mixins for DiT components.

Python counterparts of ``xllm/models/dit/utils/dit_parallel_mixin.h``. Each
mixin addresses its process group by NAME (registered via
``xllm.python.distributed.collectives.init_group``) instead of holding a
group object, so a module stays constructible before communication is set up
and resolves the group lazily from the tensors it is called with.

These mixins are the only place model components touch communication; the
primitives live in ``xllm.python.distributed.collectives``.
"""

from __future__ import annotations

from .sequence_parallel import SequenceParallelMixin
from .vae_parallel import VaeParallelMixin

__all__ = ["SequenceParallelMixin", "VaeParallelMixin"]
