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


from __future__ import annotations

from collections.abc import Callable
from typing import Any, ContextManager

import torch

from xllm.python.model_executor.runners.base import ModelExecutionOutput


def device_capture_lock(device: torch.device) -> ContextManager[None]:
    import xllm_runtime

    device_index = device.index if device.index is not None else torch.npu.current_device()
    return xllm_runtime.DeviceCaptureGuard(device_index)


def capture_acl_graph(
    forward: Callable[[], ModelExecutionOutput],
    stream: Any,
    *,
    pool: Any | None = None,
) -> tuple[Any, ModelExecutionOutput]:
    """Capture an already prepared computation; callers own warmup and context."""
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=pool, stream=stream):
        output = forward()
    return graph, output
