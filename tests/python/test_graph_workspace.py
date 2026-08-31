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

"""CPU tests for the shared Python MUSA graph activation pool."""

from __future__ import annotations

import pytest
import torch

from xllm.python.model_executor.graph_workspace import GraphActivationPool


def test_activation_pool_reuses_wrap_slots() -> None:
    pool = GraphActivationPool(max_depth=2)
    device = torch.device("cpu")
    first = pool.get((4, 8), torch.float32, device)
    second = pool.get((4, 8), torch.float32, device)
    pool.reset()
    again = pool.get((4, 8), torch.float32, device)
    wrapped = pool.get((4, 8), torch.float32, device)
    third = pool.get((4, 8), torch.float32, device)
    assert first.data_ptr() == again.data_ptr()
    assert second.data_ptr() == wrapped.data_ptr()
    assert third.data_ptr() == first.data_ptr()
    assert first.data_ptr() != second.data_ptr()


def test_activation_pool_freeze_rejects_new_shape() -> None:
    pool = GraphActivationPool()
    pool.get((2, 2), torch.float32, torch.device("cpu"))
    pool.freeze()
    pool.reset()
    reused = pool.get((2, 2), torch.float32, torch.device("cpu"))
    assert reused.shape == (2, 2)
    with pytest.raises(RuntimeError, match="frozen ring"):
        pool.get((3, 3), torch.float32, torch.device("cpu"))
