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

"""Shared activation workspace for Python MUSA graph capture.

C++ piecewise prefill uses ``PiecewiseGraphMatmulBufferPool`` so each layer
reuses a small ring of same-shaped GEMM outputs. Python captures the whole
``model()`` in one graph, so a per-module 512-token reserve keeps gigabytes of
dead buffers live and evicts weights. This pool wraps a short ring per shape
and records stable addresses for replay.
"""

from __future__ import annotations

import torch


class GraphActivationPool:
    """Grow-then-wrap ring of exact-sized activation buffers.

    ``reset()`` at the start of every captured forward so warmup, capture, and
    the recorded address sequence stay aligned. ``freeze()`` after warmup so
    capture cannot allocate a new address.
    """

    def __init__(self, max_depth: int = 8) -> None:
        if max_depth <= 0:
            raise ValueError("GraphActivationPool max_depth must be > 0")
        self._max_depth = max_depth
        self._rings: dict[
            tuple[tuple[int, ...], torch.dtype, torch.device], list[torch.Tensor]
        ] = {}
        self._cursors: dict[
            tuple[tuple[int, ...], torch.dtype, torch.device], int
        ] = {}
        self._frozen = False

    def reset(self) -> None:
        self._cursors = {key: 0 for key in self._rings}

    def freeze(self) -> None:
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def get(
        self,
        sizes: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not sizes or any(dim <= 0 for dim in sizes):
            raise ValueError(f"invalid graph workspace shape {sizes}")
        key = (sizes, dtype, device)
        ring = self._rings.get(key)
        if ring is None:
            if self._frozen:
                raise RuntimeError(
                    f"graph workspace missing frozen ring for {sizes} {dtype}"
                )
            ring = []
            self._rings[key] = ring
            self._cursors[key] = 0
        cursor = self._cursors.get(key, 0)
        slot = cursor % self._max_depth
        if slot >= len(ring):
            if self._frozen:
                raise RuntimeError(
                    f"graph workspace ring grew after freeze for {sizes}"
                )
            ring.append(torch.empty(sizes, dtype=dtype, device=device))
        self._cursors[key] = cursor + 1
        return ring[slot]
