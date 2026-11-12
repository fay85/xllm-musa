# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Cross-language scalar representation and explicit fused-gate capability."""
from __future__ import annotations

import struct
import sys
from types import SimpleNamespace

import pytest
import torch

from xllm.python.kernels_musa.engram import select_engram_gate


@pytest.mark.parametrize('eps',[1e-20,struct.unpack('=f',struct.pack('=f',1e-20))[0]])
def test_fused_accepts_both_python_and_cpp_fp32_epsilon(eps: float,monkeypatch: pytest.MonkeyPatch) -> None:
    marker=object()
    monkeypatch.setitem(sys.modules,'xllm_runtime',SimpleNamespace(musa_v41_engram_gate=marker))
    monkeypatch.setenv('XLLM_DSV41_MUSA_ENGRAM_GATE','sglang_fused')
    assert select_engram_gate(torch.device('musa'),torch.bfloat16,5120,4,eps,False) is marker


@pytest.mark.parametrize('eps',[1e-6,1e-12,0.,float('inf'),-1.])
def test_fused_rejects_different_or_invalid_epsilon(eps: float,monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('XLLM_DSV41_MUSA_ENGRAM_GATE','sglang_fused')
    with pytest.raises(ValueError):
        select_engram_gate(torch.device('musa'),torch.bfloat16,5120,4,eps,False)
