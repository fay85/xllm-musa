# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Diagnostic gating and captured token identity; no fake model inference."""
import json
from types import SimpleNamespace

import pytest
import torch

from xllm.python.models import deepseek_v41_token_audit as audit


def test_audit_default_does_not_access_device_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('XLLM_DSV41_TOKEN_INPUT_AUDIT',raising=False)
    audit.record_token_inputs(None,None,None,0,False)


def test_audit_only_records_attention_rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('XLLM_DSV41_TOKEN_INPUT_AUDIT','1')
    audit.record_token_inputs(None,None,None,1,False)


def test_audit_captures_actual_tokens_and_sequence_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('XLLM_DSV41_TOKEN_INPUT_AUDIT','1')
    captured=[]
    monkeypatch.setattr(audit.logger,'info',lambda fmt,value:captured.append((fmt,value)))
    metadata=SimpleNamespace(dsa_metadata=SimpleNamespace(seq_lens_q=torch.tensor([2,1])))
    audit.record_token_inputs(torch.tensor([7,12,5]),torch.tensor([0,1,18]),metadata,0,False)
    assert len(captured)==1
    data=json.loads(captured[0][1])
    assert data['tokens']==[7,12,5] and data['positions']==[0,1,18] and data['query_lengths']==[2,1]


def test_audit_rejects_graph_before_reading_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('XLLM_DSV41_TOKEN_INPUT_AUDIT','1')
    with pytest.raises(ValueError,match='eager diagnostics only'):
        audit.record_token_inputs(None,None,None,0,True)
