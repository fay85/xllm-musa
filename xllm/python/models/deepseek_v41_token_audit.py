# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Opt-in numerical diagnostics for actual model input tokens, never performance.

The diagnostic copies input IDs to the host and therefore synchronizes the
device. It performs no model math and must remain disabled for benchmarks.
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch

from scripts.logger import logger


def record_token_inputs(tokens: torch.Tensor, positions: torch.Tensor,
                         metadata: Any, tp_rank: int, graph_mode: bool) -> None:
    if os.environ.get('XLLM_DSV41_TOKEN_INPUT_AUDIT','0')!='1' or tp_rank!=0:
        return
    if graph_mode:
        raise ValueError('Token input audit is eager diagnostics only; disable it before graph/performance')
    dsa=getattr(metadata,'dsa_metadata',None)
    query_lengths=getattr(dsa,'seq_lens_q',None)
    if isinstance(query_lengths,torch.Tensor):
        query_lengths=query_lengths.detach().cpu().reshape(-1).tolist()
    payload={'tokens':tokens.detach().cpu().reshape(-1).tolist(),
             'positions':positions.detach().cpu().reshape(-1).tolist(),
             'query_lengths':query_lengths,'scope':'actual model input IDs; host-copy diagnostic, not benchmark'}
    logger.info('DSV41_TOKEN_INPUT_AUDIT %s',json.dumps(payload,separators=(',',':')))
