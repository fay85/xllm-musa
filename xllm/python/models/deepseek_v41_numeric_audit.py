# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Opt-in C1/eager snapshots of actual last-row hidden and full logits.

Copies are numerical diagnostics, never a model fallback or benchmark. The
normal path constructs no recorder. Every artifact is created exclusively.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from scripts.logger import logger


class V41NumericAudit:
    """One model instance, sequential forwards, one logical sequence per call."""

    def __init__(self, root: Path, max_frames: int) -> None:
        if max_frames <= 0:
            raise ValueError('Numeric audit max_frames must be positive')
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_frames = max_frames
        self._counter = 0
        self._frame: dict[str, Any] | None = None
        self._finished = False

    def begin(self, tokens: torch.Tensor, positions: torch.Tensor,
              query_lengths: torch.Tensor | list[int], graph_mode: bool) -> None:
        if graph_mode:
            raise ValueError('Numeric audit requires eager; disable it for graph/performance')
        if self._counter >= self._max_frames:
            raise ValueError('Numeric audit frame budget exhausted; stop the diagnostic client')
        lengths = (query_lengths.detach().cpu().reshape(-1).tolist()
                   if isinstance(query_lengths, torch.Tensor) else list(query_lengths))
        if len(lengths) != 1 or lengths[0] <= 0:
            raise ValueError('Numeric audit requires exactly one nonempty logical sequence')
        if tokens.numel() != positions.numel() or tokens.numel() != lengths[0]:
            raise ValueError('Numeric audit token/position/query lengths disagree')
        self._counter += 1
        frame_id = f'{os.getpid()}_{self._counter:06d}'
        self._frame = {'frame_id': frame_id,
                       'tokens': tokens.detach().cpu().reshape(-1).clone(),
                       'positions': positions.detach().cpu().reshape(-1).clone(),
                       'query_lengths': lengths, 'layers': {},
                       'scope': 'C1 eager last-row diagnostics; D2H copies, not performance'}
        self._finished = False

    def record_layer(self, layer_id: int, hidden: torch.Tensor,
                     pre_mix: torch.Tensor) -> None:
        if self._frame is None or self._finished:
            raise RuntimeError('Numeric audit layer outside an active forward')
        if layer_id in self._frame['layers']:
            raise ValueError('Numeric audit layer recorded twice')
        self._frame['layers'][layer_id] = {
            'hidden': hidden[-1:].detach().cpu().clone(),
            'pre_mix': pre_mix[-1:].detach().cpu().clone()}

    def finish(self, final_hidden: torch.Tensor) -> None:
        if self._frame is None or self._finished:
            raise RuntimeError('Numeric audit finish outside an active forward')
        self._frame['final_hidden'] = final_hidden[-1:].detach().cpu().clone()
        path = self._root / (self._frame['frame_id'] + '.hidden.pt')
        with path.open('xb') as stream:
            torch.save(self._frame, stream)
        self._finished = True
        logger.info('DSV41_NUMERIC_AUDIT hidden=%s position=%s layers=%s',
                    path, int(self._frame['positions'][-1]), len(self._frame['layers']))

    def record_logits(self, logits: torch.Tensor,
                      selected_idxes: torch.Tensor | None,
                      supplied_hidden: torch.Tensor) -> None:
        if self._frame is None or not self._finished:
            raise RuntimeError('Numeric audit logits need a completed model forward')
        selected = ([] if selected_idxes is None else
                    selected_idxes.detach().cpu().reshape(-1).tolist())
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError('Numeric audit requires exactly one selected logit row')
        last_index = len(self._frame['positions']) - 1
        if selected and (selected != [last_index] or supplied_hidden.shape[0] != last_index + 1):
            raise ValueError('Numeric audit logits must select the last recorded input position')
        if not selected and supplied_hidden.shape[0] != 1:
            raise ValueError('Numeric audit needs one preselected hidden row or an explicit last-row selection')
        selected_hidden = supplied_hidden[-1:].detach().cpu()
        if not torch.equal(selected_hidden, self._frame['final_hidden']):
            raise ValueError('Numeric audit LM head hidden differs from the recorded last model row')
        record = {'frame_id': self._frame['frame_id'],
                  'position': int(self._frame['positions'][-1]),
                  'selected_idxes': selected,
                  'logits': logits.detach().cpu().clone(),
                  'scope': 'full returned vocabulary logits before sampling, not top-k logprobs'}
        path = self._root / (record['frame_id'] + '.logits.pt')
        with path.open('xb') as stream:
            torch.save(record, stream)
        logger.info('DSV41_NUMERIC_AUDIT logits=%s position=%s vocab=%s',
                    path, record['position'], logits.shape[-1])


def create_numeric_audit(device: torch.device, tp_rank: int) -> V41NumericAudit | None:
    """Select diagnostics at construction; other devices and ranks do no work."""
    root = os.environ.get('XLLM_DSV41_NUMERIC_AUDIT_DIR', '')
    if not root or device.type != 'musa' or tp_rank != 0:
        return None
    directory = Path(root)
    if not directory.is_absolute():
        raise ValueError('Numeric audit output directory must be absolute')
    return V41NumericAudit(directory, int(os.environ.get('XLLM_DSV41_NUMERIC_AUDIT_MAX_FRAMES', '128')))
