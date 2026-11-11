# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Audit scope/row identity/immutable snapshots; no mocked inference math."""
from pathlib import Path

import pytest
import torch

from xllm.python.models.deepseek_v41_numeric_audit import V41NumericAudit, create_numeric_audit


def test_default_disabled_creates_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('XLLM_DSV41_NUMERIC_AUDIT_DIR', raising=False)
    assert create_numeric_audit(torch.device('musa'), 0) is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('device,rank', [('cpu', 0), ('musa', 1)])
def test_other_device_or_rank_does_no_io(device: str, rank: int, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path/'absent'
    monkeypatch.setenv('XLLM_DSV41_NUMERIC_AUDIT_DIR', str(root))
    assert create_numeric_audit(torch.device(device), rank) is None
    assert not root.exists()


def test_relative_output_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('XLLM_DSV41_NUMERIC_AUDIT_DIR', 'relative')
    with pytest.raises(ValueError, match='absolute'):
        create_numeric_audit(torch.device('musa'), 0)


def test_graph_is_rejected_before_tensor_access(tmp_path: Path) -> None:
    audit = V41NumericAudit(tmp_path, 1)
    with pytest.raises(ValueError, match='eager'):
        audit.begin(None, None, None, True)


@pytest.mark.parametrize('lengths', [[1, 1], [], [0]])
def test_non_c1_and_empty_sequences_rejected(lengths: list[int], tmp_path: Path) -> None:
    audit = V41NumericAudit(tmp_path, 1)
    with pytest.raises(ValueError, match='one nonempty'):
        audit.begin(None, None, lengths, False)


def test_snapshots_preserve_last_row_and_full_vocab(tmp_path: Path) -> None:
    audit = V41NumericAudit(tmp_path, 2)
    tokens, positions = torch.tensor([7, 8, 9]), torch.arange(3)
    hidden = torch.arange(24).reshape(3, 2, 4).bfloat16()
    mix = torch.arange(6).reshape(3, 2).float()
    final = torch.arange(12).reshape(3, 4).bfloat16()
    logits = torch.randn(1, 101)
    audit.begin(tokens, positions, [3], False)
    audit.record_layer(0, hidden, mix)
    audit.finish(final)
    audit.record_logits(logits, torch.tensor([2]), final)
    stored = torch.load(next(tmp_path.glob('*.hidden.pt')), weights_only=True)
    sampled = torch.load(next(tmp_path.glob('*.logits.pt')), weights_only=True)
    assert torch.equal(stored['layers'][0]['hidden'], hidden[-1:])
    assert torch.equal(stored['layers'][0]['pre_mix'], mix[-1:])
    assert torch.equal(sampled['logits'], logits) and sampled['logits'].shape == (1, 101)
    assert sampled['position'] == 2
    hidden.zero_(); mix.zero_(); final.zero_(); tokens.zero_()
    assert int(stored['tokens'][0]) == 7 and int(stored['final_hidden'][0, 0]) == 8
    with pytest.raises(FileExistsError):
        audit.record_logits(logits, torch.tensor([2]), stored['final_hidden'].expand(3, -1))


def test_preselected_head_row_is_verified(tmp_path: Path) -> None:
    audit = V41NumericAudit(tmp_path, 1)
    final = torch.randn(3, 4)
    audit.begin(torch.arange(3), torch.arange(3), [3], False)
    audit.record_layer(0, final[:, None], torch.ones(3, 1))
    audit.finish(final)
    with pytest.raises(ValueError, match='differs'):
        audit.record_logits(torch.randn(1, 101), None, final[:1])
    audit.record_logits(torch.randn(1, 101), None, final[-1:])
    with pytest.raises(ValueError, match='budget'):
        audit.begin(None, None, None, False)


def test_duplicate_layer_is_rejected(tmp_path: Path) -> None:
    audit = V41NumericAudit(tmp_path, 1)
    audit.begin(torch.tensor([7]), torch.tensor([19]), [1], False)
    audit.record_layer(0, torch.zeros(1, 2, 4), torch.ones(1, 2))
    with pytest.raises(ValueError, match='twice'):
        audit.record_layer(0, torch.zeros(1, 2, 4), torch.ones(1, 2))
