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

"""Shared checkpoint precision metadata used before model construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xllm.python.model_loader.module_loaders import (
    checkpoint_weight_is_quantized,
    quant_description_weight_is_quantized,
    read_quant_description,
)
from xllm.python.models.deepseek_v41 import V41_QUANT_ASCEND, _v41_unquantized_attn_projections
from xllm.python.models.deepseek_v41_dspark import checkpoint_weight_is_quantized as draft_weight_is_quantized


@pytest.mark.parametrize("label", ["FLOAT", "FLOAT16", "FLOAT32", "BF16", "BFLOAT16", "FP16", "FP32"])
@pytest.mark.parametrize("prefix", ["", "model."])
def test_float_aliases_match_target_and_dspark(tmp_path: Path, label: str, prefix: str) -> None:
    names = ["layers.0.attn.wq_a.weight", "mtp.0.attn.wq_a.weight", "mtp.0.main_proj.weight"]
    description = {prefix + name: " " + label.lower() + " " for name in names}
    (tmp_path / "quant_model_description.json").write_text(json.dumps(description), encoding="utf-8")
    assert read_quant_description(str(tmp_path)) == description
    for name in names:
        assert checkpoint_weight_is_quantized(str(tmp_path), name) is False
        assert draft_weight_is_quantized(str(tmp_path), name) is False
    config = {"model_path": str(tmp_path)}
    for checkpoint_prefix in ["layers", "mtp"]:
        assert _v41_unquantized_attn_projections(config, V41_QUANT_ASCEND, 1, checkpoint_prefix) == frozenset(
            {(0, "wq_a")}
        )


@pytest.mark.parametrize(
    ("bare_label", "prefixed_label", "expected"),
    [
        ("W8A8_DYNAMIC", "FLOAT", True),
        ("FLOAT", "W8A8_DYNAMIC", False),
        (None, " BF16 ", False),
        (3, "FLOAT", None),
        (False, "FLOAT", None),
        ({}, "FLOAT", None),
        ([], "FLOAT", None),
        ("W8A8_DYNAMIC", None, True),
        (None, None, None),
    ],
)
def test_label_precedence_and_missing_labels(bare_label: object, prefixed_label: object, expected: bool | None) -> None:
    name = "mtp.0.main_proj.weight"
    description = {name: bare_label, "model." + name: prefixed_label}
    assert quant_description_weight_is_quantized(description, name) is expected
    assert quant_description_weight_is_quantized(description, "missing.weight") is None
    assert quant_description_weight_is_quantized(None, name) is None


def test_missing_description_preserves_unknown_precision(tmp_path: Path) -> None:
    for path in [None, "", str(tmp_path), str(tmp_path / "missing")]:
        assert read_quant_description(path) is None
        assert checkpoint_weight_is_quantized(path, "mtp.0.main_proj.weight") is None


@pytest.mark.parametrize("contents", ["[]", "null", "1", "true", '"FLOAT"'])
def test_non_object_description_rejected_consistently(tmp_path: Path, contents: str) -> None:
    (tmp_path / "quant_model_description.json").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_quant_description(str(tmp_path))
    with pytest.raises(ValueError, match="JSON object"):
        draft_weight_is_quantized(str(tmp_path), "mtp.0.main_proj.weight")
    with pytest.raises(ValueError, match="JSON object"):
        _v41_unquantized_attn_projections({"model_path": str(tmp_path)}, V41_QUANT_ASCEND, 1)


def test_invalid_json_is_not_silently_ignored(tmp_path: Path) -> None:
    (tmp_path / "quant_model_description.json").write_text("{invalid", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_quant_description(str(tmp_path))
