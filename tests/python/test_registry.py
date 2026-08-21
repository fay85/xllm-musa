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

from unittest.mock import Mock

import pytest

from xllm.python import registry


def test_unsupported_model_fails_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_model = Mock()
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")
    monkeypatch.setattr(registry, "import_module", import_model)

    with pytest.raises(NotImplementedError, match="qwen3_5.*npu"):
        registry.get_model_class("qwen3_5")

    import_model.assert_not_called()


def test_npu_only_model_fails_on_musa_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_model = Mock()
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "musa")
    monkeypatch.setattr(registry, "import_module", import_model)

    with pytest.raises(NotImplementedError, match="deepseek_v32.*musa"):
        registry.get_model_class("deepseek_v32")

    import_model.assert_not_called()


def test_qwen3_5_is_imported_on_musa(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cls = type("Qwen3_5ForCausalLM", (), {})
    fake_module = Mock()
    fake_module.Qwen3_5ForCausalLM = fake_cls
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "musa")
    monkeypatch.setattr(registry, "import_module", Mock(return_value=fake_module))

    assert registry.get_model_class("qwen3_5") is fake_cls
    registry.import_module.assert_called_once_with("xllm.python.models.qwen3_5")
