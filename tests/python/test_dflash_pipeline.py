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

from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.python import test_dspark_pipeline as sampling_tests
from tests.python import test_execution_pipeline as pipeline_tests
from tests.python.test_dflash2_pipeline import _Backbone, _Head
from tests.python.test_dspark_pipeline import _pipeline_inputs, _SamplingInput
from xllm.python.model_executor.execution_pipeline import create_execution_pipeline
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.model_output import ModelOutput
from xllm.python.model_executor.runner import RunnerConfig
from xllm.python.model_executor.runners import base as runner_base
from xllm.python.model_executor.sampling_metadata import ModelSamplingMetadataBuilder, SamplingMode
from xllm.python.models.base import PyModelBase
from xllm.python.models.qwen3_dflash import DFlashQwen3ForCausalLM
from xllm.python.pipelines.dflash_pipeline import DFlashExecutionPipeline
from xllm.python.registry import get_execution_metadata_builder_classes

acl_graph_runtime = pipeline_tests.acl_graph_runtime
capture_lock_state = pipeline_tests.capture_lock_state
_npu_platform = pipeline_tests._npu_platform
sampling_runtime = sampling_tests.sampling_runtime


class _Model(PyModelBase):
    create_execution_pipeline = DFlashQwen3ForCausalLM.create_execution_pipeline

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(vocab_size=11, num_speculative_tokens=width - 1)
        self.model = _Backbone()
        del self.model.candidate_selector
        self.lm_head = _Head()
        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)


def _inputs(requests: int = 2, width: int = 4, *, constrained: bool = False, offset: int = 0) -> ModelInput:
    tokens = (torch.arange(requests * width) + offset) % 11
    sampling = _SamplingInput(is_unconstrained_greedy=not constrained)
    inputs = _pipeline_inputs(tokens, torch.arange(tokens.numel()), sampling, width=width)
    inputs.metadata.use_dflash_proposal_attention = True
    return inputs


def _pipeline(model: _Model, *, graph: bool = False, dp_size: int = 1) -> DFlashExecutionPipeline:
    pipeline = create_execution_pipeline(
        model,
        (ModelSamplingMetadataBuilder(),),
        RunnerConfig(torch.device("cpu"), {"dp_size": dp_size}, enable_graph=graph, max_seqs_per_batch=4),
    )
    assert isinstance(pipeline, DFlashExecutionPipeline)
    assert len(pipeline.runners) == 2
    return pipeline


def _reference(model: _Model, inputs: ModelInput) -> torch.Tensor:
    requests = inputs.input_batch.num_reqs
    hidden = model.model(inputs.input_ids, inputs.positions).view(requests, (model.cfg.num_speculative_tokens + 1), -1)[
        :, 1:
    ]
    logits = model.compute_logits(hidden.reshape(-1, hidden.shape[-1]), None)
    return logits.argmax(-1).view(requests, model.cfg.num_speculative_tokens)


@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("requests", [1, 3])
def test_pipeline_preserves_parallel_greedy_sampling(width: int, requests: int, sampling_runtime: ModuleType) -> None:
    model = _Model(width)
    inputs = _inputs(requests, width)
    with torch.no_grad():
        expected = _reference(model, inputs)
        model.lm_head.rows.clear()
        output = _pipeline(model).execute(inputs)
    assert isinstance(output, ModelOutput)
    torch.testing.assert_close(output.draft_token_ids, expected)
    assert output.draft_logits is None
    assert model.lm_head.rows == [requests * (width - 1)]
    sampling_runtime.sample_logits.assert_called_once()


def test_sampling_fallback_keeps_decoder_graph_and_recovers(
    acl_graph_runtime: list[SimpleNamespace], sampling_runtime: ModuleType
) -> None:
    model = _Model()
    pipeline = _pipeline(model, graph=True)
    with torch.no_grad():
        for offset, (requests, constrained) in enumerate(((4, False), (3, True), (2, False), (3, False), (1, True))):
            inputs = _inputs(requests, constrained=constrained, offset=offset)
            expected = _reference(model, inputs)
            output = pipeline.execute(inputs)
            assert isinstance(output, ModelOutput)
            assert output.draft_token_ids is not None
            torch.testing.assert_close(output.draft_token_ids, expected)
            assert output.draft_logits is None
            assert pipeline.entry_runner.execution_mode == "graph"
            assert pipeline.proposal_runner.execution_mode == ("eager" if constrained else "graph")


def test_constrained_sampling_uses_native_parameters(sampling_runtime: ModuleType) -> None:
    inputs = _inputs(constrained=True)
    inputs.input_batch.sampling.parameters.selected_token_idxes = torch.tensor([1, 2, 3, 5, 6, 7])
    inputs.input_batch.sampling.parameters.sample_idxes = torch.arange(6)
    original = inputs.input_batch.sampling
    sampling_runtime.sample_logits.side_effect = lambda logits, sampling: (torch.full((logits.shape[0],), 7), None)
    with torch.no_grad():
        output = _pipeline(_Model()).execute(inputs)
    assert isinstance(output, ModelOutput)
    assert output.draft_token_ids is not None
    assert output.draft_token_ids.tolist() == [[7, 7, 7], [7, 7, 7]]
    prepared = sampling_runtime.sample_logits.call_args.args[1]
    assert prepared.parameters.selected_token_idxes.numel() == 0
    assert prepared.parameters.sample_idxes.numel() == 0
    assert original.parameters.selected_token_idxes.numel() == 6
    sampling_runtime.sampling_parameters_to.assert_called_once()


@pytest.mark.parametrize("dummy", [True, False])
def test_unsampled_eager_call_runs_only_backbone(dummy: bool) -> None:
    model = _Model()
    inputs = _inputs(1)
    inputs = replace(inputs, input_batch=replace(inputs.input_batch, is_dummy=dummy, sampling=None))
    pipeline = _pipeline(model)
    with torch.no_grad():
        output = pipeline.execute(inputs)
        torch.testing.assert_close(output, model.model(inputs.input_ids, inputs.positions))
    assert not model.lm_head.rows
    assert pipeline.proposal_runner.execution_mode is None


@pytest.mark.parametrize("dummy", [True, False])
def test_unsampled_graph_warmup_keeps_head_for_later_proposals(
    acl_graph_runtime: list[SimpleNamespace], dummy: bool
) -> None:
    model = _Model()
    pipeline = _pipeline(model, graph=True)
    inputs = _inputs(4)
    warmup = replace(inputs, input_batch=replace(inputs.input_batch, is_dummy=dummy, sampling=None))
    with torch.no_grad():
        assert isinstance(pipeline.execute(warmup), torch.Tensor)
        assert bool(model.lm_head.rows) is not dummy
        current = _inputs(3, offset=2)
        output = pipeline.execute(current)
        assert isinstance(output, ModelOutput)
        assert output.draft_token_ids is not None
        torch.testing.assert_close(output.draft_token_ids, _reference(model, current))


@pytest.mark.parametrize("agreed", [True, False])
def test_idle_dp_participates_without_sampling(
    acl_graph_runtime: list[SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    agreed: bool,
    sampling_runtime: ModuleType,
) -> None:
    pipeline = _pipeline(_Model(), graph=True, dp_size=2)
    inputs = _inputs(1)
    inputs = replace(inputs, input_batch=replace(inputs.input_batch, num_reqs=0, sampling=None, is_dummy=True))
    vote = Mock(return_value=agreed)
    monkeypatch.setattr(runner_base.distributed, "dp_all_true", vote, raising=False)
    with torch.no_grad():
        assert isinstance(pipeline.execute(inputs), torch.Tensor)
    vote.assert_called_once_with(True, torch.device("cpu"), 2)
    sampling_runtime.sample_logits.assert_not_called()
    assert pipeline.proposal_runner.execution_mode is None


@pytest.mark.parametrize("constrained,agreed", [(False, True), (False, False), (True, False)])
def test_active_dp_sampling_uses_shared_agreement(
    acl_graph_runtime: list[SimpleNamespace], monkeypatch: pytest.MonkeyPatch, constrained: bool, agreed: bool
) -> None:
    model = _Model()
    pipeline = _pipeline(model, graph=True, dp_size=2)
    vote = Mock(return_value=agreed)
    monkeypatch.setattr(runner_base.distributed, "dp_all_true", vote, raising=False)
    inputs = _inputs(constrained=constrained)
    with torch.no_grad():
        output = pipeline.execute(inputs)
        assert isinstance(output, ModelOutput)
        assert output.draft_token_ids is not None
        torch.testing.assert_close(output.draft_token_ids, _reference(model, inputs))
    vote.assert_called_once_with(not constrained, torch.device("cpu"), 2)
    assert pipeline.entry_runner.execution_mode == "graph"
    assert pipeline.proposal_runner.execution_mode == ("graph" if agreed else "eager")


def test_invalid_query_and_sampling_modes_are_rejected() -> None:
    pipeline = _pipeline(_Model())
    with pytest.raises(ValueError, match="complete query blocks"):
        pipeline.execute(_inputs(width=3))
    for probabilities, greedy in ((True, True), (False, False)):
        inputs = _inputs()
        assert inputs.input_batch is not None
        sampling = inputs.input_batch.sampling
        assert sampling is not None
        sampling.return_draft_logits = probabilities
        sampling.parameters.sampling_mode = SamplingMode.GREEDY if greedy else SamplingMode.RANDOM
        with torch.no_grad(), pytest.raises(ValueError, match="must remain greedy"):
            pipeline.execute(inputs)


def test_dflash_registry_exposes_shared_sampling_builder() -> None:
    assert ModelSamplingMetadataBuilder in get_execution_metadata_builder_classes("DFlashDraftModel")
