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

import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import fields, is_dataclass, replace
from importlib import import_module
from inspect import signature
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tests.python.test_dspark_pipeline import (
    _batch_metadata,
    _DraftModel,
    _pipeline_inputs,
    _reference,
    _SamplingInput,
    _SamplingParameters,
    sampling_runtime,
)
from tests.python.test_model_executor import StubAttentionBackend, _make_attention_layer
from xllm.python import distributed
from xllm.python.model_executor.execution_context import can_execute_graph
from xllm.python.model_executor.execution_pipeline import (
    SingleModelExecutionPipeline,
    create_execution_pipeline,
)
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.forward_context import (
    get_execution_context,
    get_forward_context,
    get_forward_context_or_none,
)
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.model_output import ModelOutput
from xllm.python.model_executor.runner import Runner, RunnerConfig
from xllm.python.model_executor.runners.acl_graph import AclGraphRunner
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.model_executor.sampling_metadata import ModelSamplingMetadataBuilder
from xllm.python.models.dspark_model import DSparkDecoderModel, ProposalMetadata, ProposalMetadataBuilder
from xllm.python.pipelines.dspark_pipeline import DSparkExecutionPipeline


@pytest.fixture(autouse=True)
def _npu_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = import_module(Runner.__module__).current_platform
    monkeypatch.setattr(platform, "is_npu", lambda: True)
    monkeypatch.setattr(platform, "is_cuda", lambda: False)


def _copy_output(destination: Any, source: Any) -> None:
    if isinstance(destination, torch.Tensor):
        destination.copy_(source)
    elif isinstance(destination, tuple):
        for target, value in zip(destination, source, strict=True):
            _copy_output(target, value)
    elif is_dataclass(destination):
        for field in fields(destination):
            _copy_output(getattr(destination, field.name), getattr(source, field.name))
    else:
        assert destination is None and source is None


@pytest.fixture
def capture_lock_state(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(held=False, acquisitions=0, releases=0)

    @contextmanager
    def lock(device: torch.device) -> Iterator[None]:
        assert not state.held
        state.held = True
        state.acquisitions += 1
        try:
            yield
        finally:
            state.held = False
            state.releases += 1

    for runner_type in (AclGraphRunner, DecodeAclGraphRunner):
        monkeypatch.setattr(import_module(runner_type.__module__), "device_capture_lock", lock)
    return state


@pytest.fixture
def acl_graph_runtime(monkeypatch: pytest.MonkeyPatch, capture_lock_state: SimpleNamespace) -> list[SimpleNamespace]:
    """Exercise runner storage/order on CPU, not real ACL capture support."""
    captures: list[SimpleNamespace] = []
    caller = Mock()
    npu = SimpleNamespace(
        current_stream=Mock(return_value=caller),
        Stream=Mock(side_effect=lambda **kwargs: Mock()),
        stream=lambda stream: nullcontext(),
        synchronize=Mock(),
        graph_pool_handle=Mock(side_effect=object),
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(torch.Tensor, "record_stream", Mock())

    def capture(forward: Any, stream: Any, *, pool: Any | None = None) -> tuple[Mock, Any]:
        assert capture_lock_state.held
        output = forward()
        graph = Mock()

        def replay() -> None:
            assert not capture_lock_state.held
            _copy_output(output, forward())

        graph.replay.side_effect = replay
        captures.append(SimpleNamespace(graph=graph, output=output, stream=stream, pool=pool))
        return graph, output

    monkeypatch.setattr(import_module(AclGraphRunner.__module__), "capture_acl_graph", capture)
    return captures


class _ScaleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.allow_graph = True
        self.calls = 0

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, input_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.calls += 1
        assert get_forward_context().attention_backend is None
        return (input_ids if input_embedding is None else input_embedding) * 2

    def can_execute_graph(self, inputs: ModelInput) -> bool:
        return self.allow_graph


class _HiddenStatesModel(torch.nn.Module):
    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, input_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        metadata = get_execution_context(ProposalMetadata)
        assert metadata is not None
        hidden = metadata.hidden_states
        return (input_embedding if hidden is None else hidden) * 2


def _make_model_pipeline(
    model: torch.nn.Module, *, graph_enabled: bool = True, max_requests: int = 4
) -> SingleModelExecutionPipeline:
    config = RunnerConfig(torch.device("cpu"), enable_graph=graph_enabled, max_seqs_per_batch=max_requests)
    return SingleModelExecutionPipeline(model, (), config)


def _model_inputs(hidden: torch.Tensor, *, width: int = 1) -> ModelInput:
    token_ids = torch.arange(hidden.shape[0])
    inputs = _pipeline_inputs(token_ids, token_ids.clone(), None, width=width)
    return replace(inputs, input_embedding=hidden)


@pytest.mark.parametrize("device", [torch.device("cpu", 2), torch.device("cpu")])
def test_device_capture_lock_uses_native_device_guard(monkeypatch: pytest.MonkeyPatch, device: torch.device) -> None:
    module = import_module("xllm.python.model_executor.runners.acl_graph_capture")
    guard = nullcontext()
    factory = Mock(return_value=guard)
    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(DeviceCaptureGuard=factory))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 3), raising=False)
    assert module.device_capture_lock(device) is guard
    factory.assert_called_once_with(2 if device.index is not None else 3)


def test_graph_capture_lock_covers_warmup_but_not_replay(
    acl_graph_runtime: list[SimpleNamespace], capture_lock_state: SimpleNamespace
) -> None:
    model = _ScaleModel()
    pipeline = _make_model_pipeline(model, max_requests=8)
    forward_lock_states: list[bool] = []

    def record_lock(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        forward_lock_states.append(capture_lock_state.held)

    model.register_forward_pre_hook(record_lock)
    for rows, first_capture in ((3, True), (3, False), (5, True)):
        forward_lock_states.clear()
        pipeline.execute(_model_inputs(torch.ones(rows, 2)))
        assert any(forward_lock_states) == first_capture
        assert all(forward_lock_states[:-1])
        assert not forward_lock_states[-1]
        assert not capture_lock_state.held
    assert len(acl_graph_runtime) == 2
    assert capture_lock_state.acquisitions == capture_lock_state.releases == 2


@pytest.mark.parametrize("fail_capture", [False, True])
def test_backbone_capture_lock_covers_warmup_and_capture(
    acl_graph_runtime: list[SimpleNamespace],
    capture_lock_state: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    fail_capture: bool,
) -> None:
    runner = DecodeAclGraphRunner(
        torch.nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False, graph_uses_dsa_metadata=lambda metadata: False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
    )
    runner._stream = Mock()
    entry = SimpleNamespace(static_metadata=SimpleNamespace(), execution_state=None, execution_contexts={})

    def forward(entry: Any) -> torch.Tensor:
        assert capture_lock_state.held
        return torch.ones(1)

    def synchronize() -> None:
        assert capture_lock_state.held

    monkeypatch.setattr(runner, "_forward_static", forward)
    torch.npu.synchronize.side_effect = synchronize
    capture = (
        Mock(side_effect=RuntimeError("capture failed"))
        if fail_capture
        else import_module(AclGraphRunner.__module__).capture_acl_graph
    )
    monkeypatch.setattr(import_module(DecodeAclGraphRunner.__module__), "capture_acl_graph", capture)
    with pytest.raises(RuntimeError, match="capture failed") if fail_capture else nullcontext():
        runner._capture(entry)
    assert not capture_lock_state.held
    assert capture_lock_state.acquisitions == capture_lock_state.releases == 1


def test_runner_is_ready_for_eager_execution_after_construction() -> None:
    model = _ScaleModel()
    runner = Runner(model, (), RunnerConfig(torch.device("cpu")))
    assert runner.eager_runner.model is model
    assert runner.graph_runner is None
    torch.testing.assert_close(runner.execute(_model_inputs(torch.ones(2, 2))), torch.full((2, 2), 2.0))


def test_runner_constructs_graph_with_shared_model_and_metadata() -> None:
    model = _ScaleModel()
    builder = ProposalMetadataBuilder()
    config = RunnerConfig(
        torch.device("cpu"), enable_graph=True, max_seqs_per_batch=8, acl_graph_decode_batch_size_limit=4
    )
    runner = Runner(model, (builder,), config)
    assert runner.eager_runner.model is model
    assert runner.graph_runner.model is model
    assert runner.eager_runner.execution_metadata_builders == (builder,)
    assert runner.graph_runner.execution_metadata_builders == (builder,)
    assert runner.graph_runner.max_requests == 4
    assert runner.graph_runner._graphs == {}
    assert model.calls == 0


def test_pipeline_construction_initializes_main_and_head_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _DraftModel()
    sampling_builder = ModelSamplingMetadataBuilder()
    attention_backend = StubAttentionBackend()
    restore_history = Mock()
    graph_factory = Mock()
    monkeypatch.setattr(
        import_module("xllm.python.model_executor.runners.decode_acl_graph"), "DecodeAclGraphRunner", graph_factory
    )
    config = RunnerConfig(
        torch.device("cpu"),
        model_config={"max_position_embeddings": 32, "cp_size": 2, "cp_rank": 1, "draft_query_width": 3},
        enable_graph=True,
        max_seqs_per_batch=8,
        acl_graph_decode_batch_size_limit=4,
        is_spec_draft=True,
    )
    pipeline = create_execution_pipeline(
        model.model,
        (sampling_builder,),
        config,
        attention_backend=attention_backend,
        restore_history=restore_history,
    )
    assert isinstance(pipeline, DSparkExecutionPipeline)
    graph_factory.assert_called_once()
    assert graph_factory.call_args.args[:2] == (pipeline.entry_runner.model, attention_backend)
    assert graph_factory.call_args.args[3:8] == (8, 32, 1, 0, 4)
    assert graph_factory.call_args.kwargs["is_spec_draft"]
    assert graph_factory.call_args.kwargs["draft_query_width"] == 3
    assert isinstance(pipeline.entry_runner.model, DSparkDecoderModel)
    assert pipeline.entry_runner.eager_runner.model is pipeline.entry_runner.model
    assert pipeline.entry_runner.eager_runner.attention_backend is attention_backend
    assert pipeline.entry_runner.eager_runner.cp_size == 2
    assert pipeline.entry_runner.eager_runner.cp_rank == 1
    assert pipeline.entry_runner.restore_history is restore_history
    graph_factory.return_value.bind_execution_metadata_builders.assert_called_once_with(())
    graph_factory.return_value.warmup.assert_not_called()
    graph_factory.return_value.execute.assert_not_called()
    runner = pipeline.proposal_runner
    assert runner.eager_runner.model is runner.model
    assert runner.eager_runner.attention_backend is None
    assert runner.eager_runner.cp_size == 1
    assert runner.graph_runner.model is runner.model
    assert runner.graph_runner.max_requests == 4
    assert runner.graph_runner.execution_metadata_builders == runner.metadata_builders
    assert runner.restore_history is None
    assert pipeline.proposal_runner.metadata_builders[1] is sampling_builder
    assert model.calls == []


def test_runner_selects_backend_and_clears_graph_rows_on_fallback() -> None:
    runner = Runner(torch.nn.Identity(), (), RunnerConfig(torch.device("cpu")))
    eager = Mock()
    graph = Mock()
    runner.eager_runner = eager
    runner.graph_runner = graph
    inputs = _model_inputs(torch.ones(3, 2))
    graph.last_padded_batch_size = 4
    for allowed in (True, False, True):
        graph.can_execute.return_value = allowed
        selected = graph if allowed else eager
        assert runner.execute(inputs) is selected.execute.return_value
        assert runner.execution_mode == ("graph" if allowed else "eager")
        assert runner.last_graph_rows == (4 if allowed else None)
    assert graph.warmup.call_count == 2
    assert graph.execute.call_count == 2
    eager.execute.assert_called_once()


@pytest.mark.parametrize("return_confidence", [False, True])
def test_pipeline_only_orchestrates_runner_calls(return_confidence: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _DraftModel().model.create_execution_pipeline(
        (ModelSamplingMetadataBuilder(),), RunnerConfig(torch.device("cpu"))
    )
    assert isinstance(pipeline, DSparkExecutionPipeline)
    assert pipeline.runners == [pipeline.entry_runner, pipeline.proposal_runner]
    inputs = _inputs(_SamplingInput(return_confidence=return_confidence))
    hidden = torch.ones(6, 2)
    logits = torch.ones(6, 7)
    output = ModelOutput()
    calls = Mock()
    calls.decoder.return_value = (logits, hidden)
    calls.proposal.return_value = output
    monkeypatch.setattr(pipeline.entry_runner, "execute", calls.decoder)
    monkeypatch.setattr(pipeline.proposal_runner, "execute", calls.proposal)
    assert pipeline.execute(inputs) is output
    assert [call[0] for call in calls.mock_calls] == ["decoder", "proposal"]
    assert calls.decoder.call_args.args[0] is inputs
    assert calls.proposal.call_args.args[0].input_embedding is logits
    proposal_batch = calls.proposal.call_args.args[0].input_batch
    assert proposal_batch.hidden_states is (hidden if return_confidence else None)
    assert proposal_batch.sampling is inputs.input_batch.sampling
    assert inputs.input_batch.hidden_states is None


def test_graph_reuses_argument_storage_and_resets_padding(acl_graph_runtime: list[SimpleNamespace]) -> None:
    model = _ScaleModel()
    pipeline = _make_model_pipeline(model)
    first = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    torch.testing.assert_close(pipeline.execute(_model_inputs(first)), first * 2)
    graph = pipeline.entry_runner.graph_runner
    entry = next(iter(graph._graphs.values()))
    persistent = entry.input_embedding
    address = persistent.data_ptr()
    for value in (torch.full((4, 2), 7.0), torch.full((3, 2), 11.0)):
        torch.testing.assert_close(pipeline.execute(_model_inputs(value)), value * 2)
    assert len(acl_graph_runtime) == 1
    assert persistent.data_ptr() == address
    assert torch.count_nonzero(persistent[3:]) == 0
    assert entry.input_batch.hidden_states is None
    assert entry.context.execution_contexts == {}
    assert acl_graph_runtime[0].graph.replay.call_count == 3
    assert pipeline.entry_runner.execution_mode == "graph"
    acl_graph_runtime[0].stream.wait_stream.assert_called()
    torch.npu.current_stream.return_value.wait_stream.assert_called()
    assert get_forward_context_or_none() is None


def test_graph_pool_is_lazy_shared_across_keys_and_runner_local(acl_graph_runtime: list[SimpleNamespace]) -> None:
    pipelines = [_make_model_pipeline(_ScaleModel(), max_requests=8) for _ in range(2)]
    torch.npu.graph_pool_handle.assert_not_called()
    for pipeline in pipelines:
        for rows in (3, 5, 3):
            value = torch.full((rows, 2), float(rows))
            torch.testing.assert_close(pipeline.execute(_model_inputs(value)), value * 2)
    assert len(acl_graph_runtime) == 4
    assert torch.npu.graph_pool_handle.call_count == 2
    first_pool = acl_graph_runtime[0].pool
    second_pool = acl_graph_runtime[2].pool
    assert first_pool is not None
    assert second_pool is not None
    assert first_pool is acl_graph_runtime[1].pool
    assert second_pool is acl_graph_runtime[3].pool
    assert first_pool is not second_pool


def test_graph_distinguishes_optional_forward_arguments(acl_graph_runtime: list[SimpleNamespace]) -> None:
    pipeline = _make_model_pipeline(_ScaleModel())
    inputs = _model_inputs(torch.arange(3))
    for embedding in (None, inputs.input_embedding):
        output = pipeline.execute(replace(inputs, input_embedding=embedding))
        torch.testing.assert_close(output, torch.arange(3) * 2)
    assert len(acl_graph_runtime) == 2


def test_graph_backend_queries_model_not_metadata_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ScaleModel()
    pipeline = _make_model_pipeline(model)
    builder = Mock()
    builder.can_execute_graph.side_effect = AssertionError("Builder admission must be aggregated by the model")
    graph = pipeline.entry_runner.graph_runner
    gather = Mock(side_effect=AssertionError("Local graph stages must not introduce DP admission collectives"))
    monkeypatch.setattr(distributed, "dp_all_true", gather, raising=False)
    graph.bind_execution_metadata_builders((builder,))
    inputs = _model_inputs(torch.ones(3, 2))
    assert graph.can_execute(inputs)
    builder.can_execute_graph.assert_not_called()
    gather.assert_not_called()


def test_model_admission_does_not_hide_invalid_persistent_metadata(acl_graph_runtime: list[SimpleNamespace]) -> None:
    pipeline = _make_model_pipeline(_ScaleModel())
    pipeline.entry_runner.graph_runner.bind_execution_metadata_builders((ModelSamplingMetadataBuilder(),))
    with pytest.raises(ValueError, match="requires sampling input"):
        pipeline.execute(_model_inputs(torch.ones(3, 2)))
    assert pipeline.entry_runner.execution_mode == "graph"
    assert acl_graph_runtime == []


def test_graph_backend_owns_multimodal_admission() -> None:
    pipeline = _make_model_pipeline(_ScaleModel())
    graph = pipeline.entry_runner.graph_runner
    inputs = _model_inputs(torch.ones(3, 2))
    assert not graph.can_execute(replace(inputs, multimodal_inputs=SimpleNamespace()))


def test_runner_consumes_complete_backend_admission() -> None:
    runner = Runner(_ScaleModel(), (), RunnerConfig(torch.device("cpu")))
    runner.graph_runner = Mock()
    inputs = replace(_model_inputs(torch.ones(3, 2)), multimodal_inputs=SimpleNamespace(), eplb=SimpleNamespace())
    runner.graph_runner.can_execute.return_value = True
    assert runner.execute(inputs) is runner.graph_runner.execute.return_value
    runner.graph_runner.can_execute.assert_called_once_with(inputs)


@pytest.mark.parametrize("stage", ["backbone", "lm_head", "proposal"])
def test_dspark_models_aggregate_their_own_modules(stage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _DraftModel().model
    pipeline = model.create_execution_pipeline((ModelSamplingMetadataBuilder(),), RunnerConfig(torch.device("cpu")))
    inputs = _inputs(_SamplingInput())
    module = model.markov_head if stage == "proposal" else getattr(model, stage)
    admission = Mock(return_value=False)
    monkeypatch.setattr(module, "can_execute_graph", admission, raising=False)
    runner = pipeline.proposal_runner if stage == "proposal" else pipeline.entry_runner
    assert not runner.model.can_execute_graph(inputs)
    admission.assert_called_once_with(inputs)
    admission.return_value = True
    assert runner.model.can_execute_graph(inputs)


def test_decoder_preserves_backbone_runtime_hooks() -> None:
    model = _DraftModel().model
    model.backbone.cfg = SimpleNamespace()
    model.backbone._comm_policy = SimpleNamespace(graph_token_limit=8)
    model.backbone.restores_aux_hidden_under_cp = True
    model.backbone.configure_engram_graph_capacity = Mock()
    model.backbone.stage_engram_rows_outside_graph = Mock()
    decoder = DSparkDecoderModel(model)
    for name in (
        "cfg",
        "_comm_policy",
        "restores_aux_hidden_under_cp",
        "configure_engram_graph_capacity",
        "stage_engram_rows_outside_graph",
    ):
        assert getattr(decoder, name) is getattr(model.backbone, name)
    assert getattr(decoder, "missing_hook", None) is None
    assert list(decoder.parameters()) == []


@pytest.mark.parametrize("tuple_output", [False, True])
def test_decoder_returns_logits_and_hidden_without_sampling(
    tuple_output: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _DraftModel().model
    inputs = _inputs(_SamplingInput())
    hidden = torch.ones(inputs.input_ids.numel(), 2)
    backbone_output = (hidden, hidden + 1) if tuple_output else hidden
    backbone = Mock(return_value=backbone_output)
    monkeypatch.setattr(model.backbone, "forward", backbone)
    logits, actual_hidden = DSparkDecoderModel(model)(inputs.input_ids, inputs.positions)
    backbone.assert_called_once_with(inputs.input_ids, inputs.positions, None)
    torch.testing.assert_close(logits, model.compute_logits(hidden, None))
    assert actual_hidden is hidden
    assert model.markov_head.calls == ["lm_head", "lm_head"]


def test_model_without_graph_hook_has_no_extra_computation_constraints() -> None:
    inputs = _model_inputs(torch.ones(3, 2))
    model = torch.nn.Identity()
    assert can_execute_graph(model, inputs)
    pipeline = _make_model_pipeline(model, max_requests=2)
    assert not pipeline.entry_runner.graph_runner.can_execute(inputs)


def test_graph_backend_passes_complete_stage_input_to_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ScaleModel()
    admission = Mock(return_value=True)
    monkeypatch.setattr(model, "can_execute_graph", admission)
    inputs = _model_inputs(torch.ones(3, 2))
    graph = _make_model_pipeline(model).entry_runner.graph_runner
    assert graph.can_execute(inputs)
    admission.assert_called_once_with(inputs)


def test_graph_backend_validates_actual_position_argument() -> None:
    inputs = _model_inputs(torch.ones(3, 2))
    graph = _make_model_pipeline(_ScaleModel()).entry_runner.graph_runner
    assert graph.can_execute(inputs)
    assert not graph.can_execute(replace(inputs, positions=torch.arange(2)))


def test_runner_checks_model_admission_once_before_execution(
    acl_graph_runtime: list[SimpleNamespace], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _ScaleModel()
    admission = Mock(return_value=True)
    monkeypatch.setattr(model, "can_execute_graph", admission)
    pipeline = _make_model_pipeline(model)
    inputs = replace(_model_inputs(torch.ones(3, 2)), layer_synchronizer=Mock())
    torch.testing.assert_close(pipeline.execute(inputs), torch.full((3, 2), 2.0))
    admission.assert_called_once_with(inputs)
    assert pipeline.entry_runner.execution_mode == "graph"


def test_proposal_model_queries_registered_sampling_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _DraftModel().model
    builder = ModelSamplingMetadataBuilder()
    pipeline = model.create_execution_pipeline((builder,), RunnerConfig(torch.device("cpu")))
    inputs = _inputs(_SamplingInput())
    admission = Mock(return_value=False)
    monkeypatch.setattr(builder, "can_execute_graph", admission)
    assert builder in pipeline.proposal_runner.eager_runner.execution_metadata_builders
    assert not pipeline.proposal_runner.model.can_execute_graph(inputs)
    admission.assert_called_once_with(inputs.input_batch.sampling)
    admission.return_value = True
    assert pipeline.proposal_runner.model.can_execute_graph(inputs)


def test_auxiliary_metadata_builder_owns_its_persistent_storage() -> None:
    hidden = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    inputs = _model_inputs(hidden, width=3)
    batch = replace(inputs.input_batch, hidden_states=hidden)
    static_batch = batch.bind_graph_inputs(
        torch.zeros(12, dtype=torch.long),
        torch.zeros(12, dtype=torch.long),
        torch.zeros(12, dtype=torch.bool),
    )
    builder = ProposalMetadataBuilder()
    eager_metadata = builder.build(batch, inputs.metadata)
    assert isinstance(eager_metadata, ProposalMetadata)
    assert eager_metadata.hidden_states is hidden
    persistent = builder.allocate_persistent(static_batch, inputs.metadata)
    address = persistent.hidden_states.data_ptr()
    builder.update_persistent(persistent, batch, inputs.metadata)
    torch.testing.assert_close(persistent.hidden_states[:9], hidden)
    assert torch.count_nonzero(persistent.hidden_states[9:]) == 0
    assert persistent.hidden_states.data_ptr() == address


def test_graph_updates_builder_owned_hidden_states(acl_graph_runtime: list[SimpleNamespace]) -> None:
    runner = Runner(
        _HiddenStatesModel(),
        (ProposalMetadataBuilder(),),
        RunnerConfig(torch.device("cpu"), enable_graph=True, max_seqs_per_batch=4),
    )
    address = None
    for rows in (3, 4, 3):
        hidden = torch.full((rows, 2), float(rows))
        inputs = _model_inputs(torch.zeros_like(hidden))
        inputs = replace(inputs, input_batch=replace(inputs.input_batch, hidden_states=hidden))
        torch.testing.assert_close(runner.execute(inputs), hidden * 2)
        entry = next(iter(runner.graph_runner._graphs.values()))
        persistent = entry.context.execution_contexts[ProposalMetadata].hidden_states
        if address is None:
            address = persistent.data_ptr()
        assert persistent.data_ptr() == address
        assert persistent.data_ptr() != hidden.data_ptr()
        assert torch.count_nonzero(persistent[rows:]) == 0
        assert entry.input_batch.hidden_states is None
    assert len(acl_graph_runtime) == 1
    assert torch.Tensor.record_stream.call_count == 12


def test_graph_separates_hidden_state_signatures(acl_graph_runtime: list[SimpleNamespace]) -> None:
    runner = Runner(
        _HiddenStatesModel(),
        (ProposalMetadataBuilder(),),
        RunnerConfig(torch.device("cpu"), enable_graph=True, max_seqs_per_batch=4),
    )
    inputs = _model_inputs(torch.ones(3, 2))
    for hidden in (None, torch.ones(3, 2), torch.ones(3, 4), torch.ones(3, 2, dtype=torch.float64), None):
        batch = replace(inputs.input_batch, hidden_states=hidden)
        output = runner.execute(replace(inputs, input_batch=batch))
        expected = (inputs.input_embedding if hidden is None else hidden) * 2
        torch.testing.assert_close(output, expected)
    assert len(acl_graph_runtime) == 4


@pytest.mark.parametrize("shape,device", [((), "cpu"), ((2, 2), "cpu"), ((4, 2), "cpu"), ((3, 2), "meta")])
def test_graph_rejects_invalid_hidden_state_layout(shape: tuple[int, ...], device: str) -> None:
    inputs = _model_inputs(torch.ones(3, 2))
    batch = replace(inputs.input_batch, hidden_states=torch.empty(shape, device=device))
    graph = _make_model_pipeline(_ScaleModel()).entry_runner.graph_runner
    assert not graph.can_execute(replace(inputs, input_batch=batch))


def test_graph_separates_tensor_signatures(acl_graph_runtime: list[SimpleNamespace]) -> None:
    pipeline = _make_model_pipeline(_ScaleModel(), max_requests=8)
    for value in (torch.ones(3, 2), torch.ones(3, 4), torch.ones(3, 2, dtype=torch.float64)):
        torch.testing.assert_close(pipeline.execute(_model_inputs(value)), value * 2)
    assert len(acl_graph_runtime) == 3


@pytest.mark.parametrize("rows", [0, 9])
def test_graph_capacity_selects_eager_before_execution(rows: int, acl_graph_runtime: list[SimpleNamespace]) -> None:
    model = _ScaleModel()
    pipeline = _make_model_pipeline(model, max_requests=8)
    value = torch.ones(rows, 2)
    torch.testing.assert_close(pipeline.execute(_model_inputs(value)), value * 2)
    assert model.calls == 1
    assert pipeline.entry_runner.execution_mode == "eager"
    assert acl_graph_runtime == []


def test_model_admission_can_fall_back_then_reuse_graph(acl_graph_runtime: list[SimpleNamespace]) -> None:
    model = _ScaleModel()
    pipeline = _make_model_pipeline(model)
    inputs = _model_inputs(torch.ones(3, 2))
    for allowed in (True, False, True):
        model.allow_graph = allowed
        torch.testing.assert_close(pipeline.execute(inputs), torch.full((3, 2), 2.0))
        assert pipeline.entry_runner.execution_mode == ("graph" if allowed else "eager")
    assert len(acl_graph_runtime) == 1


def test_disabled_graph_reuses_the_existing_eager_runner(acl_graph_runtime: list[SimpleNamespace]) -> None:
    model = _ScaleModel()
    pipeline = _make_model_pipeline(model, graph_enabled=False)
    pipeline.execute(_model_inputs(torch.ones(2, 2)))
    assert type(pipeline.entry_runner.eager_runner) is EagerRunner
    assert pipeline.entry_runner.graph_runner is None
    assert acl_graph_runtime == []


def test_graph_failure_is_not_retried_as_eager() -> None:
    model = _ScaleModel()
    pipeline = _make_model_pipeline(model, graph_enabled=False)
    graph = Mock()
    graph.can_execute.return_value = True
    graph.execute.side_effect = RuntimeError("capture failed")
    pipeline.entry_runner.graph_runner = graph
    with pytest.raises(RuntimeError, match="capture failed"):
        pipeline.execute(_model_inputs(torch.ones(2, 2)))
    assert model.calls == 0


@pytest.mark.parametrize("failure_stage", ["warmup", "capture"])
def test_failed_capture_does_not_publish_an_entry(
    acl_graph_runtime: list[SimpleNamespace],
    capture_lock_state: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    model = _ScaleModel()
    pipeline = _make_model_pipeline(model)
    failure = Mock(side_effect=RuntimeError(f"{failure_stage} failed"))
    if failure_stage == "warmup":
        monkeypatch.setattr(model, "forward", failure)
    else:
        monkeypatch.setattr(import_module(AclGraphRunner.__module__), "capture_acl_graph", failure)
    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        pipeline.execute(_model_inputs(torch.ones(2, 2)))
    assert pipeline.entry_runner.graph_runner._graphs == {}
    assert model.calls == (3 if failure_stage == "capture" else 0)
    assert get_forward_context_or_none() is None
    assert not capture_lock_state.held
    assert capture_lock_state.acquisitions == capture_lock_state.releases == 1


def test_graph_keeps_complete_query_blocks_and_trims_token_rows(acl_graph_runtime: list[SimpleNamespace]) -> None:
    model = _ScaleModel()
    pipeline = _make_model_pipeline(model)
    value = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    torch.testing.assert_close(pipeline.execute(_model_inputs(value, width=3)), value * 2)
    graph = pipeline.entry_runner.graph_runner
    entry = next(iter(graph._graphs.values()))
    assert graph.last_padded_batch_size == 12
    assert entry.input_embedding.shape == (12, 2)


def _make_pipeline(model: _DraftModel, *, graph_enabled: bool = True) -> DSparkExecutionPipeline:
    config = RunnerConfig(torch.device("cpu"), enable_graph=graph_enabled, max_seqs_per_batch=8)
    pipeline = create_execution_pipeline(model.model, (ModelSamplingMetadataBuilder(),), config)
    assert isinstance(pipeline, DSparkExecutionPipeline)
    eager = Mock()
    graph = Mock()
    graph.can_execute.return_value = graph_enabled

    def forward(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: Any,
        input_embedding: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return pipeline.entry_runner.model(input_ids, positions, input_embedding)

    eager.execute.side_effect = forward
    graph.execute.side_effect = forward
    graph.last_padded_batch_size = 24
    pipeline.entry_runner.eager_runner = eager
    pipeline.entry_runner.graph_runner = graph
    return pipeline


def _inputs(sampling: _SamplingInput, *, rows: int = 2) -> ModelInput:
    token_ids = torch.zeros(rows, 3, dtype=torch.long)
    token_ids[:, 0] = torch.arange(rows) + 1
    token_ids = token_ids.flatten()
    return _pipeline_inputs(token_ids, torch.arange(token_ids.numel()), sampling)


def test_decoder_shares_one_capture_while_sampling_falls_back(
    acl_graph_runtime: list[SimpleNamespace],
) -> None:
    model = _DraftModel()
    pipeline = model.model.create_execution_pipeline(
        (ModelSamplingMetadataBuilder(),),
        RunnerConfig(torch.device("cpu"), enable_graph=True, max_seqs_per_batch=8),
    )
    captured_contexts: dict[torch.nn.Module, list[Any]] = {model.model.backbone: [], model.model.lm_head: []}

    def record_capture(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        context = get_forward_context_or_none()
        if context is not None and context.acl_graph is not None:
            captured_contexts[module].append(context.acl_graph)

    for module in captured_contexts:
        module.register_forward_pre_hook(record_capture)
    for sampling in (_SamplingInput(), _SamplingInput(return_probs=True, return_confidence=True), _SamplingInput()):
        inputs = _inputs(sampling, rows=3)
        actual = pipeline.execute(inputs)
        expected = _reference(model, inputs.input_ids, inputs.positions, 3, sampling)
        torch.testing.assert_close(actual.draft_proposal.token_ids, expected.draft_proposal.token_ids)
        if sampling.return_probs:
            torch.testing.assert_close(actual.draft_proposal.draft_probs, expected.draft_proposal.draft_probs)
            torch.testing.assert_close(actual.draft_confidence_probs, expected.draft_confidence_probs)
        assert pipeline.entry_runner.execution_mode == "graph"
        assert pipeline.proposal_runner.execution_mode == ("eager" if sampling.return_probs else "graph")
    assert len(pipeline.runners) == 2
    assert len(acl_graph_runtime) == 2
    assert len(pipeline.entry_runner.graph_runner._graphs) == 1
    assert len(pipeline.proposal_runner.graph_runner._graphs) == 1
    logits, hidden = acl_graph_runtime[0].output
    assert logits.shape == (12, 7)
    assert hidden.shape == (12, 2)
    backbone_contexts = captured_contexts[model.model.backbone]
    head_contexts = captured_contexts[model.model.lm_head]
    assert backbone_contexts
    assert len(backbone_contexts) == len(head_contexts)
    assert all(context is backbone_contexts[0] for context in backbone_contexts + head_contexts)


@pytest.mark.parametrize("reduced_vocab", [False, True])
def test_sampling_fallback_does_not_disable_other_models_graph(
    reduced_vocab: bool, acl_graph_runtime: list[SimpleNamespace]
) -> None:
    model = _DraftModel(vocab_size=3 if reduced_vocab else 7)
    if reduced_vocab:
        model.model.hot_token_ids = torch.tensor([1, 4, 6])
    pipeline = _make_pipeline(model)
    assert isinstance(pipeline.entry_runner.model, DSparkDecoderModel)
    assert pipeline.entry_runner.metadata_builders == ()
    requests = [
        _SamplingInput(),
        _SamplingInput(is_unconstrained_greedy=False),
        _SamplingInput(is_unconstrained_greedy=False, parameters=_SamplingParameters(all_greedy_sample=False)),
        _SamplingInput(return_probs=True),
        _SamplingInput(return_confidence=True),
        _SamplingInput(),
    ]
    for sampling in requests:
        inputs = _inputs(sampling)
        actual = pipeline.execute(inputs)
        expected = _reference(model, inputs.input_ids, inputs.positions, 3, sampling)
        torch.testing.assert_close(actual.draft_proposal.token_ids, expected.draft_proposal.token_ids)
        if sampling.return_probs:
            torch.testing.assert_close(actual.draft_proposal.draft_probs, expected.draft_proposal.draft_probs)
        else:
            assert actual.draft_proposal.draft_probs is None
        if sampling.return_confidence:
            torch.testing.assert_close(actual.draft_confidence_probs, expected.draft_confidence_probs)
        else:
            assert actual.draft_confidence_probs is None
        assert pipeline.entry_runner.execution_mode == "graph"
        greedy_graph = sampling.is_unconstrained_greedy and not sampling.return_probs and not sampling.return_confidence
        assert pipeline.proposal_runner.execution_mode == ("graph" if greedy_graph else "eager")
    pipeline.entry_runner.eager_runner.execute.assert_not_called()
    assert pipeline.entry_runner.graph_runner.execute.call_count == len(requests)
    assert len(pipeline.proposal_runner.graph_runner._graphs) == 1
    assert pipeline.entry_runner.last_graph_rows == 24
    assert pipeline.proposal_runner.last_graph_rows == 6


def test_dspark_execution_with_graph_disabled_matches_reference(acl_graph_runtime: list[SimpleNamespace]) -> None:
    model = _DraftModel()
    pipeline = _make_pipeline(model, graph_enabled=False)
    sampling = _SamplingInput(return_probs=True, return_confidence=True)
    inputs = _inputs(sampling)
    actual = pipeline.execute(inputs)
    expected = _reference(model, inputs.input_ids, inputs.positions, 3, sampling)
    torch.testing.assert_close(actual.draft_proposal.token_ids, expected.draft_proposal.token_ids)
    torch.testing.assert_close(actual.draft_proposal.draft_probs, expected.draft_proposal.draft_probs)
    torch.testing.assert_close(actual.draft_confidence_probs, expected.draft_confidence_probs)
    assert pipeline.proposal_runner.graph_runner is None
    assert acl_graph_runtime == []


def test_pipeline_checks_sampling_before_running_any_model() -> None:
    pipeline = _make_pipeline(_DraftModel(), graph_enabled=False)
    inputs = _inputs(_SamplingInput())
    inputs.input_batch = replace(inputs.input_batch, sampling=None)
    with pytest.raises(RuntimeError, match="requires sampling metadata"):
        pipeline.execute(inputs)
    pipeline.entry_runner.graph_runner.can_execute.assert_not_called()
    pipeline.entry_runner.eager_runner.execute.assert_not_called()


@pytest.mark.parametrize("graph_enabled", [False, True])
def test_runner_trims_dummy_request_rows(graph_enabled: bool, acl_graph_runtime: list[SimpleNamespace]) -> None:
    pipeline = _make_pipeline(_DraftModel(), graph_enabled=graph_enabled)
    inputs = _inputs(_SamplingInput(), rows=1)
    inputs.input_batch = replace(inputs.input_batch, num_reqs=0, num_tokens=0, is_dummy=True)
    result = pipeline.execute(inputs)
    assert result.draft_proposal.token_ids.shape == (0, 3)


def test_decoder_model_uses_weights_shared_after_pipeline_creation(acl_graph_runtime: list[SimpleNamespace]) -> None:
    model = _DraftModel()
    lm_head = model.model.lm_head
    model.model.lm_head = None
    pipeline = _make_pipeline(model)
    inputs = _inputs(_SamplingInput())
    assert not pipeline.entry_runner.model.can_execute_graph(inputs)
    model.model.lm_head = lm_head
    assert pipeline.entry_runner.model.can_execute_graph(inputs)
    result = pipeline.execute(inputs)
    expected = _reference(model, inputs.input_ids, inputs.positions, 3, inputs.input_batch.sampling)
    torch.testing.assert_close(result.draft_proposal.token_ids, expected.draft_proposal.token_ids)
    assert list(pipeline.entry_runner.model.parameters()) == []
    assert list(pipeline.proposal_runner.model.parameters()) == []


def test_models_share_forward_signature_and_do_not_leak_context() -> None:
    model = _DraftModel()
    pipeline = model.model.create_execution_pipeline(
        (ModelSamplingMetadataBuilder(),), RunnerConfig(torch.device("cpu"))
    )
    expected_parameters = tuple(signature(model.model.backbone.forward).parameters)
    for runner in pipeline.runners:
        assert isinstance(runner.model, torch.nn.Module)
        assert tuple(signature(runner.model.forward).parameters) == expected_parameters
        assert type(runner.eager_runner) is EagerRunner
    pipeline.execute(_inputs(_SamplingInput()))
    assert get_forward_context_or_none() is None


def test_pipeline_passes_features_explicitly_and_keeps_layer_runtime_on_backbone() -> None:
    pipeline = _DraftModel().model.create_execution_pipeline(
        (ModelSamplingMetadataBuilder(),), RunnerConfig(torch.device("cpu"))
    )
    inputs = _inputs(_SamplingInput(return_confidence=True))
    inputs.layer_synchronizer = Mock()
    inputs.layer_load_context = Mock()
    inputs.eplb = SimpleNamespace(expert_load_data=torch.zeros(1))
    forwarded: dict[torch.nn.Module, torch.Tensor] = {}
    produced: dict[torch.nn.Module, tuple[torch.Tensor, torch.Tensor]] = {}

    def record_input(model: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        forwarded[model] = args[2]
        context = get_forward_context()
        assert context.layer_synchronizer is None
        assert context.layer_load_context is None
        assert context.eplb is None
        assert get_execution_context(ProposalMetadata).hidden_states is produced[pipeline.entry_runner.model][1]

    def record_output(
        model: torch.nn.Module, args: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        context = get_forward_context()
        assert context.layer_synchronizer is inputs.layer_synchronizer
        assert context.layer_load_context is inputs.layer_load_context
        assert context.eplb is inputs.eplb
        produced[model] = output

    pipeline.entry_runner.model.register_forward_hook(record_output)
    pipeline.proposal_runner.model.register_forward_pre_hook(record_input)
    output = pipeline.execute(inputs)
    assert isinstance(output, ModelOutput)
    assert forwarded[pipeline.proposal_runner.model] is produced[pipeline.entry_runner.model][0]


@pytest.mark.parametrize("tuple_output", [False, True])
def test_plain_model_uses_single_stage_and_preserves_result_identity(tuple_output: bool) -> None:
    model = torch.nn.Identity()
    pipeline = create_execution_pipeline(model, (), RunnerConfig(torch.device("cpu")))
    assert isinstance(pipeline, SingleModelExecutionPipeline)
    assert pipeline.entry_runner.model is model
    hidden = torch.ones(2, 3)
    output = (hidden, hidden + 1) if tuple_output else hidden
    eager = Mock()
    eager.execute.return_value = output
    pipeline.entry_runner.eager_runner = eager
    assert pipeline.execute(_inputs(_SamplingInput())) is output


def test_executor_constructs_and_runs_the_registered_dspark_pipeline(
    acl_graph_runtime: list[SimpleNamespace], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _DraftModel()
    model.model.backbone.attention = _make_attention_layer()
    backend = StubAttentionBackend()
    monkeypatch.setattr(
        import_module(ModelExecutor.__module__), "_create_attention_backend", Mock(return_value=backend)
    )
    graph = Mock()
    graph.can_execute.return_value = True

    def forward(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: Any,
        input_embedding: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return executor.execution_pipeline.entry_runner.model(input_ids, positions, input_embedding)

    graph.execute.side_effect = forward
    graph_factory = Mock(return_value=graph)
    monkeypatch.setattr(
        import_module("xllm.python.model_executor.runners.decode_acl_graph"),
        "DecodeAclGraphRunner",
        graph_factory,
    )
    executor = ModelExecutor(
        model,
        {
            "model_type": "DSparkDraftModel",
            "enable_graph": True,
            "max_position_embeddings": 32,
            "draft_query_width": 3,
        },
        max_seqs_per_batch=4,
        is_spec_draft=True,
    )
    executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
    assert isinstance(executor.execution_pipeline, DSparkExecutionPipeline)
    assert graph_factory.call_args.args[0] is executor.execution_pipeline.entry_runner.model
    assert executor.eager_runner.model is executor.execution_pipeline.entry_runner.model
    assert executor.eager_runner.execution_metadata_builders == ()
    graph.bind_execution_metadata_builders.assert_called_once_with(())
    inputs = _inputs(_SamplingInput(return_probs=True))
    output = executor.execute(
        inputs.input_ids,
        inputs.positions,
        inputs.metadata,
        input_batch_metadata=_batch_metadata(inputs.input_ids.numel(), 3),
        sampling_input=inputs.input_batch.sampling,
    )
    assert isinstance(output, ModelOutput)
    assert output.draft_proposal.token_ids.shape == (2, 3)
    assert output.draft_proposal.draft_probs.shape == (2, 3, 7)
    graph.execute.assert_called_once()
    pipeline = executor.execution_pipeline
    assert pipeline.entry_runner.execution_mode == "graph"
    assert pipeline.proposal_runner.execution_mode == "eager"
