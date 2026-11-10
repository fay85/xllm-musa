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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.python import test_execution_pipeline as pipeline_tests
from tests.python.test_decode_acl_graph import _metadata, _query_batch
from tests.python.test_dspark_pipeline import _batch_metadata, _pipeline_inputs
from tests.python.test_model_executor import StubAttentionBackend, _make_attention_layer
from xllm.python.model_executor import executor as executor_module
from xllm.python.model_executor.execution_pipeline import create_execution_pipeline
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.model_output import DraftCandidates, DraftDecoderOutput, DraftProposal, ModelOutput
from xllm.python.model_executor.runner import Runner, RunnerConfig
from xllm.python.model_executor.runners import base as runner_base
from xllm.python.model_executor.runners.base import BaseRunner
from xllm.python.models.base import PyModelBase
from xllm.python.models.dflash2_model import DFlash2DecoderModel, DFlash2SamplingMetadataBuilder
from xllm.python.models.qwen3_dflash2 import DFlash2CandidateSelector, DFlash2Qwen3ForCausalLM
from xllm.python.pipelines.dflash2_pipeline import DFlash2ExecutionPipeline
from xllm.python.registry import get_execution_metadata_builder_classes

acl_graph_runtime = pipeline_tests.acl_graph_runtime
capture_lock_state = pipeline_tests.capture_lock_state
_npu_platform = pipeline_tests._npu_platform


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(11, 6)
        self.candidate_selector = DFlash2CandidateSelector(6, 11, 3, 4, torch.float32, torch.device("cpu"))

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, input_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids) + positions[:, None].to(self.embed_tokens.weight.dtype) * 0.1


class _Head(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(6, 11, bias=False)
        self.allow_graph = True
        self.rows: list[int] = []

    def can_execute_graph(self, inputs: ModelInput) -> bool:
        return self.allow_graph

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.rows.append(hidden.shape[0])
        return super().forward(hidden)


class _Model(PyModelBase):
    create_execution_pipeline = DFlash2Qwen3ForCausalLM.create_execution_pipeline
    dflash2_candidates = DFlash2Qwen3ForCausalLM.dflash2_candidates

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.model = _Backbone()
        self.lm_head = _Head()
        self.cfg = SimpleNamespace(vocab_size=11)
        self.dtype = dtype
        generator = torch.Generator().manual_seed(17)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)
        self.to(dtype)


def _inputs(modes: list[bool], *, seed: int = 0, temperatures: list[float] | None = None) -> ModelInput:
    requests = len(modes)
    generator = torch.Generator().manual_seed(seed)
    do_sample = torch.tensor(modes)
    noise = -torch.log(-torch.log(torch.rand((requests, 3, 4), generator=generator)))
    noise = torch.where(do_sample[:, None, None], noise, torch.zeros_like(noise))
    sampling = SimpleNamespace(
        parameters=SimpleNamespace(
            all_greedy_sample=not any(modes),
            all_random_sample=all(modes),
            do_sample=do_sample,
            temperatures=None if temperatures is None else torch.tensor(temperatures),
            frequency_penalties=torch.ones(requests),
            filter_mask=torch.zeros(requests, 11),
        ),
        gumbel_noise=noise,
        return_probs=True,
        return_confidence=False,
    )
    tokens = torch.arange(requests * 4) % 11
    inputs = _pipeline_inputs(tokens, torch.arange(tokens.numel()), sampling, width=4)
    inputs.metadata.use_dflash_proposal_attention = True
    return inputs


def _reference(model: _Model, inputs: ModelInput) -> DraftProposal:
    requests = inputs.input_batch.num_reqs
    hidden = model.model(inputs.input_ids, inputs.positions).view(requests, 4, -1)[:, 1:]
    logits = model.compute_logits(hidden.reshape(requests * 3, -1), None).view(requests, 3, -1)
    candidate_ids, edges = model.dflash2_candidates(hidden, logits, inputs.input_ids.view(requests, 4)[:, 0])
    sampling = inputs.input_batch.sampling
    tokens = torch.empty((requests, 3), dtype=torch.long)
    dense_probs = torch.zeros(requests, 3, 11)
    for request in range(requests):
        previous = 0
        temperature = (
            1.0
            if sampling.parameters.temperatures is None
            else float(sampling.parameters.temperatures[request].to(logits.dtype).float())
        )
        temperature = temperature or 1.0
        for step in range(3):
            log_probs = torch.log_softmax(edges[request, step, previous] / temperature, dim=-1)
            chosen = int((log_probs + sampling.gumbel_noise[request, step]).argmax())
            tokens[request, step] = candidate_ids[request, step, chosen]
            if bool(sampling.parameters.do_sample[request]):
                dense_probs[request, step, candidate_ids[request, step]] = log_probs.exp()
            else:
                dense_probs[request, step, tokens[request, step]] = 1.0
            previous = chosen
    return DraftProposal(tokens, dense_probs)


def _pipeline(model: _Model, *, graph: bool = False, dp_size: int = 1) -> DFlash2ExecutionPipeline:
    pipeline = create_execution_pipeline(
        model,
        (DFlash2SamplingMetadataBuilder(4, 4),),
        RunnerConfig(torch.device("cpu"), {"dp_size": dp_size}, enable_graph=graph, max_seqs_per_batch=4),
    )
    assert isinstance(pipeline, DFlash2ExecutionPipeline)
    assert len(pipeline.runners) == 2
    return pipeline


def _sampling_inputs(modes: list[bool], *, temperatures: list[float] | None = None) -> ModelInput:
    original = _inputs(modes, temperatures=temperatures)
    rows = len(modes) * 3
    inputs = _pipeline_inputs(
        torch.zeros(rows, dtype=torch.long), torch.arange(rows), original.input_batch.sampling, width=3
    )
    candidates = torch.arange(rows * 4).view(rows, 4) % 11
    return replace(inputs, input_batch=replace(inputs.input_batch, hidden_states=candidates))


@pytest.mark.parametrize("modes", [[False, False, False], [True, True, True], [False, True, False]])
@pytest.mark.parametrize("temperatures", [None, [0.0, 0.7, 1.3]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pipeline_matches_selector_reference(
    modes: list[bool], temperatures: list[float] | None, dtype: torch.dtype
) -> None:
    model = _Model(dtype)
    inputs = _inputs(modes, temperatures=temperatures)
    with torch.no_grad():
        expected = _reference(model, inputs)
        model.lm_head.rows.clear()
        output = _pipeline(model).execute(inputs)
    assert isinstance(output, ModelOutput)
    torch.testing.assert_close(output.draft_proposal.token_ids, expected.token_ids)
    torch.testing.assert_close(output.draft_proposal.draft_probs, expected.draft_probs)
    assert model.lm_head.rows == [9]
    assert output.hidden_states is None


def test_sampling_falls_back_and_recovers_without_disabling_decoder_graph(
    acl_graph_runtime: list[SimpleNamespace],
) -> None:
    model = _Model()
    pipeline = _pipeline(model, graph=True)
    cases = (
        ([False] * 4, None, "graph"),
        ([True] * 3, None, "eager"),
        ([False, True, True, False], None, "eager"),
        ([False] * 3, [0.7, 0.8, 0.9], "eager"),
        ([False] * 3, None, "graph"),
    )
    with torch.no_grad():
        for seed, (modes, temperatures, mode) in enumerate(cases):
            inputs = _inputs(modes, seed=seed, temperatures=temperatures)
            expected = _reference(model, inputs)
            output = pipeline.execute(inputs)
            torch.testing.assert_close(output.draft_proposal.token_ids, expected.token_ids)
            torch.testing.assert_close(output.draft_proposal.draft_probs, expected.draft_probs)
            assert output.draft_proposal.token_ids.shape == (len(modes), 3)
            assert pipeline.entry_runner.execution_mode == "graph"
            assert pipeline.proposal_runner.execution_mode == mode
    assert len(acl_graph_runtime) == 2
    assert acl_graph_runtime[0].output.candidates is not None


@pytest.mark.parametrize(
    "modes,agreed,expected_mode",
    [([False, False], False, "eager"), ([True, False], False, "eager"), ([False], True, "graph")],
)
def test_dp_sampling_agreement_does_not_disable_decoder_graph(
    acl_graph_runtime: list[SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    modes: list[bool],
    agreed: bool,
    expected_mode: str,
) -> None:
    model = _Model()
    pipeline = _pipeline(model, graph=True, dp_size=2)
    vote = Mock(return_value=agreed)
    monkeypatch.setattr(runner_base.distributed, "dp_all_true", vote, raising=False)
    inputs = _inputs(modes)
    with torch.no_grad():
        expected = _reference(model, inputs)
        output = pipeline.execute(inputs)
    torch.testing.assert_close(output.draft_proposal.token_ids, expected.token_ids)
    torch.testing.assert_close(output.draft_proposal.draft_probs, expected.draft_probs)
    vote.assert_called_once_with(not any(modes), torch.device("cpu"), 2)
    assert pipeline.entry_runner.execution_mode == "graph"
    assert pipeline.proposal_runner.execution_mode == expected_mode


@pytest.mark.parametrize("agreed", [True, False])
def test_empty_dp_sampling_joins_agreement_without_launching(
    acl_graph_runtime: list[SimpleNamespace], monkeypatch: pytest.MonkeyPatch, agreed: bool
) -> None:
    model = _Model()
    pipeline = _pipeline(model, graph=True, dp_size=2)
    inputs = _inputs([False])
    inputs = replace(inputs, input_batch=replace(inputs.input_batch, num_reqs=0, sampling=None, is_dummy=True))
    vote = Mock(return_value=agreed)
    monkeypatch.setattr(runner_base.distributed, "dp_all_true", vote, raising=False)
    graph = pipeline.proposal_runner.graph_runner
    monkeypatch.setattr(graph, "can_execute", Mock(side_effect=AssertionError("Idle sampling has no graph inputs")))
    monkeypatch.setattr(graph, "execute", Mock(side_effect=AssertionError("Idle sampling must not replay")))
    monkeypatch.setattr(
        pipeline.proposal_runner.eager_runner, "execute", Mock(side_effect=AssertionError("Idle sampling must not run"))
    )
    with torch.no_grad():
        output = pipeline.execute(inputs)
    assert isinstance(output, torch.Tensor)
    vote.assert_called_once_with(True, torch.device("cpu"), 2)
    assert pipeline.proposal_runner.execution_mode is None


@pytest.mark.parametrize("component", ["backbone", "head", "selector"])
def test_decoder_components_share_graph_admission_without_disabling_sampling(
    acl_graph_runtime: list[SimpleNamespace], monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    model = _Model()
    pipeline = _pipeline(model, graph=True)
    module = {"backbone": model.model, "head": model.lm_head, "selector": model.model.candidate_selector}[component]
    with torch.no_grad():
        for allowed in (True, False, True):
            monkeypatch.setattr(module, "can_execute_graph", lambda inputs, allowed=allowed: allowed, raising=False)
            inputs = _inputs([False, False], seed=9)
            expected = _reference(model, inputs)
            output = pipeline.execute(inputs)
            torch.testing.assert_close(output.draft_proposal.draft_probs, expected.draft_probs)
            assert pipeline.entry_runner.execution_mode == ("graph" if allowed else "eager")
            assert pipeline.proposal_runner.execution_mode == "graph"
    assert len(acl_graph_runtime) == 2


@pytest.mark.parametrize("dummy", [True, False])
def test_backbone_only_execution_without_proposal(dummy: bool) -> None:
    model = _Model()
    pipeline = _pipeline(model)
    inputs = _inputs([False])
    inputs = replace(inputs, input_batch=replace(inputs.input_batch, sampling=None, is_dummy=dummy))
    with torch.no_grad():
        output = pipeline.execute(inputs)
        torch.testing.assert_close(output, model.model(inputs.input_ids, inputs.positions))
    assert pipeline.proposal_runner.execution_mode is None
    assert model.lm_head.rows == []


def test_dp_draft_warmup_uses_eager_without_relaxing_target_row_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _Model()
    runner = Runner(
        DFlash2DecoderModel(model, 8),
        (),
        RunnerConfig(
            torch.device("cpu"),
            {"dp_size": 2, "draft_query_width": 8, "max_position_embeddings": 256},
            enable_graph=True,
            max_seqs_per_batch=128,
            is_spec_draft=True,
        ),
        attention_backend=SimpleNamespace(page_size=128, is_mla=False),
    )
    metadata = _metadata(None)
    metadata.is_chunked_prefill = True
    metadata.use_dflash_proposal_attention = False
    metadata.block_table = torch.zeros((16, 2), dtype=torch.int32)
    metadata.kv_seq_lens = torch.full((16,), 64, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [64] * 16
    metadata.slot_mapping = torch.arange(128, dtype=torch.int32)
    metadata.dp_execution_token_counts = (128, 128)
    metadata.dp_is_decode = (True, True)
    tokens = torch.zeros(128, dtype=torch.int32)
    inputs = ModelInput(tokens, tokens, metadata, input_batch=_query_batch(16, 8))
    with monkeypatch.context() as target_context:
        target_context.setattr(runner.graph_runner, "_is_spec_draft", False)
        with pytest.raises(RuntimeError, match="row layout"):
            runner.graph_runner._can_execute_inner(tokens, metadata)
    assert not runner.graph_runner._can_execute_inner(tokens, metadata)
    eager = Mock(return_value=tokens)
    vote = Mock(side_effect=lambda allowed, device, size: allowed)
    monkeypatch.setattr(runner.eager_runner, "execute", eager)
    monkeypatch.setattr(runner_base.distributed, "dp_all_true", vote, raising=False)
    assert runner.execute(inputs) is tokens
    assert runner.execution_mode == "eager"
    vote.assert_called_once_with(False, torch.device("cpu"), 2)
    eager.assert_called_once()


@pytest.mark.parametrize("num_reqs", [0, 2])
def test_expanded_draft_dp_admission_preserves_empty_and_busy_shards(
    monkeypatch: pytest.MonkeyPatch, num_reqs: int
) -> None:
    model = _Model()
    width = 8
    runner = Runner(
        DFlash2DecoderModel(model, width),
        (),
        RunnerConfig(
            torch.device("cpu"),
            {"dp_size": 2, "dp_rank": 0 if num_reqs else 1, "draft_query_width": width, "max_position_embeddings": 256},
            enable_graph=True,
            max_seqs_per_batch=16,
            is_spec_draft=True,
        ),
        attention_backend=SimpleNamespace(page_size=128, is_mla=False),
    )
    batch = _query_batch(num_reqs, width)
    rows = batch.num_tokens if num_reqs else 1
    metadata = _metadata(None if num_reqs else torch.zeros(1, dtype=torch.int32))
    metadata.is_chunked_prefill = True
    metadata.use_dflash_proposal_attention = True
    metadata.is_dummy = batch.is_dummy
    metadata.num_accepted_tokens = None
    metadata.block_table = torch.zeros((rows, width), dtype=torch.int32)
    metadata.kv_seq_lens = torch.full((rows,), width, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [width] * rows
    metadata.q_cu_seq_lens = torch.arange(rows + 1, dtype=torch.int32) * (1 if num_reqs else width)
    metadata.kv_cu_seq_lens = torch.arange(rows + 1, dtype=torch.int32) * width
    metadata.slot_mapping = torch.full((batch.num_tokens,), -1, dtype=torch.int32)
    metadata.dp_execution_token_counts = (2 * width, width)
    metadata.raw_dp_execution_token_counts = (2 * width, 0)
    metadata.dp_is_decode = (True, True)
    vote = Mock(side_effect=lambda allowed, device, size: allowed)
    monkeypatch.setattr(runner_base.distributed, "dp_all_true", vote, raising=False)
    graph = runner.graph_runner
    monkeypatch.setattr(graph, "warmup", Mock())
    monkeypatch.setattr(graph, "execute", Mock(return_value=batch.input_ids))
    monkeypatch.setattr(
        runner.eager_runner, "execute", Mock(side_effect=AssertionError("Expanded draft should use graph"))
    )
    inputs = ModelInput(batch.input_ids, batch.input_ids, metadata, input_batch=batch)
    assert runner.execute(inputs) is batch.input_ids
    assert runner.execution_mode == "graph"
    vote.assert_called_once_with(True, torch.device("cpu"), 2)


def test_warmup_without_sampling_captures_decoder_for_later_proposals(
    acl_graph_runtime: list[SimpleNamespace],
) -> None:
    model = _Model()
    pipeline = _pipeline(model, graph=True)
    warmup = _inputs([False] * 4)
    warmup = replace(warmup, input_batch=replace(warmup.input_batch, sampling=None))
    with torch.no_grad():
        output = pipeline.execute(warmup)
        torch.testing.assert_close(output, model.model(warmup.input_ids, warmup.positions))
        assert pipeline.entry_runner.execution_mode == "graph"
        assert pipeline.proposal_runner.execution_mode is None
        assert len(acl_graph_runtime) == 1
        assert acl_graph_runtime[0].output.candidates is not None
        for modes in ([True, False, True], [False] * 3, [False]):
            inputs = _inputs(modes, seed=19)
            expected = _reference(model, inputs)
            proposal = pipeline.execute(inputs).draft_proposal
            torch.testing.assert_close(proposal.token_ids, expected.token_ids)
            torch.testing.assert_close(proposal.draft_probs, expected.draft_probs)
            assert proposal.token_ids.shape == (len(modes), 3)
    assert len(acl_graph_runtime) == 4


def test_decoder_graph_skips_anchor_logits_and_shares_selected_hidden(
    acl_graph_runtime: list[SimpleNamespace], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _Model()
    pipeline = _pipeline(model, graph=True)
    inputs = _inputs([False] * 3)
    original_candidates = model.dflash2_candidates
    head_inputs: list[torch.Tensor] = []
    model.lm_head.register_forward_pre_hook(lambda module, args: head_inputs.append(args[0]))

    def candidates(
        hidden: torch.Tensor, logits: torch.Tensor, anchor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden.is_contiguous()
        assert hidden.data_ptr() == head_inputs[-1].data_ptr()
        return original_candidates(hidden, logits, anchor)

    monkeypatch.setattr(model, "dflash2_candidates", candidates)
    with torch.no_grad():
        output = pipeline.execute(inputs)
    assert len(acl_graph_runtime) == 2
    assert set(model.lm_head.rows) == {12}
    assert output.draft_proposal.token_ids.shape == (3, 3)


@pytest.mark.parametrize("clone", [False, True])
def test_decoder_output_slices_hidden_by_tokens_and_candidates_by_requests(clone: bool) -> None:
    hidden = torch.randn(16, 6)
    candidates = DraftCandidates(torch.arange(48).view(4, 3, 4), torch.randn(4, 3, 4, 4))
    output = DraftDecoderOutput(hidden_states=hidden, candidates=candidates)
    trimmed = BaseRunner._slice_output(output, 12, num_reqs=3, clone=clone)
    torch.testing.assert_close(trimmed.hidden_states, hidden[:12])
    torch.testing.assert_close(trimmed.candidates.token_ids, candidates.token_ids[:3])
    torch.testing.assert_close(trimmed.candidates.edge_logits, candidates.edge_logits[:3])
    assert (trimmed.hidden_states.data_ptr() != hidden.data_ptr()) == clone
    assert (trimmed.candidates.token_ids.data_ptr() != candidates.token_ids.data_ptr()) == clone
    assert (trimmed.candidates.edge_logits.data_ptr() != candidates.edge_logits.data_ptr()) == clone
    with pytest.raises(ValueError, match="request-scoped"):
        BaseRunner._slice_output(output, 12)


def test_builder_keeps_addresses_and_resets_padding() -> None:
    builder = DFlash2SamplingMetadataBuilder(4, 4)
    inputs = _sampling_inputs([False] * 3)
    batch = replace(inputs.input_batch, num_tokens_after_padding=12)
    persistent = builder.allocate_persistent(batch, inputs.metadata)
    addresses = [value.data_ptr() for value in (persistent.noise, persistent.candidate_ids)]
    assert persistent.temperatures is None
    assert persistent.do_sample is None
    for modes in ([False] * 3, [False], [False] * 2):
        current = _sampling_inputs(modes)
        builder.update_persistent(persistent, current.input_batch, current.metadata)
        torch.testing.assert_close(persistent.noise[: len(modes)], current.input_batch.sampling.gumbel_noise)
        torch.testing.assert_close(persistent.candidate_ids[: len(modes) * 3], current.input_batch.hidden_states)
        assert torch.count_nonzero(persistent.noise[len(modes) :]) == 0
        assert torch.count_nonzero(persistent.candidate_ids[len(modes) * 3 :]) == 0
        assert addresses == [value.data_ptr() for value in (persistent.noise, persistent.candidate_ids)]


@pytest.mark.parametrize("modes", [[False, False], [True, True], [False, True]])
@pytest.mark.parametrize("temperatures", [None, [0.7, 1.0]])
def test_builder_preserves_dynamic_sampling_controls(modes: list[bool], temperatures: list[float] | None) -> None:
    builder = DFlash2SamplingMetadataBuilder(4, 4)
    inputs = _sampling_inputs(modes, temperatures=temperatures)
    metadata = builder.build(inputs.input_batch, inputs.metadata)
    assert metadata.all_greedy_sample == (not any(modes))
    assert metadata.all_random_sample == all(modes)
    assert (metadata.temperatures is None) == (temperatures is None)
    assert (metadata.do_sample is None) == (not any(modes) or all(modes))
    assert builder.can_execute_graph(inputs.input_batch) == (not any(modes) and temperatures is None)


def test_registered_builder_and_invalid_noise() -> None:
    assert get_execution_metadata_builder_classes("DFlash2DraftModel") == (DFlash2SamplingMetadataBuilder,)
    builder = DFlash2SamplingMetadataBuilder.from_config({"draft_query_width": 4, "dflash2_selector_top_k": 4})
    inputs = _sampling_inputs([False])
    inputs.input_batch.sampling.gumbel_noise = torch.zeros(1, 2, 4)
    assert not builder.can_execute_graph(inputs.input_batch)
    with pytest.raises(ValueError, match="per-request selector inputs"):
        builder.build(inputs.input_batch, inputs.metadata)


@pytest.mark.parametrize(
    ("modes", "expected_exp", "expected_one_hot", "expected_where"),
    [([False, False], 0, 3, 0), ([True, True], 3, 0, 0), ([False, True], 3, 3, 3)],
)
@pytest.mark.parametrize("temperatures", [None, [0.7, 1.0]])
def test_sampling_skips_unused_computation(
    modes: list[bool],
    expected_exp: int,
    expected_one_hot: int,
    expected_where: int,
    temperatures: list[float] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    inputs = _inputs(modes, temperatures=temperatures)
    counts = dict(exp=0, one_hot=0, where=0, temperature=0)
    original_exp = torch.Tensor.exp
    original_scatter = torch.Tensor.scatter_
    original_where = torch.where
    original_div = torch.Tensor.div_

    def counted_exp(tensor: torch.Tensor) -> torch.Tensor:
        counts["exp"] += 1
        return original_exp(tensor)

    def counted_scatter(
        tensor: torch.Tensor, dim: int, index: torch.Tensor, source: torch.Tensor | float
    ) -> torch.Tensor:
        counts["one_hot"] += int(isinstance(source, float))
        return original_scatter(tensor, dim, index, source)

    def counted_where(condition: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        counts["where"] += 1
        return original_where(condition, left, right)

    def counted_div(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        counts["temperature"] += 1
        return original_div(tensor, other)

    monkeypatch.setattr(torch.Tensor, "exp", counted_exp)
    monkeypatch.setattr(torch.Tensor, "scatter_", counted_scatter)
    monkeypatch.setattr(torch, "where", counted_where)
    monkeypatch.setattr(torch.Tensor, "div_", counted_div)
    with torch.no_grad():
        _pipeline(model).execute(inputs)
    assert counts == dict(
        exp=expected_exp,
        one_hot=expected_one_hot,
        where=expected_where,
        temperature=int(temperatures is not None),
    )


def test_executor_uses_outer_provider_and_passes_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _Model()
    model.model.attention = _make_attention_layer()
    backend = StubAttentionBackend()
    monkeypatch.setattr(executor_module, "_create_attention_backend", lambda *args, **kwargs: backend)
    executor = ModelExecutor(
        model,
        {"model_type": "DFlash2DraftModel", "draft_query_width": 4, "dflash2_selector_top_k": 4},
        max_seqs_per_batch=4,
        is_spec_draft=True,
    )
    assert isinstance(executor.execution_pipeline, DFlash2ExecutionPipeline)
    executor.bind_kv_caches([(None, None)])
    inputs = _inputs([True, False])
    with torch.no_grad():
        expected = _reference(model, inputs)
        output = executor.execute(
            inputs.input_ids,
            inputs.positions,
            inputs.metadata,
            input_batch_metadata=_batch_metadata(inputs.input_ids.numel(), 4),
            sampling_input=inputs.input_batch.sampling,
        )
    torch.testing.assert_close(output.draft_proposal.token_ids, expected.token_ids)
    torch.testing.assert_close(output.draft_proposal.draft_probs, expected.draft_probs)
