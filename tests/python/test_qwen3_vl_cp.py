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

"""CPU behavior tests for fused Qwen3-VL text streams under CP.

Only device kernels and communication are replaced.
The eager runner, model forward, fusion, shard/restore and final norm are real.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Condition
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xllm.python import distributed, kernels
from xllm.python.model_executor.cp_utils import build_cp_context, cp_merge_rows, cp_shard_positions, cp_shard_rows
from xllm.python.model_executor.forward_context import ForwardContext, forward_context, get_forward_context
from xllm.python.model_executor.runners import eager
from xllm.python.models.multimodal import MultimodalInputs
from xllm.python.models.qwen3 import Qwen3Config
from xllm.python.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLModel


@pytest.fixture(scope="module", autouse=True)
def _native_cp_builder() -> None:
    """Use the production host indexer through the real torch dispatcher."""
    library_path = os.getenv("XLLM_CP_TEST_OP_LIBRARY")
    if library_path:
        torch.ops.load_library(library_path)
    if not hasattr(torch.ops.xllm_ops, "build_cp_context"):
        pytest.skip("Requires native build_cp_context; set XLLM_CP_TEST_OP_LIBRARY to an operator-only library")


def _metadata(lengths: list[int], *, prefill: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=prefill,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        is_dummy=False,
        kv_split_size=1,
        q_seq_lens_host=torch.tensor(lengths),
        kv_seq_lens_host=torch.tensor(lengths),
    )


class _CpuCollective:
    """Exchange the two concurrently computed rank tensors at each gather."""

    def __init__(self) -> None:
        self.condition = Condition()
        self.rounds = [0, 0]
        self.values: dict[int, dict[int, torch.Tensor]] = {}

    def all_gather(self, value: torch.Tensor, dim: int, size: int, group: str) -> torch.Tensor:
        assert dim == 0 and size == 2 and group == "cp"
        plan = get_forward_context().cp_context
        assert plan is not None
        rank = plan.cp_rank
        with self.condition:
            step = self.rounds[rank]
            self.rounds[rank] += 1
            bucket = self.values.setdefault(step, {})
            bucket[rank] = value.clone()
            self.condition.notify_all()
            assert self.condition.wait_for(lambda: len(bucket) == size, timeout=10), "CP collective timed out"
            return torch.cat([bucket[0], bucket[1]], dim=dim)


class _CausalDecoder(nn.Module):
    """Small causal reference layer sensitive to all position axes and rows."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = get_forward_context()
        plan = context.cp_context
        self.inputs[plan.cp_rank if plan is not None else -1] = (hidden.clone(), positions.clone())
        if positions.dim() == 2:
            assert mrope_section is not None and cos is None and sin is None
            time, height, width = positions.float()
            position_features = torch.stack([time + 2 * height, width - height], dim=-1) * 0.01
        else:
            assert cos is not None and cos.shape[1] == hidden.shape[0]
            position_features = torch.stack([positions, -positions], dim=-1).float() * 0.01
        residual = hidden if residual is None else hidden + residual
        values = residual + position_features
        global_values = cp_merge_rows(values, plan) if plan is not None else values
        # Each request attends its own causal prefix, including peer-owned rows.
        prefixes = [
            sequence.cumsum(0) / torch.arange(1, len(sequence) + 1).unsqueeze(1)
            for sequence in global_values.split(context.metadata.q_seq_lens_host.tolist())
        ]
        attended = torch.cat(prefixes)
        if plan is not None:
            attended = cp_shard_rows(attended, plan)
        return torch.sin(values + attended * self.scale), residual * 0.25


class _CpuBackend:
    is_mla = False

    def prepare(self, metadata: object) -> None:
        pass


def _model(monkeypatch: pytest.MonkeyPatch) -> Qwen3VLForConditionalGeneration:
    def residual_norm(
        hidden: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = hidden + residual
        return torch.nn.functional.rms_norm(combined, (combined.shape[-1],), weight, epsilon), combined

    monkeypatch.setattr(kernels, "fused_add_rms_norm", residual_norm, raising=False)
    cfg = Qwen3Config(
        hidden_size=2,
        head_dim=2,
        n_heads=1,
        n_kv_heads=1,
        n_layers=0,
        vocab_size=4,
        max_position_embeddings=256,
        mrope_section=[1, 0, 0],
    )
    host = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
    nn.Module.__init__(host)
    host.model = Qwen3VLModel(cfg, torch.float32, torch.device("cpu"))
    host.model.embed_tokens = nn.Embedding.from_pretrained(
        torch.tensor([[0.1, 0.2], [0.3, 0.1], [0.2, 0.4], [0.4, 0.3]])
    )
    host.model.layers = nn.ModuleList([_CausalDecoder(0.5), _CausalDecoder(1.5), _CausalDecoder(0.75)])
    host.deepstack_num_level = 2
    return host


def test_position_shards_preserve_absolute_axes_and_zero_padding() -> None:
    positions = torch.tensor(
        [[11, 12, 17, 18, 19, 51, 52, 91], [21, 22, 27, 28, 29, 61, 62, 92], [31, 32, 37, 38, 39, 71, 72, 93]],
        dtype=torch.int32,
    )
    plan = build_cp_context([5, 2, 1], [5, 2, 1], 2, 1, torch.device("cpu"))
    expected = torch.tensor(
        [[17, 18, 19, 0, 52, 0, 0, 0], [27, 28, 29, 0, 62, 0, 0, 0], [37, 38, 39, 0, 72, 0, 0, 0]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(cp_shard_positions(positions, plan), expected)
    torch.testing.assert_close(cp_shard_positions(positions[0], plan), expected[0])
    padding_plan = build_cp_context([1, 1], [1, 1], 2, 1, torch.device("cpu"))
    torch.testing.assert_close(cp_shard_positions(positions[:, :2], padding_plan), torch.zeros(3, 4, dtype=torch.int32))
    with pytest.raises(ValueError, match=r"got \(8, 3\)"):
        cp_shard_positions(positions.t(), plan)
    with pytest.raises(ValueError, match=r"got \(1, 3, 8\)"):
        cp_shard_positions(positions.unsqueeze(0), plan)


def test_fused_cp_restores_ragged_outputs_and_next_decode_has_no_cp(monkeypatch: pytest.MonkeyPatch) -> None:
    host = _model(monkeypatch)
    # Ragged non-divisible prompts and a distinct rank with no real query rows.
    for lengths in ([5, 2, 1], [1, 1]):
        total = sum(lengths)
        input_ids = torch.arange(total) % 4
        mask = torch.zeros(total, dtype=torch.bool)
        mask[1::2] = True
        features = torch.arange(int(mask.sum()) * 6).reshape(-1, 6).float() * 0.03 + 0.2
        prepared = host.prepare_multimodal_inputs(input_ids, {"image|mask": mask, "image|embedding": features})
        original = (prepared.inputs_embeds.clone(), *(level.clone() for level in prepared.deepstack_input_embeds))
        base_positions = torch.arange(total, dtype=torch.int32) + 11
        for positions in (base_positions, torch.stack([base_positions, base_positions + 30, base_positions + 80])):
            metadata = _metadata(lengths)
            baseline = eager.EagerRunner(host.model, _CpuBackend(), torch.device("cpu"))
            expected = baseline.execute(input_ids, positions, metadata, multimodal_inputs=prepared)
            global_layer_inputs = [layer.inputs[-1][0].clone() for layer in host.model.layers]
            collective = _CpuCollective()
            monkeypatch.setattr(distributed, "all_gather", collective.all_gather, raising=False)

            def execute_rank(
                rank: int,
                *,
                input_ids: torch.Tensor = input_ids,
                positions: torch.Tensor = positions,
                metadata: SimpleNamespace = metadata,
                prepared: MultimodalInputs = prepared,
            ) -> tuple[torch.Tensor, eager.EagerRunner]:
                runner = eager.EagerRunner(host.model, _CpuBackend(), torch.device("cpu"))
                runner.cp_size, runner.cp_rank = 2, rank
                return runner.execute(input_ids, positions, metadata, multimodal_inputs=prepared), runner

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(execute_rank, range(2)))
            # Check the rows entering each decoder after both DeepStack
            # injections, not just the final normalized output.
            for rank in range(2):
                plan = build_cp_context(lengths, lengths, 2, rank, torch.device("cpu"))
                for layer, global_hidden in zip(host.model.layers, global_layer_inputs, strict=True):
                    local_hidden, local_positions = layer.inputs[rank]
                    torch.testing.assert_close(
                        local_hidden[plan.shard_valid_mask], global_hidden[plan.shard_index[plan.shard_valid_mask]]
                    )
                    assert local_positions.shape[-1] == plan.total_local
            for output, runner in results:
                torch.testing.assert_close(output, expected)
                # Decode must neither shard nor inherit the previous DeepStack.
                decoded = runner.execute(torch.tensor([2]), torch.tensor([103]), _metadata([1], prefill=False))
                reference = baseline.execute(torch.tensor([2]), torch.tensor([103]), _metadata([1], prefill=False))
                torch.testing.assert_close(decoded, reference)
            with pytest.raises(RuntimeError, match="not set"):
                get_forward_context()
            first_hidden, first_positions = host.model.layers[0].inputs[1]
            plan = build_cp_context(lengths, lengths, 2, 1, torch.device("cpu"))
            assert first_hidden.shape[0] == plan.total_local
            padding = first_hidden[~plan.shard_valid_mask]
            torch.testing.assert_close(padding, torch.zeros_like(padding))
            assert first_positions.shape[-1] == plan.total_local
        for actual, before in zip((prepared.inputs_embeds, *prepared.deepstack_input_embeds), original, strict=True):
            torch.testing.assert_close(actual, before)


def test_cp_rejects_unsupported_capture_and_misaligned_global_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(monkeypatch).model
    plan = build_cp_context([1, 1], [1, 1], 2, 0, torch.device("cpu"))
    context = ForwardContext(None, torch.device("cpu"), _metadata([1, 1]), [], cp_context=plan)
    ids, positions = torch.tensor([0, 1]), torch.tensor([11, 21])
    metadata = _metadata([1, 1])
    metadata.kv_seq_lens_host = torch.tensor([2, 1])
    with forward_context(replace(context, metadata=metadata)), pytest.raises(NotImplementedError, match="KV prefix"):
        model(ids, positions)
    model.cfg.layers_to_capture = (0,)
    with forward_context(context), pytest.raises(NotImplementedError, match="auxiliary hidden capture"):
        model(ids, positions)
    model.cfg.layers_to_capture = ()
    with (
        forward_context(replace(context, layer_synchronizer=object())),
        pytest.raises(NotImplementedError, match="PD or auxiliary"),
    ):
        model(ids, positions)
    with forward_context(context), pytest.raises(ValueError, match="unsharded"):
        model(ids, positions, inputs_embeds=torch.zeros(4, 2))
    with forward_context(context), pytest.raises(ValueError, match="DeepStack"):
        model(ids, positions, deepstack_input_embeds=(torch.zeros(4, 2),))
