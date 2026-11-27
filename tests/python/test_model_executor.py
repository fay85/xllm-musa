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

"""Unit tests for xllm.python.model_executor.executor.

Tests the device-conditional backend dispatch, ModelExecutor construction
validation, and execution routing — using CPU mocks so no GPU/NPU required.
"""

from __future__ import annotations

import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python import distributed

# conftest.py stands in for xllm.python, whose import would bind the active
# platform's kernel package and reach for operators from the C++ binary.
from xllm.python.attention.backend import (  # noqa: E402
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    normalize_layer_caches,
)
from xllm.python.attention.csa2_attention import Csa2AttentionBackend  # noqa: E402
from xllm.python.attention.csa2_block_metadata import Csa2BlockAttentionMetadataBuilder
from xllm.python.layers.npu.qwen3_5.moe_metadata_builder import Qwen3_5MoeMetadataBuilder

# These dispatch tests replace NPU backend constructors. Import the real option
# resolver with a scoped dependency placeholder on CPU; native calls stay absent
# and would fail rather than silently emulate kernels.
if "torch_npu" not in sys.modules and find_spec("torch_npu") is None:
    sys.modules["torch_npu"] = types.ModuleType("torch_npu")
    try:
        from xllm.python.attention.sfa_dcp_backend import dcp_layer_options
    finally:
        del sys.modules["torch_npu"]
else:
    from xllm.python.attention.sfa_dcp_backend import dcp_layer_options
from xllm.python.layers.attention import Attention  # noqa: E402
from xllm.python.layers.moe_parallel import Eplv2CommPolicy  # noqa: E402
from xllm.python.layers.npu.mega_moe_metadata import MegaMoeMetadata  # noqa: E402
from xllm.python.layers.npu.qwen3_5.moe_metadata import Qwen3_5MoeMetadata  # noqa: E402
from xllm.python.layers.sequence_parallel import SpMetadataBuilder  # noqa: E402
from xllm.python.layers.sequence_parallel_metadata import SpMetadata  # noqa: E402
from xllm.python.model_executor.execution_context import (  # noqa: E402
    LayerCacheAwareExecutionMetadataBuilder,
)
from xllm.python.model_executor.executor import (  # noqa: E402
    ModelExecutor,
    _create_attention_backend,
)
from xllm.python.model_executor.forward_context import (  # noqa: E402
    EplbRuntimeState,
    ForwardContext,
    LayerLoadContext,
    forward_context,
    get_execution_context,
    get_forward_context,
    record_layer_event,
)
from xllm.python.model_executor.input_batch import InputBatch  # noqa: E402
from xllm.python.model_executor.model_input import ModelInput
from xllm.python.model_executor.model_output import ModelOutput
from xllm.python.model_executor.mtp_topk import MtpTopkReuse
from xllm.python.model_executor.runners.decode_acl_graph import (  # noqa: E402
    DecodeAclGraphRunner,
)
from xllm.python.model_executor.runners.decode_cuda_graph import (  # noqa: E402
    DecodeCudaGraphRunner,
    _decode_bucket,
    _decode_graph_buckets,
)
from xllm.python.model_executor.runners.eager import EagerRunner  # noqa: E402
from xllm.python.model_executor.sampling_metadata import ModelSamplingMetadataBuilder
from xllm.python.models.multimodal import MultimodalInputs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubAttentionBackend(AttentionBackend):
    """Minimal backend that records calls for assertion."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self._kv_caches: list[LayerCache] = []
        self._prepared = False

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        self._kv_caches = kv_caches

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        self._prepared = True

    def execute(self, q, k, v, layer) -> torch.Tensor:
        return q

    @property
    def num_kv_blocks(self) -> int:
        return 0

    @property
    def page_size(self) -> int:
        return 1


class _Csa2StubAttentionBackend(StubAttentionBackend):
    # Exercise the real configuration adapter while stubbing device execution.
    from_model_config = classmethod(Csa2AttentionBackend.from_model_config.__func__)


class _PagedStubAttentionBackend(StubAttentionBackend):
    @property
    def page_size(self) -> int:
        return 4


class _MlaStubAttentionBackend(StubAttentionBackend):
    @property
    def is_mla(self) -> bool:
        return True


def _make_attention_layer(
    num_heads=8,
    num_kv_heads=2,
    head_dim=64,
    scale=0.125,
    sliding_window=0,
    layer_id=0,
) -> Attention:
    return Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        sliding_window=sliding_window,
        layer_id=layer_id,
    )


class _FakeModel(nn.Module):
    """Model with configurable number of uniform Attention layers."""

    def __init__(self, num_layers: int = 2, device: str = "cpu", **attn_kwargs):
        super().__init__()
        self.model = nn.Linear(1, 1)  # execution_model placeholder
        self.layers = nn.ModuleList([_make_attention_layer(layer_id=i, **attn_kwargs) for i in range(num_layers)])
        self._param = nn.Parameter(torch.zeros(1, device=device))

    def forward(self, input_ids, positions):
        return input_ids


@pytest.fixture
def _v41_backend_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    from xllm.python.model_executor import executor as executor_module

    monkeypatch.setattr(executor_module.current_platform, "is_npu", lambda: True)
    return {
        "model_path": str(tmp_path),
        "model_type": "deepseek_v41",
        "weight_quant_method": "ascend",
        "quantization_config": {"quant_method": "ascend"},
        "max_position_embeddings": 1024,
        "compress_ratios": [0],
        "n_layers": 1,
    }


@pytest.mark.parametrize(("enable_graph", "dp_size"), [(False, 1), (True, 2)])
def test_v41_executor_preserves_csa2_dispatch_after_main_integration(
    enable_graph: bool, dp_size: int, tmp_path: Path, _v41_backend_config: dict[str, object]
) -> None:
    """Exercise real CSA2 construction without launching NPU graph execution."""
    from xllm.python.attention.csa2_attention import Csa2AttentionBackend

    model = _FakeModel(num_layers=1, num_heads=2, num_kv_heads=1, head_dim=64)
    (tmp_path / "quant_model_description.json").write_text("{}", encoding="utf-8")
    config = {
        **_v41_backend_config,
        "enable_graph": enable_graph,
        "window_size": 128,
        "index_topk": 8,
        "index_n_heads": 2,
        "index_head_dim": 64,
        "qk_rope_head_dim": 32,
        "dp_size": dp_size,
        "dp_rank": 0,
        "ep_size": 16,
        "expert_parallel_degree": 2,
        "enable_mega_moe": False,
        "num_speculative_tokens": 0,
    }
    with patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner") as graph_runner:
        executor = ModelExecutor(model, config, max_seqs_per_batch=8, acl_graph_decode_batch_size_limit=4)

    backend = executor.attention_backend
    assert isinstance(backend, Csa2AttentionBackend)
    assert backend.quant_mode == "ascend"
    assert backend.moe_graph_capturable is True
    assert backend.graph_token_capacity_ceiling == 0
    assert executor._execution_metadata_builders == ()
    assert executor.eager_runner.cp_size == 1
    if not enable_graph:
        graph_runner.assert_not_called()
        assert executor.decode_graph_runner is None
    else:
        graph_runner.assert_called_once_with(
            model.model,
            backend,
            torch.device("cpu"),
            8,
            1024,
            dp_size,
            0,
            4,
            num_speculative_tokens=0,
            enable_mega_moe_token_mask=False,
            is_spec_draft=False,
            model_type="",
            capture_sizes=(),
        )
        executor.decode_graph_runner.bind_execution_metadata_builders.assert_called_once_with(())


@pytest.mark.parametrize(("quant_method", "graph_capturable"), [("ascend", True), ("fp8", False), ("none", False)])
def test_v41_backend_keeps_checkpoint_specific_graph_capability(
    quant_method: str, graph_capturable: bool, tmp_path: Path, _v41_backend_config: dict[str, object]
) -> None:
    from xllm.python.attention.csa2_attention import Csa2AttentionBackend

    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": quant_method}}), encoding="utf-8"
    )
    backend = _create_attention_backend(
        _make_attention_layer(num_heads=2, num_kv_heads=1, head_dim=64),
        torch.device("cpu"),
        torch.bfloat16,
        {
            **_v41_backend_config,
            "weight_quant_method": quant_method,
            "quantization_config": {"quant_method": quant_method},
        },
        acl_graph_enabled=True,
    )
    assert isinstance(backend, Csa2AttentionBackend)
    assert backend.quant_mode == quant_method
    assert backend.moe_graph_capturable is graph_capturable


class _FakeModelHeterogeneous(nn.Module):
    """Model with non-uniform Attention layers (should fail validation)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.attn1 = _make_attention_layer(num_heads=8, layer_id=0)
        self.attn2 = _make_attention_layer(num_heads=4, layer_id=1)
        self._param = nn.Parameter(torch.zeros(1))


class _FakeModelNoAttention(nn.Module):
    """Model without any Attention layers."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self._param = nn.Parameter(torch.zeros(1))


class _FailingLayerSynchronizer:
    def record_event(self, layer_id: int) -> bool:
        return False


def _make_input_batch(num_scheduled_tokens: tuple[int, ...]) -> InputBatch:
    num_tokens = sum(num_scheduled_tokens)
    query_start_loc = [0]
    for count in num_scheduled_tokens:
        query_start_loc.append(query_start_loc[-1] + count)
    metadata = SimpleNamespace(
        num_reqs=len(num_scheduled_tokens),
        num_tokens=num_tokens,
        num_scheduled_tokens=list(num_scheduled_tokens),
        num_computed_tokens=[0] * len(num_scheduled_tokens),
        query_start_loc=query_start_loc,
        is_prefilling=[0] * len(num_scheduled_tokens),
    )
    return InputBatch.from_runtime(
        torch.zeros(num_tokens, dtype=torch.int32),
        torch.arange(num_tokens, dtype=torch.int32),
        metadata,
        is_dummy=False,
    )


def test_record_layer_event_propagates_record_failure() -> None:
    context = ForwardContext(
        attention_backend=StubAttentionBackend(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        layer_synchronizer=_FailingLayerSynchronizer(),
    )

    with (
        forward_context(context),
        pytest.raises(RuntimeError, match="failed to record layer completion event for layer 3"),
    ):
        record_layer_event(3)


# ---------------------------------------------------------------------------
# Tests: graph backend resolution
# ---------------------------------------------------------------------------


class _HostLoadSynchronizer:
    def __init__(self, count: int, failed_event: int | None = None) -> None:
        self.count = count
        self.failed_event = failed_event
        self.calls: list[int] = []

    def size(self) -> int:
        return self.count

    def synchronize_layer(self, event_index: int) -> bool:
        self.calls.append(event_index)
        return event_index != self.failed_event


def test_host_load_groups_wait_once_and_reset_between_forwards() -> None:
    synchronizer = _HostLoadSynchronizer(3)
    for _ in range(2):
        context = LayerLoadContext(synchronizer, layers_per_event=2)
        # A layerwise partition may start in the middle of an event group.
        for layer_id in (1, 2, 3, 4, 4, 5):
            context.synchronize_layer(layer_id)
        context.synchronize_all_layers()
    assert synchronizer.calls == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize("layers_per_event", [0])
def test_host_load_rejects_invalid_group_size(layers_per_event: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        LayerLoadContext(_HostLoadSynchronizer(1), layers_per_event)


@pytest.mark.parametrize("layer_id", [-1, 2])
def test_host_load_rejects_invalid_layer_index(layer_id: int) -> None:
    synchronizer = _HostLoadSynchronizer(1)
    context = LayerLoadContext(synchronizer, layers_per_event=2)
    with pytest.raises(ValueError):
        context.synchronize_layer(layer_id)
    assert synchronizer.calls == []


def test_host_load_failure_does_not_mark_event_complete() -> None:
    synchronizer = _HostLoadSynchronizer(1, failed_event=0)
    context = LayerLoadContext(synchronizer)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Host cache load event 0"):
            context.synchronize_layer(0)
    assert synchronizer.calls == [0, 0]


@pytest.mark.parametrize(
    "module_name,class_name,num_inputs",
    [
        ("layers.attention", "Attention", 3),
        ("models.glm5_next", "Glm5NextMlaAttention", 3),
        ("models.glm5_next", "Glm5NextKdaAttention", 2),
        ("models.glm5_2", "Glm52MLAAttention", 3),
        ("models.deepseek_v32", "DeepseekV3MLAAttention", 5),
        ("models.deepseek_v4", "DeepseekV4Attention", 3),
    ],
)
def test_attention_aborts_before_any_projection_or_cache_access(
    module_name: str, class_name: str, num_inputs: int
) -> None:
    attention_class = getattr(import_module(f"xllm.python.{module_name}"), class_name)
    attention = attention_class.__new__(attention_class)
    nn.Module.__init__(attention)
    attention.layer_id = 0
    # No weights or cache state are initialized: the failed load must abort
    # before the attention body can touch either of them.
    context = ForwardContext(
        StubAttentionBackend(),
        torch.device("cpu"),
        MagicMock(),
        [],
        layer_load_context=LayerLoadContext(_HostLoadSynchronizer(1, failed_event=0)),
    )
    with forward_context(context), pytest.raises(RuntimeError, match="Host cache load event 0"):
        attention(*[torch.zeros(1) for _ in range(num_inputs)])


def test_attention_blocks_until_its_host_copy_completes() -> None:
    waiting = [Event(), Event()]
    ready = [Event(), Event()]
    computed = [Event(), Event()]

    class AsyncLoads(_HostLoadSynchronizer):
        def synchronize_layer(self, event_index: int) -> bool:
            self.calls.append(event_index)
            waiting[event_index].set()
            return ready[event_index].wait(timeout=5)

    class Backend(StubAttentionBackend):
        def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
            assert ready[layer.layer_id].is_set()
            computed[layer.layer_id].set()
            return q

    synchronizer = AsyncLoads(2)
    context = ForwardContext(
        Backend(),
        torch.device("cpu"),
        MagicMock(),
        [],
        layer_load_context=LayerLoadContext(synchronizer),
    )

    def forward() -> None:
        with forward_context(context):
            q = torch.zeros(2, 8, 64)
            k = torch.zeros(2, 2, 64)
            v = torch.ones_like(k)
            for layer_id in range(2):
                _make_attention_layer(layer_id=layer_id)(q, k, v)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(forward)
        try:
            assert waiting[0].wait(timeout=5)
            assert not computed[0].is_set()
            ready[0].set()
            assert computed[0].wait(timeout=5)
            assert waiting[1].wait(timeout=5)
            assert not computed[1].is_set()
        finally:
            for event in ready:
                event.set()
        future.result(timeout=5)
    assert computed[1].is_set()
    assert synchronizer.calls == [0, 1]


class TestGraphRunnerSelection:
    @pytest.mark.parametrize(
        ("platform", "enable_graph", "legacy_backend"),
        [
            ("npu", None, None),
            ("npu", False, "inductor"),
            ("npu", True, "off"),
            ("cuda", False, "cudagraphs"),
            ("cuda", True, None),
            ("unsupported", False, "off"),
            ("unsupported", True, None),
        ],
    )
    def test_graph_selection_uses_only_enable_graph_and_platform(
        self, platform: str, enable_graph: bool | None, legacy_backend: str | None
    ) -> None:
        config: dict[str, object] = {"max_position_embeddings": 128}
        if enable_graph is not None:
            config["enable_graph"] = enable_graph
        if legacy_backend is not None:
            config["python_graph_backend"] = legacy_backend
        with (
            patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=platform == "npu"),
            patch("xllm.python.model_executor.executor.current_platform.is_cuda", return_value=platform == "cuda"),
            patch(
                "xllm.python.model_executor.executor._create_attention_backend", return_value=StubAttentionBackend()
            ) as create_attention_backend,
            patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner") as acl_graph_runner,
            patch("xllm.python.model_executor.runners.decode_cuda_graph.DecodeCudaGraphRunner") as cuda_graph_runner,
            patch("xllm.python.model_executor.executor.logger.warning") as warning,
        ):
            if enable_graph and platform == "unsupported":
                with pytest.raises(NotImplementedError, match="requires CUDA or NPU"):
                    ModelExecutor(_FakeModel(num_layers=1), config, max_seqs_per_batch=4)
                create_attention_backend.assert_not_called()
            else:
                executor = ModelExecutor(_FakeModel(num_layers=1), config, max_seqs_per_batch=4)
                expected_runner = None
                if enable_graph:
                    expected_runner = (
                        acl_graph_runner.return_value if platform == "npu" else cuda_graph_runner.return_value
                    )
                assert executor.decode_graph_runner is expected_runner
                assert create_attention_backend.call_args.kwargs["acl_graph_enabled"] == (
                    bool(enable_graph) and platform == "npu"
                )

            assert acl_graph_runner.call_count == int(bool(enable_graph) and platform == "npu")
            assert cuda_graph_runner.call_count == int(bool(enable_graph) and platform == "cuda")
            messages = [warning_call.args[0] for warning_call in warning.call_args_list]
            cuda_graph_requested = bool(enable_graph) and platform == "cuda"
            assert len(messages) == int(legacy_backend is not None) + int(cuda_graph_requested)
            if legacy_backend is not None:
                message = messages[0]
                assert "python_graph_backend has been removed and is ignored" in message
            if cuda_graph_requested:
                message = messages[-1]
                assert "decode will run eagerly without CUDA graph capture/replay" in message

    @pytest.mark.parametrize(("platform", "enable_graph"), [("npu", False), ("npu", True), ("cuda", True)])
    def test_layerwise_split_requires_graph_disabled(self, platform: str, enable_graph: bool) -> None:
        config = {
            "model_type": "glm_moe_dsa",
            "layerwise_split_size": 2,
            "enable_graph": enable_graph,
            "max_position_embeddings": 128,
        }
        with (
            patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=platform == "npu"),
            patch("xllm.python.model_executor.executor.current_platform.is_cuda", return_value=platform == "cuda"),
            patch("xllm.python.model_executor.executor._create_attention_backend", return_value=StubAttentionBackend()),
            patch("xllm.python.model_executor.executor.get_execution_metadata_builder_classes", return_value=()),
            patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner") as acl_graph_runner,
            patch("xllm.python.model_executor.runners.decode_cuda_graph.DecodeCudaGraphRunner") as cuda_graph_runner,
        ):
            if enable_graph:
                with pytest.raises(NotImplementedError, match="layerwise split requires eager execution"):
                    ModelExecutor(_FakeModel(num_layers=1), config, max_seqs_per_batch=4)
            else:
                executor = ModelExecutor(_FakeModel(num_layers=1), config, max_seqs_per_batch=4)
                assert executor.decode_graph_runner is None
            acl_graph_runner.assert_not_called()
            cuda_graph_runner.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _create_attention_backend dispatch
# ---------------------------------------------------------------------------


class TestCreateAttentionBackend:
    @pytest.mark.parametrize("is_ascend_a3", [False, True])
    @pytest.mark.parametrize("proposal_width", [0, 8])
    def test_dflash2_xfia_requires_a3(self, is_ascend_a3: bool, proposal_width: int) -> None:
        native_query = MagicMock(return_value=is_ascend_a3)
        runtime = types.ModuleType("xllm_runtime")
        runtime.is_ascend_a3 = native_query
        with (
            patch.dict(sys.modules, {"xllm_runtime": runtime}),
            patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
            patch(
                "xllm.python.model_executor.executor.current_platform.get_ascend_soc_generation",
                return_value="a2" if is_ascend_a3 else "a3",
            ) as python_soc_query,
            patch("xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend", StubAttentionBackend),
        ):
            backend = _create_attention_backend(
                _make_attention_layer(num_kv_heads=1, head_dim=128),
                MagicMock(type="npu", index=0),
                torch.bfloat16,
                {
                    "model_type": "DFlash2DraftModel" if proposal_width else "",
                    "num_speculative_tokens": max(proposal_width - 1, 0),
                    "dflash_xfia_enabled": True,
                },
            )
        assert backend.init_kwargs["use_xfia_decode"] == (is_ascend_a3 and proposal_width > 0)
        assert backend.init_kwargs["xfia_query_width"] == max(proposal_width, 1)
        python_soc_query.assert_not_called()
        if proposal_width > 0:
            native_query.assert_called_once_with()
        else:
            native_query.assert_not_called()

    def test_dflash2_xfia_stays_off_without_prelaunch_gate(self) -> None:
        native_query = MagicMock(return_value=True)
        runtime = types.ModuleType("xllm_runtime")
        runtime.is_ascend_a3 = native_query
        config = {"model_type": "DFlash2DraftModel", "num_speculative_tokens": 7, "dflash_xfia_enabled": False}
        with (
            patch.dict(sys.modules, {"xllm_runtime": runtime}),
            patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
            patch("xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend", StubAttentionBackend),
        ):
            backend = _create_attention_backend(
                _make_attention_layer(num_kv_heads=1, head_dim=128),
                MagicMock(type="npu", index=0),
                torch.bfloat16,
                config,
            )
        assert backend.init_kwargs["use_xfia_decode"] is False
        assert backend.init_kwargs["xfia_query_width"] == 8
        native_query.assert_not_called()

    @pytest.mark.parametrize(
        ("dcp_size", "model_type"),
        [(None, "DFlashDraftModel"), (4, "DFlashDraftModel"), (4, "DFlash2DraftModel")],
    )
    def test_block_query_graph_uses_draft_kv_layout(self, dcp_size: int | None, model_type: str) -> None:
        pytest.importorskip("torch_npu")

        dcp_group = None if dcp_size is None else MagicMock()
        if dcp_group is not None:
            dcp_group.size.return_value = dcp_size
        attention = _make_attention_layer(num_kv_heads=1, head_dim=128)
        attention.attention_window = (7, 3)
        device = torch.device("cpu")
        num_speculative_tokens = 3
        query_width = num_speculative_tokens + 1
        with (
            patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
            patch("xllm.python.model_executor.executor.distributed.dcp_group", return_value=dcp_group),
        ):
            backend = _create_attention_backend(
                attention,
                device,
                torch.bfloat16,
                {"is_draft_engine": True, "num_speculative_tokens": num_speculative_tokens, "model_type": model_type},
                block_query_graph_metadata=True,
            )
        cache = torch.empty(1, 128, 1, 128, dtype=torch.bfloat16, device=device)
        backend.bind_kv_caches([LayerCache(key=cache, value=torch.empty_like(cache))])
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            backend,
            device,
            max_batch=1,
            max_model_len=128,
            is_spec_draft=True,
            num_speculative_tokens=num_speculative_tokens,
            model_type=model_type,
        )
        metadata = SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=True,
            is_dummy=False,
            is_mixed=False,
            use_dflash_proposal_attention=True,
            kv_split_size=1,
            slot_mapping=torch.arange(query_width, dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            kv_seq_lens=torch.tensor([query_width], dtype=torch.int32),
            kv_seq_lens_host_values=[query_width],
            q_cu_seq_lens=torch.tensor([0, query_width], dtype=torch.int32),
            dp_global_kv_max_seq_lens=(),
        )
        input_ids = torch.zeros(query_width, dtype=torch.int32)
        model_input = ModelInput(input_ids, torch.arange(query_width), metadata)
        assert runner.can_execute(model_input)
        if dcp_size is not None:
            metadata.kv_split_size = dcp_size
            assert not runner.can_execute(model_input)

    @pytest.mark.parametrize("is_ascend_a3", [False, True])
    @pytest.mark.parametrize("dflash_xfia_enabled", [False, True])
    @pytest.mark.parametrize("block_query_graph_metadata", [False, True])
    def test_dspark_xfia_uses_generic_proposal_width(
        self,
        is_ascend_a3: bool,
        dflash_xfia_enabled: bool,
        block_query_graph_metadata: bool,
    ) -> None:
        native_query = MagicMock(return_value=is_ascend_a3)
        runtime = types.ModuleType("xllm_runtime")
        runtime.is_ascend_a3 = native_query
        with (
            patch.dict(sys.modules, {"xllm_runtime": runtime}),
            patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
            patch("xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend", StubAttentionBackend),
        ):
            backend = _create_attention_backend(
                _make_attention_layer(num_kv_heads=1, head_dim=128),
                MagicMock(type="npu", index=0),
                torch.bfloat16,
                {
                    "model_type": "DSparkDraftModel",
                    "num_speculative_tokens": 15,
                    "dflash_xfia_enabled": dflash_xfia_enabled,
                },
                block_query_graph_metadata=block_query_graph_metadata,
            )
        assert backend.init_kwargs["use_xfia_decode"] == (
            is_ascend_a3 and dflash_xfia_enabled and not block_query_graph_metadata
        )
        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["xfia_query_width"] == 15
        if dflash_xfia_enabled and not block_query_graph_metadata:
            native_query.assert_called_once_with()
        else:
            native_query.assert_not_called()

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_deepseek_v4_creates_dsa_backend(self, _mock_is_npu):
        attn = _make_attention_layer(head_dim=512)
        module = types.ModuleType("xllm.python.attention.dsa_attention")
        module.DsaAttentionBackend = StubAttentionBackend
        config = {
            "model_type": "deepseek_v4",
            "compress_ratios": [1, 4, 128],
            "num_hidden_layers": 3,
            "window_size": 128,
            "index_topk": 512,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "qk_rope_head_dim": 64,
        }
        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, MagicMock(type="npu", index=0), torch.bfloat16, config)

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["attn_head_dim"] == 512
        assert backend.init_kwargs["num_speculative_tokens"] == 0
        assert backend.init_kwargs["n_layers"] == 3
        assert backend.init_kwargs["compress_ratios"] == [1, 4, 128]

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_deepseek_v4_dspark_passes_block_size(self, mock_is_npu: MagicMock) -> None:
        """The DSpark model keeps its configured query width."""
        mock_is_npu.assert_not_called()
        attn = _make_attention_layer(head_dim=512)
        module = types.ModuleType("xllm.python.attention.dsa_attention")
        module.DsaAttentionBackend = StubAttentionBackend
        config = {
            "model_type": "deepseek_v4_dspark",
            "compress_ratios": [1, 1, 1],
            "n_layers": 3,
            "window_size": 128,
            "num_speculative_tokens": 5,
        }

        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, MagicMock(type="npu", index=0), torch.bfloat16, config)

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["num_speculative_tokens"] == 5

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_deepseek_v41_creates_csa2_backend(self, _mock_is_npu):
        attn = _make_attention_layer(head_dim=512)
        module = types.ModuleType("xllm.python.attention.csa2_attention")
        module.Csa2AttentionBackend = _Csa2StubAttentionBackend
        config = {
            "model_type": "deepseek_v41",
            "compress_ratios": [0, 2, 1],
            "n_layers": 3,
            "window_size": 128,
            "index_topk": 512,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "qk_rope_head_dim": 64,
            "kv_source_layer_ids": [1, 2],
            "index_source_layer_ids": [1, 2],
            "candidate_source_layer_id": 2,
        }
        with patch.dict(sys.modules, {module.__name__: module}):
            # The device only reaches the stubbed backend: the dispatch under
            # test keys on current_platform.is_npu(), and "cpu" keeps this
            # test runnable on hosts without torch_npu.
            backend = _create_attention_backend(attn, torch.device("cpu"), torch.bfloat16, config)

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["attn_head_dim"] == 512
        assert backend.init_kwargs["compress_ratios"] == [0, 2, 1]

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_deepseek_v41_dspark_creates_csa2_backend(self, _mock_is_npu):
        """The V4.1 DSpark draft must reuse the target's CSA2 backend.

        The C++ side routes --model_impl=python drafts to the
        deepseek_v41_dspark model_type (xllm/python/registry.py) instead of the
        C++ TORCH deepseek_v4_dspark name. The draft is still a V4.1-family
        model, so it must dispatch to CSA2 and see its own pure-SWA cache plan:
        3 draft layers with compress ratio 0 and no compressed-KV sources.
        """
        attn = _make_attention_layer(head_dim=512)
        module = types.ModuleType("xllm.python.attention.csa2_attention")
        module.Csa2AttentionBackend = _Csa2StubAttentionBackend
        config = {
            "model_type": "deepseek_v41_dspark",
            "num_speculative_tokens": 3,
            "compress_ratios": [0, 0, 0],
            "n_layers": 3,
            "window_size": 128,
            "index_topk": 512,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "qk_rope_head_dim": 64,
            # Cleared by configure_deepseek_v4_dspark_args: draft stage ids are
            # decoder ids and would fall outside the draft's layer range.
            "kv_source_layer_ids": [],
            "index_source_layer_ids": [],
            "candidate_source_layer_id": -1,
            "candidate_topk_blocks": 0,
            "candidate_block_size": 0,
        }
        with patch.dict(sys.modules, {module.__name__: module}):
            # The device only reaches the stubbed backend: the dispatch under
            # test keys on current_platform.is_npu(), and "cpu" keeps this
            # test runnable on hosts without torch_npu.
            backend = _create_attention_backend(attn, torch.device("cpu"), torch.bfloat16, config)

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["n_layers"] == 3
        assert backend.init_kwargs["compress_ratios"] == [0, 0, 0]
        assert backend.init_kwargs["kv_source_layer_ids"] == []
        assert backend.init_kwargs["candidate_source_layer_id"] == -1

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_deepseek_v4_dspark_does_not_reuse_v41_csa2_backend(self, _mock_is_npu):
        """Guard: the C++ TORCH draft name must not silently pick up CSA2.

        Only the Python-registered V4.1 draft name maps to the CSA2 backend; the
        C++ TORCH deepseek_v4_dspark name keeps its own V4 DSA backend, so a
        mis-routed draft is caught here instead of running the V4.1 draft body
        against a non-V4.1 cache plan.
        """

        class _Csa2Stub(StubAttentionBackend):
            pass

        class _DsaStub(StubAttentionBackend):
            pass

        attn = _make_attention_layer(head_dim=512)
        csa2_module = types.ModuleType("xllm.python.attention.csa2_attention")
        csa2_module.Csa2AttentionBackend = _Csa2Stub
        dsa_module = types.ModuleType("xllm.python.attention.dsa_attention")
        dsa_module.DsaAttentionBackend = _DsaStub
        config = {
            "model_type": "deepseek_v4_dspark",
            "compress_ratios": [0, 0, 0],
            "n_layers": 3,
        }
        with patch.dict(
            sys.modules,
            {csa2_module.__name__: csa2_module, dsa_module.__name__: dsa_module},
        ):
            # The device only reaches the stubbed backend: the dispatch under
            # test keys on current_platform.is_npu(), and "cpu" keeps this
            # test runnable on hosts without torch_npu.
            backend = _create_attention_backend(attn, torch.device("cpu"), torch.bfloat16, config)

        assert isinstance(backend, _DsaStub)
        assert not isinstance(backend, _Csa2Stub)

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    @pytest.mark.parametrize("num_decoding_tokens", [1, 4])
    def test_npu_device_creates_npu_backend(self, _mock_is_npu: MagicMock, num_decoding_tokens: int) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        backend = _create_attention_backend(
            attn,
            MagicMock(type="npu", index=0),
            torch.float16,
            {"num_speculative_tokens": num_decoding_tokens - 1, "enable_mla": False},
        )
        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["num_heads"] == 8
        assert backend.init_kwargs["num_kv_heads"] == 1
        assert backend.init_kwargs["head_dim"] == 256
        assert backend.init_kwargs["is_mla"] is False
        assert backend.init_kwargs["num_decoding_tokens"] == num_decoding_tokens

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    def test_prefill_cp_uses_npu_backend_with_dcp_group(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = MagicMock()
        sfa_module.dcp_layer_options = MagicMock(return_value=512)

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            backend = _create_attention_backend(
                attn,
                MagicMock(type="npu", index=0),
                torch.float16,
                {"cp_size": 4, "enable_mla": True},
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["is_mla"] is True
        sfa_module.SfaDcpAttentionBackend.assert_not_called()

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_dflash2_draft_cp1_falls_through_to_plain_paged_backend(self, _mock_is_npu: MagicMock) -> None:
        """M11.3 admission pin companion: the DFlash2 GQA draft at cp_size == 1
        with a live DCP group is NOT sharded -- it keeps a full replica of its
        own pool, read through the expanded block table -- so it must skip the
        SfaDcp branch entirely (executor's ``is_draft_engine`` fall-through)
        and land on the plain paged backend with no DCP group attached. A
        dense (non-draft) target in the same shape is refused instead."""
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = MagicMock()
        sfa_module.dcp_layer_options = MagicMock(return_value=512)

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch(
                "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
                StubAttentionBackend,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            backend = _create_attention_backend(
                attn,
                MagicMock(type="npu", index=0),
                torch.bfloat16,
                {"cp_size": 1, "enable_mla": False, "is_draft_engine": True},
            )

        assert isinstance(backend, StubAttentionBackend)
        # No DCP group and no MLA: the draft's pool stays an unsharded
        # full replica.
        assert "dcp_group" not in backend.init_kwargs
        assert backend.init_kwargs["is_mla"] is False
        sfa_module.SfaDcpAttentionBackend.assert_not_called()

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
            pytest.raises(NotImplementedError, match="Dense DCP attention is not supported"),
        ):
            _create_attention_backend(
                attn,
                MagicMock(type="npu", index=0),
                torch.bfloat16,
                {"cp_size": 1, "enable_mla": False, "is_draft_engine": False},
            )

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_qwen35_uses_dense_dcp_backend(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        dense_module = types.ModuleType("xllm.python.attention.dense_dcp_backend")
        dense_module.DenseDcpAttentionBackend = StubAttentionBackend

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {dense_module.__name__: dense_module}),
        ):
            backend = _create_attention_backend(
                attn,
                MagicMock(type="npu", index=0),
                torch.bfloat16,
                {
                    "model_type": "qwen3_5_moe_text",
                    "cp_size": 1,
                    "enable_mla": False,
                    "tp_size": 8,
                    "n_kv_heads": 2,
                },
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["dcp_group"] is dcp_group
        assert backend.init_kwargs["num_heads"] == 8
        assert backend.init_kwargs["num_kv_heads"] == 1
        assert backend.init_kwargs["head_dim"] == 128

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_qwen35_rejects_dcp_outside_kv_replica_group(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 4

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            pytest.raises(ValueError, match="replicated KV-head group"),
        ):
            _create_attention_backend(
                attn,
                MagicMock(type="npu", index=0),
                torch.bfloat16,
                {
                    "model_type": "qwen3_5_moe_text",
                    "cp_size": 1,
                    "enable_mla": False,
                    "tp_size": 8,
                    "n_kv_heads": 4,
                },
            )

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_qwen35_dcp_is_compatible_with_expert_parallelism(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        dense_module = types.ModuleType("xllm.python.attention.dense_dcp_backend")
        dense_module.DenseDcpAttentionBackend = StubAttentionBackend

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {dense_module.__name__: dense_module}),
        ):
            backend = _create_attention_backend(
                attn,
                MagicMock(type="npu", index=0),
                torch.bfloat16,
                {
                    "model_type": "qwen3_5_moe_text",
                    "cp_size": 1,
                    "tp_size": 8,
                    "n_kv_heads": 2,
                    "ep_size": 8,
                    "enable_eplb": True,
                },
            )

        assert isinstance(backend, StubAttentionBackend)

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_qwen35_dcp_rejects_disaggregated_pd(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            pytest.raises(NotImplementedError, match="disaggregated P/D"),
        ):
            _create_attention_backend(
                attn,
                MagicMock(type="npu", index=0),
                torch.bfloat16,
                {
                    "model_type": "qwen3_5_moe_text",
                    "cp_size": 1,
                    "tp_size": 8,
                    "n_kv_heads": 2,
                    "enable_disagg_pd": True,
                },
            )

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_decode_cp1_uses_sfa_dcp_backend(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = StubAttentionBackend
        sfa_module.dcp_layer_options = MagicMock(return_value=512)

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            backend = _create_attention_backend(
                attn,
                MagicMock(type="npu", index=0),
                torch.float16,
                {"num_speculative_tokens": 3, "cp_size": 1, "enable_mla": True},
                max_num_reqs=3,
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["dcp_group"] is dcp_group
        assert backend.init_kwargs["index_topk"] == 512
        assert backend.init_kwargs["max_num_reqs"] == 3
        assert backend.init_kwargs["num_decoding_tokens"] == 4

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=False,
    )
    @patch(
        "xllm.python.model_executor.executor.current_platform.is_cuda",
        return_value=True,
    )
    def test_cuda_device_creates_flashinfer_backend(self, _mock_is_cuda, _mock_is_npu):
        attn = _make_attention_layer()
        module = types.ModuleType("xllm.python.attention.flashinfer")
        module.FlashInferBackend = StubAttentionBackend
        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, torch.device("cuda"), torch.float16)
        assert isinstance(backend, StubAttentionBackend)


# ---------------------------------------------------------------------------
# Tests: ModelExecutor construction
# ---------------------------------------------------------------------------


class TestModelExecutorConstruction:
    @pytest.mark.parametrize(
        ("model_type", "jev_prompt_attention", "use_xfia_decode"),
        [
            ("dflash", None, False),
            ("dflash2", "bidirectional", True),
            ("qwen3_5", None, False),
            ("qwen3_5", "causal", True),
        ],
    )
    def test_non_jev_noncausal_attention_keeps_draft_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_type: str,
        jev_prompt_attention: str | None,
        use_xfia_decode: bool,
    ) -> None:
        pytest.importorskip("torch_npu")
        from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

        backend = NpuPagedAttentionBackend(
            num_heads=8,
            num_kv_heads=2,
            head_dim=128,
            scale=0.125,
            sliding_window=0,
            is_mla=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
            use_xfia_decode=use_xfia_decode,
            xfia_query_width=2,
        )
        model = _FakeModel(num_layers=1, head_dim=128)
        model.layers[0].causal = False
        # Mock only the device boundary; keep real executor initialization and backend state changes.
        parameter = SimpleNamespace(device=torch.device("privateuseone"), dtype=torch.float32)
        monkeypatch.setattr(model, "parameters", lambda: iter([parameter]))
        monkeypatch.setattr(
            sys.modules[ModelExecutor.__module__], "_create_attention_backend", lambda *args, **kwargs: backend
        )
        config = {"model_type": model_type, "enable_graph": False, "is_draft_engine": True}
        if jev_prompt_attention is not None:
            config["jev_prompt_attention"] = jev_prompt_attention
        executor = ModelExecutor(model, config, max_seqs_per_batch=1)

        assert executor.attention_backend is backend
        assert executor.eager_runner.attention_backend is backend
        assert not backend._prefix_bidir_enabled
        assert backend._use_xfia_decode is use_xfia_decode
        cache = torch.empty(1, 128, 2, 128)
        backend.bind_kv_caches([LayerCache(key=cache, value=torch.empty_like(cache))])
        metadata = SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=True,
            is_mixed=False,
            is_spec_verify=False,
            use_dflash_proposal_attention=True,
            block_table=torch.tensor([[0]], dtype=torch.int32),
            kv_seq_lens=torch.tensor([4], dtype=torch.int32),
            kv_seq_lens_host_values=[4],
            q_cu_seq_lens=torch.tensor([0, 2], dtype=torch.int32),
            slot_mapping=torch.tensor([2, 3], dtype=torch.int64),
        )
        backend.prepare(metadata)
        assert backend._metadata is metadata
        if use_xfia_decode:
            assert backend._xfia_kv_lengths.tolist() == [4, 4]
        else:
            assert backend._actual_seq_lens == [2]

    @pytest.mark.parametrize(
        ("model_type", "q_cu_lens"),
        [("qwen3_5", [2]), ("qwen3_5_text", [0, 2]), ("qwen3_5_moe", [2]), ("qwen3_5_moe_text", [0, 2])],
    )
    def test_jev_config_enables_prefix_bidir_and_rejects_chunked_prefill(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_type: str,
        q_cu_lens: list[int],
    ) -> None:
        pytest.importorskip("torch_npu")
        from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

        backend = NpuPagedAttentionBackend(8, 2, 64, 0.125, 0, False, torch.device("cpu"), torch.float32)
        model = _FakeModel(num_layers=1)
        model.layers[0].causal = False
        parameter = SimpleNamespace(device=torch.device("privateuseone"), dtype=torch.float32)
        monkeypatch.setattr(model, "parameters", lambda: iter([parameter]))
        monkeypatch.setattr(
            sys.modules[ModelExecutor.__module__], "_create_attention_backend", lambda *args, **kwargs: backend
        )
        executor = ModelExecutor(
            model,
            {"model_type": model_type, "jev_prompt_attention": "bidirectional", "enable_graph": False},
            max_seqs_per_batch=1,
        )

        assert executor.attention_backend is backend
        assert backend._prefix_bidir_enabled
        metadata = SimpleNamespace(
            is_prefill=True,
            is_chunked_prefill=False,
            is_mixed=False,
            is_spec_verify=False,
            block_table=None,
            kv_seq_lens=None,
            q_cu_seq_lens_host_values=q_cu_lens,
            kv_seq_lens_host_values=[2],
        )
        backend.prepare(metadata)
        assert backend._metadata is metadata
        assert backend._actual_seq_lens == [2]
        metadata.is_chunked_prefill = True
        with pytest.raises(ValueError, match="pure, non-speculative full prefill"):
            backend.prepare(metadata)

    def test_jev_prefix_bidir_accepts_dummy_prefill_host_geometry(self) -> None:
        pytest.importorskip("torch_npu")
        from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

        backend = NpuPagedAttentionBackend(8, 2, 64, 0.125, 0, False, torch.device("cpu"), torch.float32)
        backend.enable_prefix_bidir()
        metadata = SimpleNamespace(
            is_dummy=True,
            is_prefill=True,
            is_chunked_prefill=False,
            is_mixed=False,
            is_spec_verify=False,
            block_table=torch.tensor([[0]], dtype=torch.int32),
            kv_seq_lens=torch.tensor([1], dtype=torch.int32),
            q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
            q_cu_seq_lens_host_values=[0, 1],
            kv_seq_lens_host_values=[1],
            block_size=128,
        )
        backend.prepare(metadata)
        assert backend._metadata is metadata
        assert backend._actual_seq_lens == [1]

    @pytest.mark.parametrize("q_cu_lens", [[], None])
    def test_jev_prefix_bidir_rejects_empty_query_cumulative_lengths(self, q_cu_lens: list[int] | None) -> None:
        pytest.importorskip("torch_npu")
        from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

        metadata = SimpleNamespace(
            is_prefill=True,
            is_chunked_prefill=False,
            is_mixed=False,
            is_spec_verify=False,
            q_cu_seq_lens_host_values=q_cu_lens,
            q_seq_lens_host=torch.tensor([2], dtype=torch.int32),
            kv_seq_lens_host_values=[2],
        )
        with pytest.raises(ValueError, match="non-empty host cumulative query lengths"):
            NpuPagedAttentionBackend._validate_prefix_bidir(metadata)

    @pytest.mark.parametrize(
        ("enable_graph", "config_depth"),
        [(False, 0), (True, 3)],
    )
    @patch("xllm.python.model_executor.executor.current_platform.is_npu", new=lambda: True)
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_decoding_width_reaches_backend_and_graph_runner(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
        enable_graph: bool,
        config_depth: int,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        ModelExecutor(
            _FakeModel(num_layers=1),
            {
                "enable_graph": enable_graph,
                "num_speculative_tokens": config_depth,
                "max_position_embeddings": 128,
            },
            max_seqs_per_batch=4,
        )

        if enable_graph:
            assert mock_graph_runner.call_args.kwargs["num_speculative_tokens"] == config_depth
        else:
            mock_graph_runner.assert_not_called()

    @pytest.mark.parametrize("enable_graph", [False, True])
    @patch("xllm.python.model_executor.executor.current_platform.is_npu", new=lambda: True)
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_block_draft_width_is_independent_of_target_verify_width(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
        enable_graph: bool,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(
            model,
            {
                "is_draft_engine": True,
                "enable_graph": enable_graph,
                "num_speculative_tokens": 15,
                "model_type": "DSparkDraftModel",
                "max_position_embeddings": 128,
            },
            max_seqs_per_batch=4,
        )
        assert executor.eager_runner.model is model.model
        if enable_graph:
            assert mock_graph_runner.call_args.args[0] is model.model
            assert mock_graph_runner.call_args.kwargs["num_speculative_tokens"] == 15

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_no_attention_layers_raises(self, _mock_backend):
        model = _FakeModelNoAttention()
        with pytest.raises(ValueError, match="does not contain an Attention layer"):
            ModelExecutor(model, {}, max_seqs_per_batch=4)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_heterogeneous_attention_raises(self, _mock_backend):
        model = _FakeModelHeterogeneous()
        with pytest.raises(ValueError, match="identical attention configuration"):
            ModelExecutor(model, {}, max_seqs_per_batch=4)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_hybrid_attention_builds_paged_backend_from_mla(self, mock_backend: MagicMock) -> None:
        model = _FakeModel(num_layers=2)
        model.layers[0].attention_kind = "linear"
        model.layers[1].attention_kind = "mla"
        model.layers[0].head_dim = 128
        model.layers[1].head_dim = 512
        ModelExecutor(model, {"enable_graph": False}, max_seqs_per_batch=4)
        assert mock_backend.call_args.args[0] is model.layers[1]

    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_hybrid_attention_still_validates_mla_configuration(self, mock_backend: MagicMock) -> None:
        model = _FakeModel(num_layers=3)
        model.layers[0].attention_kind = "linear"
        for layer in model.layers[1:]:
            layer.attention_kind = "mla"
        model.layers[2].head_dim *= 2
        with pytest.raises(ValueError, match="identical MLA configuration"):
            ModelExecutor(model, {"enable_graph": False}, max_seqs_per_batch=4)
        mock_backend.assert_not_called()

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_glm_next_decode_cp1_selects_sfa_dcp_backend_from_dsa_layer(
        self,
        _mock_is_npu: MagicMock,
    ) -> None:
        """Mixed GLM layers select the DCP backend using real DSA layer options."""

        class _DsaAttention(Attention):
            is_glm_next_mla = True

        class _KdaAttention(Attention):
            is_glm_next_kda = True

        class _FakeGlmNextDecode(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(1, 1)
                # KDA (linear attention) layers come first in glm5_next with a
                # different head config; the backend must be built from the
                # first DSA layer instead.
                self.kda = _KdaAttention(
                    num_heads=16,
                    num_kv_heads=16,
                    head_dim=64,
                    scale=0.125,
                    sliding_window=0,
                    layer_id=0,
                )
                self.dsa = _DsaAttention(
                    num_heads=64,
                    num_kv_heads=1,
                    head_dim=512,
                    scale=0.044,
                    sliding_window=0,
                    layer_id=3,
                )
                self.dsa.cfg = SimpleNamespace(index_topk=2048)
                self._param = nn.Parameter(torch.zeros(1))

        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        dcp_group.rank.return_value = 0
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = StubAttentionBackend
        # The real option resolution, not a stub: index_topk must come from
        # the DSA layer's cfg.
        sfa_module.dcp_layer_options = dcp_layer_options

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            executor = ModelExecutor(
                _FakeGlmNextDecode(),
                {
                    "model_type": "glm5_next",
                    "cp_size": 1,
                    "enable_mla": True,
                    "enable_graph": False,
                },
                max_seqs_per_batch=4,
            )

        backend = executor.attention_backend
        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["dcp_group"] is dcp_group
        assert backend.init_kwargs["index_topk"] == 2048
        # The GLOBAL head count flows in from the DSA layer's Attention base
        # (glm5_next convention); the SFA merge derives per-rank heads from
        # output shapes instead, so this value is inert on the decode path.
        assert backend.init_kwargs["num_heads"] == 64
        assert backend.init_kwargs["num_kv_heads"] == 1
        assert backend.init_kwargs["head_dim"] == 512
        assert backend.init_kwargs["max_num_reqs"] == 4
        assert backend.init_kwargs["num_decoding_tokens"] == 1

    @patch("xllm.python.model_executor.executor.current_platform.is_npu", new=lambda: False)
    @patch("xllm.python.model_executor.executor.current_platform.is_cuda", new=lambda: True)
    @patch("xllm.python.model_executor.runners.decode_cuda_graph.DecodeCudaGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_data_parallel_cuda_graph_is_supported(self, mock_create, mock_graph_runner):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "dp_size": 2,
                "dp_rank": 1,
                "max_position_embeddings": 128,
                "enable_graph": True,
            },
            max_seqs_per_batch=4,
        )

        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            4,
            128,
            2,
            1,
        )

    @pytest.mark.parametrize("is_spec_draft", [False, True])
    @patch("xllm.python.model_executor.executor.current_platform.is_npu", new=lambda: True)
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_capacity_respects_decode_batch_limit(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
        is_spec_draft: bool,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "is_draft_engine": is_spec_draft,
                "num_speculative_tokens": 3,
                "max_position_embeddings": 128,
                "enable_graph": True,
            },
            max_seqs_per_batch=256,
            acl_graph_decode_batch_size_limit=16,
        )

        # PR3 passes max_seqs_per_batch (sequences) directly as the runner's
        # max_batch capacity; the decode_batch_size_limit is enforced at
        # runtime by can_execute (bucket > limit falls back to eager), not by
        # folding it into max_batch at construction.
        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            256,
            128,
            1,
            0,
            16,
            num_speculative_tokens=3,
            enable_mega_moe_token_mask=False,
            is_spec_draft=is_spec_draft,
            model_type="",
            capture_sizes=(),
        )

    @patch("xllm.python.model_executor.executor.current_platform.is_npu", new=lambda: True)
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_glm_mega_moe_enables_legacy_token_mask(
        self,
        mock_create,
        mock_graph_runner,
    ):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "enable_mega_moe": True,
                "model_type": "glm_moe_dsa",
                "max_position_embeddings": 128,
                "enable_graph": True,
            },
            max_seqs_per_batch=4,
        )

        assert mock_graph_runner.call_args.kwargs["enable_mega_moe_token_mask"] is True

    @pytest.mark.parametrize(
        ("model_type", "degree", "ep_size", "expected", "enable_mega_moe"),
        [
            ("glm5_next", 2, 8, True, True),
            ("glm5_next_text", 2, 2, True, False),
            ("glm5_next", 1, 8, False, True),
            ("glm5_next", 0, 1, False, False),
            ("qwen3_5_moe", 2, 8, False, True),
            ("qwen3_5_moe", 2, 8, False, False),
        ],
    )
    @patch("xllm.python.model_executor.executor.current_platform.is_npu", new=lambda: True)
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_glm_eplv2_skips_legacy_megamoe_mask(
        self, mock_create, mock_graph_runner, model_type, degree, ep_size, expected, enable_mega_moe
    ):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(
            model,
            {
                "model_type": model_type,
                "expert_parallel_degree": degree,
                "ep_size": ep_size,
                "enable_mega_moe": enable_mega_moe,
                "hidden_size": 8,
                "num_experts_per_tok": 2,
                "mega_moe_num_max_tokens_per_rank": 4,
                "max_position_embeddings": 128,
                "enable_graph": True,
            },
            max_seqs_per_batch=16,
            acl_graph_decode_batch_size_limit=16,
        )
        expected_builder = expected or (model_type == "qwen3_5_moe" and enable_mega_moe)
        assert bool(executor._execution_metadata_builders) is expected_builder
        assert bool(executor.eager_runner.execution_metadata_builders) is expected_builder
        mock_graph_runner.return_value.bind_execution_metadata_builders.assert_called_once_with(
            executor.eager_runner.execution_metadata_builders
        )
        assert mock_graph_runner.call_args.kwargs["enable_mega_moe_token_mask"] is (enable_mega_moe and not expected)

    @pytest.mark.parametrize(
        ("model_type", "tp_size", "mc2_capacity", "speculative_tokens"),
        [("glm5_next", 8, 3, 7), ("qwen3_5_moe", 4, 86, 0)],
    )
    @patch("xllm.python.model_executor.executor.current_platform.is_npu", new=lambda: True)
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_eplv2_graph_uses_model_mc2_capacity(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
        model_type: str,
        tp_size: int,
        mc2_capacity: int,
        speculative_tokens: int,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        model.model._comm_policy = Eplv2CommPolicy(mc2_capacity=mc2_capacity)
        ModelExecutor(
            model,
            {
                "model_type": model_type,
                "expert_parallel_degree": 2,
                "tp_size": tp_size,
                "ep_size": tp_size,
                "num_speculative_tokens": speculative_tokens,
                "max_position_embeddings": 128,
                "enable_graph": True,
            },
            max_seqs_per_batch=128,
        )
        assert mock_graph_runner.call_args.kwargs["eplv2_graph_token_limit"] == mc2_capacity * tp_size
        assert mock_graph_runner.call_args.kwargs["num_speculative_tokens"] == speculative_tokens

    @pytest.mark.parametrize("mc2_capacity", [8])
    @patch("xllm.python.model_executor.executor.current_platform.is_npu", new=lambda: True)
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_glm_eplv2_graph_uses_megamoe_cap_instead_of_mc2(
        self, mock_create: MagicMock, mock_graph_runner: MagicMock, mc2_capacity: int
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        model.model._comm_policy = Eplv2CommPolicy(mc2_capacity=mc2_capacity, mega_moe_capacity=4)
        ModelExecutor(
            model,
            {
                "model_type": "glm5_next",
                "expert_parallel_degree": 2,
                "tp_size": 8,
                "ep_size": 8,
                "enable_mega_moe": True,
                "max_position_embeddings": 128,
                "enable_graph": True,
            },
            max_seqs_per_batch=16,
        )
        assert mock_graph_runner.call_args.kwargs["eplv2_graph_token_limit"] == 32
        assert mock_graph_runner.call_args.kwargs["enable_mega_moe_token_mask"] is False

    @patch("xllm.python.model_executor.executor.current_platform.is_npu", new=lambda: True)
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_data_parallel_passes_global_token_budget(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "num_speculative_tokens": 3,
                "dp_size": 16,
                "dp_rank": 0,
                "enable_mega_moe": True,
                "model_type": "glm_moe_dsa",
                "max_position_embeddings": 128,
                "enable_graph": True,
            },
            max_seqs_per_batch=17,
        )

        # PR3 passes max_seqs_per_batch (sequences) directly as the runner's
        # max_batch capacity; the runner derives its local DP capacity with
        # ceil-div internally.
        assert mock_create.call_args.args[4] == 17
        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            17,
            128,
            16,
            0,
            None,
            num_speculative_tokens=3,
            enable_mega_moe_token_mask=True,
            is_spec_draft=False,
            model_type="",
            capture_sizes=(),
        )

    @patch("xllm.python.model_executor.executor.current_platform.is_npu", new=lambda: True)
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_forwards_nondefault_draft_query_width(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "is_draft_engine": True,
                "max_position_embeddings": 128,
                "enable_graph": True,
                "model_type": "DFlash2DraftModel",
                "num_speculative_tokens": 7,
                "dflash2_selector_top_k": 4,
            },
            max_seqs_per_batch=8,
        )

        assert mock_graph_runner.call_args.kwargs["num_speculative_tokens"] == 7


class TestDecodeCudaGraphDataParallelKeys:
    @staticmethod
    def _runner(dp_rank: int = 0) -> DecodeCudaGraphRunner:
        return DecodeCudaGraphRunner(
            nn.Identity(),
            StubAttentionBackend(),
            torch.device("cpu"),
            max_batch=16,
            max_model_len=128,
            dp_size=2,
            dp_rank=dp_rank,
        )

    @staticmethod
    def _metadata(
        token_counts: list[int] | tuple[int, ...] = (),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            dp_execution_token_counts=tuple(token_counts),
        )

    def test_graph_key_uses_global_max_data_parallel_bucket(self):
        runner = self._runner()
        input_ids = torch.zeros(3, dtype=torch.int32)

        first = runner._graph_key(input_ids, self._metadata([3, 1]))
        second = runner._graph_key(input_ids, self._metadata([3, 2]))

        assert first == (4, (4, 4))
        assert second == first

    def test_data_parallel_warmup_uses_local_batch_capacity(self):
        assert _decode_graph_buckets(16, 2) == [1, 2, 4, 8]
        assert _decode_graph_buckets(20, 2) == [1, 2, 4, 8, 16]

    def test_single_rank_graph_key_reuses_padded_bucket(self):
        runner = self._runner()
        runner.dp_size = 1
        runner.dp_rank = 0

        first = runner._graph_key(torch.zeros(3, dtype=torch.int32), self._metadata([3]))
        second = runner._graph_key(torch.zeros(4, dtype=torch.int32), self._metadata([4]))

        assert first == (4, (4,))
        assert second == first

    def test_graph_key_accepts_dummy_execution_row_for_empty_data_parallel_rank(self):
        runner = self._runner(dp_rank=1)
        input_ids = torch.zeros(1, dtype=torch.int32)

        assert runner._graph_key(input_ids, self._metadata([5, 1])) == (
            8,
            (8, 8),
        )

    def test_graph_key_rejects_unbalanced_unwarmed_bucket(self):
        runner = self._runner()
        input_ids = torch.zeros(9, dtype=torch.int32)

        assert runner._graph_key(input_ids, self._metadata([9, 7])) is None

    def test_can_execute_requires_warmed_graph(self):
        runner = self._runner()
        input_ids = torch.zeros(3, dtype=torch.int32)
        metadata = self._metadata([3, 1])
        graph_key = runner._graph_key(input_ids, metadata)

        assert not runner.can_execute(ModelInput(input_ids, torch.zeros_like(input_ids), metadata))
        runner._graphs[graph_key] = object()
        assert runner.can_execute(ModelInput(input_ids, torch.zeros_like(input_ids), metadata))

    @pytest.mark.parametrize("allowed", [False, True])
    def test_model_admission_is_honored(self, allowed: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = self._runner()
        input_ids = torch.zeros(3, dtype=torch.int32)
        metadata = self._metadata([3, 1])
        runner._graphs[runner._graph_key(input_ids, metadata)] = object()
        admission = MagicMock(return_value=allowed)
        monkeypatch.setattr(runner.model, "can_execute_graph", admission, raising=False)
        agreement = MagicMock(return_value=allowed)
        monkeypatch.setattr(distributed, "dp_all_true", agreement, raising=False)
        inputs = ModelInput(input_ids, torch.zeros_like(input_ids), metadata)
        assert runner.can_execute(inputs) is allowed
        admission.assert_called_once_with(inputs)
        agreement.assert_called_once_with(allowed, runner.device, 2)

    @pytest.mark.parametrize("context_name", ["eplb", "multimodal_inputs"])
    def test_unsupported_graph_context_is_rejected(self, context_name: str) -> None:
        runner = self._runner()
        input_ids = torch.zeros(3, dtype=torch.int32)
        metadata = self._metadata([3, 1])
        runner._graphs[runner._graph_key(input_ids, metadata)] = object()
        assert not runner.can_execute(
            ModelInput(
                input_ids,
                torch.zeros_like(input_ids),
                metadata,
                **{
                    context_name: (
                        EplbRuntimeState(torch.zeros(3), decode_token_mask=None, is_graph_warmup=False)
                        if context_name == "eplb"
                        else MultimodalInputs(torch.zeros(3, 64))
                    )
                },
            )
        )

    def test_remote_model_veto_rejects_local_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = self._runner()
        monkeypatch.setattr(runner.model, "can_execute_graph", MagicMock(return_value=True), raising=False)
        input_ids = torch.zeros(3, dtype=torch.int32)
        metadata = self._metadata([3, 1])
        runner._graphs[runner._graph_key(input_ids, metadata)] = object()
        agreement = MagicMock(return_value=False)
        monkeypatch.setattr(distributed, "dp_all_true", agreement, raising=False)
        assert not runner.can_execute(ModelInput(input_ids, torch.zeros_like(input_ids), metadata))
        agreement.assert_called_once_with(True, runner.device, 2)

    @pytest.mark.parametrize("token_counts", ([3], [3, 0], [3, 2]))
    def test_graph_key_rejects_invalid_data_parallel_metadata(self, token_counts):
        runner = self._runner(dp_rank=1)
        input_ids = torch.zeros(1, dtype=torch.int32)

        with pytest.raises(RuntimeError):
            runner._graph_key(input_ids, self._metadata(token_counts))


# ---------------------------------------------------------------------------
# Tests: DecodeAclGraphRunner speculative metadata
# ---------------------------------------------------------------------------


class TestDecodeAclGraphSpeculativeMetadata:
    @staticmethod
    def _runner(*, enable_mega_moe_token_mask: bool = False) -> DecodeAclGraphRunner:
        return DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=4,
            max_model_len=8,
            enable_mega_moe_token_mask=enable_mega_moe_token_mask,
        )

    @staticmethod
    def _metadata() -> SimpleNamespace:
        return SimpleNamespace(
            slot_mapping=torch.arange(4, dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32),
            paged_kv_indices=torch.tensor([10, 10, 20, 21, 20, 21], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([3, 4, 3, 4], dtype=torch.int32),
            q_cu_seq_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
            kv_cu_seq_lens=torch.tensor([0, 4, 12], dtype=torch.int32),
            kv_seq_lens_host=torch.tensor([4, 8], dtype=torch.int32),
            kv_seq_lens_host_values=[4, 8],
            block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            kv_seq_lens=torch.tensor([4, 8], dtype=torch.int32),
            q_seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            expanded_decode_metadata=SimpleNamespace(
                enabled=True,
                kv_seq_lens=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
                block_table=torch.tensor(
                    [[10, 11], [10, 11], [20, 21], [20, 21]],
                    dtype=torch.int32,
                ),
                paged_kv_indptr=torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32),
                paged_kv_indices=torch.tensor([10, 10, 20, 21, 20, 21], dtype=torch.int32),
                paged_kv_last_page_len=torch.tensor([3, 4, 3, 4], dtype=torch.int32),
                paged_attention_tiling_data=None,
                kv_seq_lens_host=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
                kv_seq_lens_host_values=[3, 4, 7, 8],
            ),
            is_prefill=False,
            is_chunked_prefill=True,
        )

    @staticmethod
    def _decode_metadata(
        batch_size: int,
        *,
        dp_execution_token_counts: tuple[int, ...] = (),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            slot_mapping=torch.arange(batch_size, dtype=torch.int32),
            paged_kv_indptr=torch.arange(batch_size + 1, dtype=torch.int32),
            paged_kv_indices=torch.arange(batch_size, dtype=torch.int32),
            paged_kv_last_page_len=torch.ones(batch_size, dtype=torch.int32),
            kv_cu_seq_lens=torch.arange(batch_size + 1, dtype=torch.int32),
            kv_seq_lens_host_values=[1] * batch_size,
            q_cu_seq_lens=None,
            block_table=torch.arange(batch_size, dtype=torch.int32).view(-1, 1),
            kv_seq_lens=torch.ones(batch_size, dtype=torch.int32),
            expanded_decode_metadata=None,
            dp_execution_token_counts=dp_execution_token_counts,
            is_prefill=False,
            is_chunked_prefill=False,
        )

    def test_expanded_metadata_selects_matching_paged_kv_rows(self) -> None:
        runner = self._runner()
        (
            block_table,
            kv_seq_lens,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = runner._decode_metadata(self._metadata())

        assert block_table.tolist() == [
            [10, 11],
            [10, 11],
            [20, 21],
            [20, 21],
        ]
        assert kv_seq_lens.tolist() == [3, 4, 7, 8]
        assert paged_kv_indptr.tolist() == [0, 1, 2, 4, 6]
        assert paged_kv_indices.tolist() == [10, 10, 20, 21, 20, 21]
        assert paged_kv_last_page_len.tolist() == [3, 4, 3, 4]

    def test_decode_metadata_rebuilds_token_row_paging_metadata(self) -> None:
        metadata = self._metadata()
        metadata.expanded_decode_metadata = None
        metadata.is_chunked_prefill = False
        metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
        metadata.block_table = torch.tensor(
            [[10, 11], [10, 11], [20, 21], [20, 21]],
            dtype=torch.int32,
        )
        metadata.kv_seq_lens = torch.tensor([3, 4, 7, 8], dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [3, 4, 7, 8]
        # Mark this as a GENERIC (untyped) MTP verify batch: 4 token rows back
        # 2 logical sequences (width=2). _is_untyped_spec_verify then admits
        # it and _expanded_verify_view synthesizes the token-row expanded
        # view, rebuilding the per-seq paging metadata per row.
        metadata.linear_state_indices = torch.arange(2, dtype=torch.int32)
        metadata.q_cu_seq_lens = None
        metadata.paged_kv_indptr = torch.tensor([0, 1, 3], dtype=torch.int32)
        metadata.paged_kv_indices = torch.tensor([10, 20, 21], dtype=torch.int32)
        metadata.paged_kv_last_page_len = torch.tensor([4, 4], dtype=torch.int32)

        (
            _,
            _,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._runner()._decode_metadata(metadata)

        assert paged_kv_indptr.tolist() == [0, 1, 2, 4, 6]
        assert paged_kv_indices.tolist() == [10, 10, 20, 21, 20, 21]
        assert paged_kv_last_page_len.tolist() == [3, 4, 3, 4]

    def test_decode_batch_limit_gates_dp_global_token_bucket(self) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            dp_size=2,
            dp_rank=0,
            decode_batch_size_limit=16,
        )
        for counts, expected in [((8, 8), True), ((8, 64), False)]:
            metadata = self._decode_metadata(8, dp_execution_token_counts=counts)
            metadata.dp_is_decode = (1, 1)
            inputs = ModelInput(torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int32), metadata)
            assert runner.can_execute(inputs) is expected

    @pytest.mark.parametrize(
        ("dp_size", "rows", "limit", "expected"),
        [(1, 336, 344, True), (1, 337, 344, False), (2, 336, 344, True), (2, 337, 344, False), (2, 1, 0, False)],
    )
    def test_eplv2_graph_admission_uses_padded_ep_group_capacity(
        self, dp_size: int, rows: int, limit: int, expected: bool
    ) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=512,
            max_model_len=8,
            dp_size=dp_size,
            dp_rank=0,
            eplv2_graph_token_limit=limit,
        )
        # 337 rows pad to352 >344. An empty rank still follows its busy peer.
        local_rows = rows if dp_size == 1 else 1
        counts = (rows,) if dp_size == 1 else (1, rows)
        metadata = self._decode_metadata(local_rows, dp_execution_token_counts=counts)
        metadata.raw_dp_execution_token_counts = (rows,) if dp_size == 1 else (0, rows)
        metadata.dp_is_decode = (1,) * dp_size
        inputs = ModelInput(
            torch.zeros(local_rows, dtype=torch.int32), torch.zeros(local_rows, dtype=torch.int32), metadata
        )
        assert runner.can_execute(inputs) is expected

    def test_mtp3_batch_eight_uses_32_row_graph_bucket(self) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            decode_batch_size_limit=16,
            num_speculative_tokens=3,
        )
        # 8 logical sequences x 4 verify rows (mtp3) = 32 token rows. The
        # graph admits on the SEQUENCE count (8, bucket 8 <= limit 16) while
        # capturing/replaying at the full 32-row token bucket.
        metadata = SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=True,
            is_spec_verify=True,
            slot_mapping=torch.arange(32, dtype=torch.int32),
            kv_seq_lens=torch.full((8,), 8, dtype=torch.int32),
            block_table=torch.zeros((8, 2), dtype=torch.int32),
            linear_state_indices=torch.arange(8, dtype=torch.int32),
            q_cu_seq_lens=torch.arange(0, 33, 4, dtype=torch.int32),
            expanded_decode_metadata=SimpleNamespace(
                kv_seq_lens=torch.full((32,), 8, dtype=torch.int32),
                block_table=torch.zeros((32, 2), dtype=torch.int32),
                paged_kv_indptr=None,
                paged_kv_indices=None,
                paged_kv_last_page_len=None,
                paged_attention_tiling_data=None,
                kv_seq_lens_host=None,
                kv_seq_lens_host_values=[8] * 32,
            ),
        )
        assert runner.can_execute(
            ModelInput(torch.zeros(32, dtype=torch.int32), torch.zeros(32, dtype=torch.int32), metadata)
        )

    def test_warmup_validates_capacity_without_capturing(self) -> None:
        runner = self._runner()
        input_ids = torch.arange(4, dtype=torch.int32)
        positions = torch.arange(4, dtype=torch.int32)
        metadata = self._metadata()

        with patch.object(runner, "_allocate_entry") as allocate:
            runner.warmup(input_ids, positions, metadata)
            allocate.assert_not_called()
            runner.warmup(input_ids, positions, metadata)
            allocate.assert_not_called()

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            (
                "block_table",
                torch.arange(8, dtype=torch.int32),
                "block_table must be two-dimensional",
            ),
            (
                "kv_seq_lens",
                torch.tensor([3, 4, 7], dtype=torch.int32),
                "kv_seq_lens must contain one value per sequence",
            ),
            (
                "kv_seq_lens_host",
                torch.tensor([3, 4, 7], dtype=torch.int32),
                "kv_seq_lens_host must contain one value per sequence",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 1, 2, 4], dtype=torch.int32),
                "paged_kv_indptr must contain one offset per sequence",
            ),
            (
                "paged_kv_indices",
                torch.tensor([[10, 10], [20, 21]], dtype=torch.int32),
                "paged_kv_indices must be a non-empty flat page list",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 3], dtype=torch.int32),
                "paged_kv_last_page_len must contain one value per sequence",
            ),
        ],
    )
    def test_expanded_metadata_shape_mismatch_fails(
        self,
        field: str,
        value: torch.Tensor,
        message: str,
    ) -> None:
        metadata = self._metadata()
        setattr(metadata.expanded_decode_metadata, field, value)

        with pytest.raises(RuntimeError, match=message):
            self._runner()._decode_metadata(metadata)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            (
                "paged_kv_indptr",
                torch.tensor([1, 1, 2, 4, 6], dtype=torch.int32),
                "must start at zero",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 2, 1, 4, 6], dtype=torch.int32),
                "must be monotonic",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 1, 2, 4, 5], dtype=torch.int32),
                "terminal page offset must match page count",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 0, 4], dtype=torch.int32),
                "last-page lengths must be positive",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 5, 4], dtype=torch.int32),
                "must not exceed block size",
            ),
        ],
    )
    def test_expanded_paged_metadata_invariant_fails(
        self,
        field: str,
        value: torch.Tensor,
        message: str,
    ) -> None:
        metadata = self._metadata()
        setattr(metadata.expanded_decode_metadata, field, value)

        with pytest.raises(RuntimeError, match=message):
            self._runner()._decode_metadata(metadata)

    def test_expanded_page_count_exceeding_capacity_fails(self) -> None:
        metadata = self._metadata()
        expanded = metadata.expanded_decode_metadata
        expanded.kv_seq_lens_host_values = [3, 4, 7, 9]
        expanded.kv_seq_lens = torch.tensor([3, 4, 7, 9], dtype=torch.int32)
        expanded.kv_seq_lens_host = expanded.kv_seq_lens.clone()
        expanded.paged_kv_indptr = None
        expanded.paged_kv_indices = None
        expanded.paged_kv_last_page_len = None

        with pytest.raises(RuntimeError, match="exceeds block-table capacity"):
            self._runner()._decode_metadata(metadata)

    @pytest.mark.parametrize(
        ("input_ids", "slot_mapping", "message"),
        [
            (
                torch.arange(3, dtype=torch.int32),
                torch.arange(4, dtype=torch.int32),
                "input_ids must contain one token per sequence",
            ),
            (
                torch.arange(4, dtype=torch.int32),
                torch.arange(3, dtype=torch.int32),
                "slot_mapping must contain one slot per token",
            ),
        ],
    )
    def test_token_layout_mismatch_fails(
        self,
        input_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(RuntimeError, match=message):
            self._runner()._validate_decode_token_layout(
                input_ids,
                torch.arange(4, dtype=torch.int32),
                slot_mapping,
                sequence_count=4,
            )

    def test_replay_returns_static_output_view(self) -> None:
        runner = self._runner()
        refresh_dsa = MagicMock()
        runner.attention_backend.refresh_dsa_metadata_for_graph_replay = refresh_dsa
        batch_size = 3
        padded_batch_size = 4
        static_output = torch.arange(12).reshape(padded_batch_size, 3)
        graph = MagicMock()
        metadata = self._decode_metadata(batch_size)
        entry = runner._allocate_entry(
            padded_batch_size,
            torch.arange(batch_size, dtype=torch.int32),
            torch.arange(batch_size, dtype=torch.int32),
            metadata,
        )
        entry.graph = graph
        entry.static_output = static_output
        graph_key = runner._graph_key(
            padded_batch_size,
            is_expanded=False,
            input_embedding=None,
        )
        runner._graphs[graph_key] = entry

        replay_stream = MagicMock()
        update_stream = MagicMock()
        replay_done_event = MagicMock()
        current_stream = MagicMock()
        runner._stream = replay_stream
        runner._update_stream = update_stream
        runner._replay_done_event = replay_done_event
        fake_npu = SimpleNamespace(
            current_stream=MagicMock(return_value=current_stream),
            stream=MagicMock(return_value=nullcontext()),
        )

        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(runner, "_fill_entry"),
        ):
            output = runner.execute(
                torch.arange(batch_size, dtype=torch.int32),
                torch.arange(batch_size, dtype=torch.int32),
                metadata,
            )

        assert output.shape == (batch_size, 3)
        assert output.data_ptr() == static_output.data_ptr()
        output[0, 0] = -1
        assert static_output[0, 0].item() == -1
        replay_stream.wait_stream.assert_called_once_with(current_stream)
        current_stream.wait_stream.assert_called_once_with(replay_stream)
        graph.replay.assert_called_once_with()
        refresh_dsa.assert_called_once_with(entry.static_metadata)

    def test_graph_token_mask_updates_in_place_for_reused_bucket(self) -> None:
        runner = self._runner(enable_mega_moe_token_mask=True)
        metadata = self._decode_metadata(4)
        entry = runner._allocate_entry(
            4,
            torch.arange(4, dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
            metadata,
        )
        mask = entry.static_metadata.mega_moe_token_mask
        assert mask is not None
        data_ptr = mask.data_ptr()

        runner._fill_mega_moe_token_mask(entry, metadata, batch_size=4)
        assert mask.tolist() == [1, 1, 1, 1]

        runner._fill_mega_moe_token_mask(
            entry,
            self._decode_metadata(3),
            batch_size=3,
        )
        assert mask.data_ptr() == data_ptr
        assert mask.tolist() == [1, 1, 1, 0]

    def test_graph_token_mask_uses_real_data_parallel_counts(self) -> None:
        # dp_execution_token_counts floors an empty DP rank to 1 (its
        # worker-materialized dummy row), so a rank contributing exactly 1
        # here also covers the empty-rank case: MegaMoe dispatches that row
        # like a real token, and the mask marks it active accordingly.
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            enable_mega_moe_token_mask=True,
        )
        metadata = self._decode_metadata(3, dp_execution_token_counts=(3, 1))
        entry = runner._allocate_entry(
            4,
            torch.arange(3, dtype=torch.int32),
            torch.arange(3, dtype=torch.int32),
            metadata,
        )

        runner._fill_mega_moe_token_mask(entry, metadata, batch_size=3)

        mask = entry.static_metadata.mega_moe_token_mask
        assert mask is not None
        assert mask.tolist() == [1, 1, 1, 0, 1, 0, 0, 0]


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.bind_kv_caches
# ---------------------------------------------------------------------------


class TestNormalizeLayerCaches:
    def test_legacy_five_slot_cache_keeps_generic_layout(self):
        tensors = tuple(torch.full((1,), value) for value in range(1, 6))

        cache = normalize_layer_caches([tensors])[0]

        assert cache.key is tensors[0]
        assert cache.value is tensors[1]
        assert cache.index is tensors[2]
        assert cache.conv is tensors[3]
        assert cache.ssm is tensors[4]
        assert cache.swa is None
        assert cache.compress_kv_state is None
        assert cache.compress_score_state is None
        assert cache.compress_index_kv_state is None
        assert cache.compress_index_score_state is None
        assert cache.indexer_scale is None

    def test_deepseek_v4_eleven_slot_cache_maps_all_slots(self):
        tensors = tuple(torch.full((1,), value) for value in range(1, 12))

        cache = normalize_layer_caches([tensors])[0]

        mapped = (
            cache.key,
            cache.value,
            cache.index,
            cache.conv,
            cache.ssm,
            cache.swa,
            cache.compress_kv_state,
            cache.compress_score_state,
            cache.compress_index_kv_state,
            cache.compress_index_score_state,
            cache.indexer_scale,
        )
        assert all(actual is source for actual, source in zip(mapped, tensors, strict=True))

    def test_empty_deepseek_v4_slots_are_normalized_to_none(self) -> None:
        key, value = torch.ones(1), torch.full((1,), 2.0)
        cache = normalize_layer_caches([(key, value, *(torch.empty(0),) * 9)])[0]
        assert cache.key is key
        assert cache.value is value
        assert all(
            slot is None
            for slot in (
                cache.index,
                cache.conv,
                cache.ssm,
                cache.swa,
                cache.compress_kv_state,
                cache.compress_score_state,
                cache.compress_index_kv_state,
                cache.compress_index_score_state,
                cache.indexer_scale,
            )
        )


class TestBindKvCaches:
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_wrong_count_raises(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=2)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        with pytest.raises(ValueError, match="layer count does not match"):
            executor.bind_kv_caches([kv])

    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_bind_idempotent(self, mock_create: MagicMock) -> None:
        backend = StubAttentionBackend()
        backend.bind_kv_caches = MagicMock(wraps=backend.bind_kv_caches)
        mock_create.return_value = backend
        executor = ModelExecutor(_FakeModel(num_layers=1), {}, max_seqs_per_batch=4)
        kv = (torch.zeros(2, 1, 2, 64), torch.ones(2, 1, 2, 64))
        executor.bind_kv_caches([kv])
        executor.bind_kv_caches([(torch.full_like(kv[0], 3), torch.full_like(kv[1], 4))])
        backend.bind_kv_caches.assert_called_once()
        assert backend._kv_caches[0].key is kv[0]
        assert backend._kv_caches[0].value is kv[1]

    @patch("xllm.python.model_executor.executor.get_execution_metadata_builder_classes")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_bind_forwards_layer_caches_to_cache_aware_builder(
        self, mock_create: MagicMock, mock_builder_classes: MagicMock
    ) -> None:
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        builder = Csa2BlockAttentionMetadataBuilder(1, 4, 16)
        builder_class = MagicMock()
        builder_class.from_config.return_value = builder
        mock_builder_classes.return_value = (builder_class,)
        executor = ModelExecutor(_FakeModel(num_layers=1), {"model_type": "cache_aware"}, max_seqs_per_batch=4)
        cache = LayerCache(
            key=torch.zeros(2, 1, 2, 64),
            value=torch.ones(2, 1, 2, 64),
            swa=torch.zeros(2, 8, 1, 64),
        )
        executor.bind_kv_caches([cache])
        batch = _make_input_batch((1,))
        metadata = SimpleNamespace(multi_block_tables=(torch.tensor([[1, 0]], dtype=torch.int32),))
        allocated = builder.allocate_persistent(batch, metadata)
        assert allocated.block_table.shape == (1, 2)  # Bound page8, not initial page4.
        builder.update_persistent(allocated, batch, metadata)
        assert allocated.context_lengths.tolist() == [1]
        assert allocated.slot_mapping.tolist() == [8]
        assert backend._kv_caches[0] is cache


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.execute routing
# ---------------------------------------------------------------------------


def _make_eager_runner(*, is_mla: bool = True, restores_aux_hidden_under_cp: bool = False) -> EagerRunner:
    backend_type = _MlaStubAttentionBackend if is_mla else StubAttentionBackend
    runner = EagerRunner(MagicMock(return_value=torch.ones(2)), backend_type(), torch.device("cpu"))
    runner.cp_size = 4
    runner.cp_rank = 2
    # A plain MagicMock() auto-vivifies any attribute lookup (including
    # ``restores_aux_hidden_under_cp``) as a truthy MagicMock; pin it
    # explicitly so ``model_restores_aux_hidden_under_cp`` reflects the
    # caller's intent instead of defaulting to True by mock
    # auto-attribute accident.
    runner.model.restores_aux_hidden_under_cp = restores_aux_hidden_under_cp
    return runner


def test_eager_runner_builds_model_visible_execution_metadata() -> None:
    runner = _make_eager_runner(is_mla=False)
    runner.cp_size = 1
    builder = Qwen3_5MoeMetadataBuilder(tp_size=1, tp_rank=0, dp_size=1, dp_rank=0)
    builder.build = MagicMock(wraps=builder.build)
    runner.bind_execution_metadata_builders((builder,))
    input_batch = _make_input_batch((2,))
    metadata = SimpleNamespace(is_prefill=True, is_chunked_prefill=False, is_mixed=False, is_spec_verify=False)

    def _execute_model(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        visible = get_execution_context(Qwen3_5MoeMetadata)
        assert visible.active_token_mask.tolist() == [1, 1]
        assert visible.active_rows == 2
        return input_ids + positions

    runner.model.side_effect = _execute_model
    output = runner.execute(input_batch.input_ids, input_batch.positions, metadata, input_batch=input_batch)
    builder.build.assert_called_once_with(input_batch, metadata)
    torch.testing.assert_close(output, input_batch.input_ids + input_batch.positions)


def test_eager_runner_prepares_sp_ep_layout_once_for_all_layers() -> None:
    runner = _make_eager_runner(is_mla=False)
    runner.cp_size = 1
    builder = SpMetadataBuilder(tp_size=2, tp_rank=1, dp_size=2, dp_rank=0)
    real_build = builder.build
    builder.build = MagicMock(wraps=real_build)
    runner.bind_execution_metadata_builders((builder,))
    input_batch = _make_input_batch((3,))
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        dp_execution_token_counts=(3, 5),
        raw_dp_execution_token_counts=(3, 5),
    )

    def execute_model(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        first = get_execution_context(SpMetadata).ep_layout
        assert first.dispatch_counts == (2, 1, 3, 2)
        return input_ids + positions

    runner.model.side_effect = execute_model
    output = runner.execute(input_batch.input_ids, input_batch.positions, metadata, input_batch=input_batch)
    builder.build.assert_called_once_with(input_batch, metadata)
    torch.testing.assert_close(output, input_batch.input_ids + input_batch.positions)


def test_eager_runner_requires_input_batch_for_execution_metadata() -> None:
    runner = _make_eager_runner(is_mla=False)
    runner.cp_size = 1
    runner.bind_execution_metadata_builders((MagicMock(),))
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
    )

    with pytest.raises(RuntimeError, match="require upstream InputBatch"):
        runner.execute(torch.zeros(2), torch.arange(2), metadata)


def test_eager_runner_preserves_qwen_pure_prefill_cp_context_contract() -> None:
    runner = _make_eager_runner(is_mla=False)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
        kv_seq_lens_host=None,
    )

    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        return_value=object(),
    ) as build_context:
        runner.execute(torch.zeros(8), torch.arange(8), metadata)

    build_context.assert_called_once_with(
        [3, 5],
        [3, 5],
        4,
        2,
        torch.device("cpu"),
    )


def test_eager_runner_reuses_cp_plan_only_for_matching_geometry() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
    )
    contexts = []

    def execute_model(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        contexts.append(get_forward_context().cp_context)
        return input_ids

    runner.model.side_effect = execute_model
    plans = [object(), object(), object()]
    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        side_effect=plans,
    ) as build_context:
        runner.execute(torch.zeros(8), torch.arange(8), metadata)
        runner.execute(torch.zeros(8), torch.arange(8), metadata)
        metadata.kv_seq_lens_host = torch.tensor([7, 9], dtype=torch.int32)
        runner.execute(torch.zeros(8), torch.arange(8), metadata)
        metadata.q_seq_lens_host = torch.tensor([4, 4], dtype=torch.int32)
        runner.execute(torch.zeros(8), torch.arange(8), metadata)

    assert build_context.call_count == 3
    assert contexts == [plans[0], plans[0], plans[1], plans[2]]


def test_eager_runner_skips_cp_for_empty_rank() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        is_dummy=True,
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
    )

    with patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context:
        runner.execute(torch.ones(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert runner.attention_backend._prepared
    runner.model.assert_called_once()


def test_eager_runner_preserves_qwen_missing_lengths_without_cp() -> None:
    runner = _make_eager_runner(is_mla=False)
    runner.cp_size = 1
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=None,
        kv_seq_lens_host=None,
    )

    with patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context:
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert runner.attention_backend._prepared
    runner.model.assert_called_once()


def test_eager_runner_builds_cp_context_for_chunked_prefill() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([11, 13], dtype=torch.int32),
    )

    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        return_value=object(),
    ) as build_context:
        runner.execute(torch.zeros(8), torch.arange(8), metadata)

    build_context.assert_called_once_with(
        [3, 5],
        [11, 13],
        4,
        2,
        torch.device("cpu"),
    )


def test_eager_runner_rejects_mixed_cp_before_collective() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=True,
        is_spec_verify=False,
    )

    with (
        patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context,
        pytest.raises(NotImplementedError, match="mixed batches"),
    ):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert not runner.attention_backend._prepared


def test_eager_runner_rejects_mla_spec_verify_cp_without_aux_restore_before_collective() -> None:
    runner = _make_eager_runner(restores_aux_hidden_under_cp=False)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=False,
        is_spec_verify=True,
    )

    with (
        patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context,
        pytest.raises(NotImplementedError, match="target-side speculative verification"),
    ):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert not runner.attention_backend._prepared


def test_eager_runner_admits_mla_spec_verify_cp_with_aux_restore_capability() -> None:
    runner = _make_eager_runner(restores_aux_hidden_under_cp=True)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=False,
        is_spec_verify=True,
        q_seq_lens_host=torch.tensor([2], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([2], dtype=torch.int32),
    )

    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        return_value=object(),
    ) as build_context:
        runner.execute(torch.zeros(2), torch.arange(2), metadata)

    build_context.assert_called_once_with([2], [2], 4, 2, torch.device("cpu"))
    assert runner.attention_backend._prepared


@pytest.mark.parametrize(
    ("is_mla", "is_chunked_prefill", "is_mixed", "is_spec_verify"),
    [
        (False, True, True, False),
        (False, True, False, True),
        (True, False, False, True),
    ],
)
def test_eager_runner_preserves_non_cp_fallback(
    is_mla: bool,
    is_chunked_prefill: bool,
    is_mixed: bool,
    is_spec_verify: bool,
) -> None:
    runner = _make_eager_runner(is_mla=is_mla)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=is_chunked_prefill,
        is_mixed=is_mixed,
        is_spec_verify=is_spec_verify,
    )

    def execute_model(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert get_forward_context().cp_context is None
        return input_ids + positions

    runner.model.side_effect = execute_model
    with patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context:
        output = runner.execute(torch.ones(1), torch.ones(1), metadata)

    build_context.assert_not_called()
    assert runner.attention_backend._prepared
    runner.model.assert_called_once()
    torch.testing.assert_close(output, torch.full((1,), 2.0))


@pytest.mark.parametrize(
    ("is_mla", "q_seq_lens_host", "kv_seq_lens_host"),
    [
        (True, None, None),
        (True, None, torch.tensor([1], dtype=torch.int32)),
        (True, torch.tensor([1], dtype=torch.int32), None),
        (False, None, torch.tensor([1], dtype=torch.int32)),
    ],
)
def test_eager_runner_rejects_missing_cp_lengths(
    monkeypatch: pytest.MonkeyPatch,
    is_mla: bool,
    q_seq_lens_host: torch.Tensor | None,
    kv_seq_lens_host: torch.Tensor | None,
) -> None:
    runner = _make_eager_runner()
    monkeypatch.setattr(type(runner.attention_backend), "is_mla", property(lambda self: is_mla))
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=q_seq_lens_host,
        kv_seq_lens_host=kv_seq_lens_host,
    )

    with pytest.raises(RuntimeError, match="requires host query and KV"):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    assert not runner.attention_backend._prepared


def _mtp_executor_with_output(output: ModelOutput) -> ModelExecutor:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor._kv_bound = True
    executor.layerwise_split_size = 1
    executor._mtp_topk_reuse = MtpTopkReuse()
    executor._execution_metadata_builders = []
    executor._is_spec_draft = False
    executor._static_kernel = None
    executor.execution_pipeline = SimpleNamespace(
        execute=MagicMock(return_value=output), entry_runner=SimpleNamespace(last_graph_rows=0)
    )
    executor._attach_persistent_aux_hidden_states = MagicMock(side_effect=lambda output, *args, **kwargs: output)
    return executor


@pytest.mark.parametrize("chunked_decode", [False, True])
def test_mtp_executor_uses_sampler_current_rows_and_retains_owned_selection(chunked_decode: bool) -> None:
    source = torch.arange(12).reshape(4, 1, 3)
    expected = source[[1, 3]].clone()
    executor = _mtp_executor_with_output(ModelOutput(hidden_states=torch.zeros(4, 4), mtp_topk_indices=source))
    metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=chunked_decode, is_mixed=False)
    batch_metadata = SimpleNamespace(is_prefilling=[False] * 4) if chunked_decode else None
    sampling = SimpleNamespace(parameters=SimpleNamespace(selected_token_idxes=torch.tensor([1, 3])))
    executor.execute(
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        metadata,
        sampling_input=sampling,
        input_batch_metadata=batch_metadata,
        mtp_draft_step=0,
    )
    owned = executor._mtp_topk_reuse.input_for_step(1, 2)
    torch.testing.assert_close(owned, expected)
    source.fill_(-1)
    torch.testing.assert_close(owned, expected)
    executor.execution_pipeline.execute.return_value = ModelOutput(
        hidden_states=torch.zeros(2, 4), mtp_topk_indices=expected
    )
    sampling = SimpleNamespace(parameters=SimpleNamespace(selected_token_idxes=torch.tensor([0, 1])))
    batch_metadata = SimpleNamespace(is_prefilling=[False] * 2) if chunked_decode else None
    executor.execute(
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        metadata,
        sampling_input=sampling,
        input_batch_metadata=batch_metadata,
        mtp_draft_step=1,
    )
    assert executor.execution_pipeline.execute.call_args.args[0].mtp_topk_indices is owned


def test_mtp_executor_prefill_clears_old_state_without_gathering_an_unused_selection() -> None:
    executor = _mtp_executor_with_output(ModelOutput(hidden_states=torch.zeros(4, 4)))
    executor._mtp_topk_reuse.finish_step(0, torch.ones(2, 1, 3), None)
    executor._mtp_topk_reuse.finish_step = MagicMock(wraps=executor._mtp_topk_reuse.finish_step)
    metadata = SimpleNamespace(is_prefill=True, is_chunked_prefill=False, is_mixed=False)
    executor.execute(torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long), metadata)
    executor.execution_pipeline.execute.assert_called_once()
    executor._mtp_topk_reuse.finish_step.assert_not_called()
    with pytest.raises(ValueError, match="consecutive draft steps"):
        executor._mtp_topk_reuse.input_for_step(1, 2)


def test_mtp_reuse_requires_sampling_before_state_reset_or_forward() -> None:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor._kv_bound = True
    executor.layerwise_split_size = 1
    executor._mtp_topk_reuse = MagicMock()
    executor.execution_pipeline = MagicMock()
    metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False, is_mixed=False)

    with pytest.raises(RuntimeError, match="MTP top-k reuse requires upstream sampling row selection"):
        executor.execute(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), metadata)

    executor._mtp_topk_reuse.input_for_step.assert_not_called()
    executor.execution_pipeline.execute.assert_not_called()


class TestExecuteRouting:
    @pytest.mark.parametrize("prefill,chunked", [(True, False), (False, True)])
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_multimodal_inputs_reach_eager_runner(self, mock_create: MagicMock, prefill: bool, chunked: bool) -> None:
        class Body(nn.Module):
            def forward(
                self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                *,
                inputs_embeds: torch.Tensor,
                deepstack_input_embeds: tuple[torch.Tensor, ...],
            ) -> torch.Tensor:
                assert get_forward_context().cp_context is None
                return inputs_embeds + deepstack_input_embeds[0]

        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        model.model = Body()
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)
        executor.bind_kv_caches([(torch.zeros(2, 1, 2, 64), torch.ones(2, 1, 2, 64))])
        executor.decode_graph_runner = MagicMock()
        executor.decode_graph_runner.can_execute.return_value = False
        prepared = MultimodalInputs(torch.ones(1, 2), (torch.full((1, 2), 2.0),))
        result = executor.execute(
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            SimpleNamespace(is_prefill=prefill, is_chunked_prefill=chunked),
            multimodal_inputs=prepared,
        )
        torch.testing.assert_close(result, torch.full((1, 2), 3.0))
        executor.decode_graph_runner.can_execute.assert_called_once()
        assert executor.decode_graph_runner.can_execute.call_args.args[0].multimodal_inputs is prepared
        executor.decode_graph_runner.execute.assert_not_called()

    @pytest.mark.parametrize(
        ("draft_event_index", "expected_calls"),
        [(2, [2]), (None, [0, 1, 2])],
    )
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_spec_draft_waits_for_dedicated_load_event(
        self,
        mock_create: MagicMock,
        draft_event_index: int | None,
        expected_calls: list[int],
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        executor = ModelExecutor(
            _FakeModel(num_layers=1),
            {"is_draft_engine": True},
            max_seqs_per_batch=4,
        )
        executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
        executor.eager_runner = MagicMock()
        executor.eager_runner.execute.return_value = torch.ones(1)
        synchronizer = _HostLoadSynchronizer(3)

        executor.execute(
            torch.ones(1),
            torch.zeros(1),
            SimpleNamespace(is_prefill=True, is_chunked_prefill=False),
            layer_load_synchronizer=synchronizer,
            draft_load_event_index=draft_event_index,
        )

        assert synchronizer.calls == expected_calls

    @pytest.mark.parametrize("prefill,chunked", [(True, False), (False, True), (False, False)])
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_eager_waits_at_each_attention(self, mock_create: MagicMock, prefill: bool, chunked: bool) -> None:
        order: list[tuple[str, int]] = []

        class Loads(_HostLoadSynchronizer):
            def synchronize_layer(self, event_index: int) -> bool:
                order.append(("load", event_index))
                return super().synchronize_layer(event_index)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([_make_attention_layer(layer_id=i) for i in range(2)])

            def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
                q = torch.ones(input_ids.numel(), 8, 64)
                k = torch.zeros(input_ids.numel(), 2, 64)
                v = torch.ones_like(k)
                for i, layer in enumerate(self.layers):
                    q = layer(q, k, v)
                    order.append(("attention", i))
                return q

        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=0)
        model.model = Model()
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)
        executor.bind_kv_caches([(torch.zeros(2, 1, 2, 64), torch.ones(2, 1, 2, 64)) for _ in range(2)])
        metadata = SimpleNamespace(is_prefill=prefill, is_chunked_prefill=chunked, is_dummy=False)
        synchronizer = Loads(2)
        for _ in range(2):
            executor.execute(torch.ones(1), torch.zeros(1), metadata, layer_load_synchronizer=synchronizer)
        assert order == [("load", 0), ("attention", 0), ("load", 1), ("attention", 1)] * 2

    @pytest.mark.parametrize("failed_event", [None, 1])
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_graph_waits_before_warmup_on_every_request(self, mock_create: MagicMock, failed_event: int | None) -> None:
        mock_create.return_value = StubAttentionBackend()
        executor = ModelExecutor(_FakeModel(num_layers=1), {}, max_seqs_per_batch=4)
        executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
        metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False)
        synchronizer = _HostLoadSynchronizer(2, failed_event)
        graph = MagicMock()
        graph.can_execute.return_value = True
        executor.decode_graph_runner = graph
        for request_index in range(2):

            def warmup(*args: object, request_count: int = request_index + 1) -> None:
                assert synchronizer.calls == [0, 1] * request_count

            graph.warmup.side_effect = warmup
            expectation = (
                pytest.raises(RuntimeError, match="Host cache load event 1")
                if failed_event is not None
                else nullcontext()
            )
            with expectation:
                executor.execute(torch.ones(1), torch.zeros(1), metadata, layer_load_synchronizer=synchronizer)
        assert graph.warmup.call_count == (2 if failed_event is None else 0)
        assert graph.execute.call_count == (2 if failed_event is None else 0)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_execute_without_bind_raises(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        metadata = MagicMock(spec=AttentionMetadata)
        with pytest.raises(RuntimeError, match="KV caches are not bound"):
            executor.execute(torch.zeros(1), torch.zeros(1), metadata)

    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_execute_routes_to_eager_runner(self, mock_create: MagicMock) -> None:
        class Body(nn.Module):
            def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
                assert not torch.is_grad_enabled()
                return input_ids + positions

        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        model.model = Body()
        executor = ModelExecutor(model, {"enable_graph": False, "dp_size": 2}, max_seqs_per_batch=4)
        assert executor.decode_graph_runner is None
        executor.bind_kv_caches([(torch.zeros(2, 1, 2, 64), torch.ones(2, 1, 2, 64))])
        metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False)
        batch_metadata = SimpleNamespace(
            num_reqs=1,
            num_tokens=1,
            num_scheduled_tokens=[1],
            num_computed_tokens=[0],
            query_start_loc=[0, 1],
            is_prefilling=[False],
        )
        with patch.object(InputBatch, "from_runtime") as from_runtime:
            result = executor.execute(
                torch.ones(1, dtype=torch.int32),
                torch.full((1,), 2, dtype=torch.int32),
                metadata,
                input_batch_metadata=batch_metadata,
            )
        from_runtime.assert_not_called()
        torch.testing.assert_close(result, torch.full((1,), 3, dtype=torch.int32))
