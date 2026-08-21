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

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    LayerCacheInput,
    normalize_layer_caches,
)
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.platform import current_platform
from scripts.logger import logger


def _is_npu_device(device: torch.device) -> bool:
    return device.type in ("npu", "privateuseone") and not current_platform.is_musa()


def _is_musa_device(device: torch.device) -> bool:
    return current_platform.is_musa() or device.type == "musa"


def _use_decode_graph(metadata: AttentionMetadata) -> bool:
    """True for decode and MTP target verify.

    Qwen3.5 verify is marked ``CHUNKED_PREFILL`` in C++ so FlashInfer
    sees causal token order. That flag must not send verify to the
    prefill graph or skip MusaGraph capture.
    """
    if bool(getattr(metadata, "is_spec_verify", False)):
        return True
    return not bool(metadata.is_prefill or metadata.is_chunked_prefill)


def _decode_graph_max_tokens(max_seqs_per_batch: int, config: dict) -> int:
    """Token width for decode/verify graphs.

    ``max_seqs_per_batch`` is sequence concurrency. MTP verify forwards
    ``num_speculative_tokens + 1`` tokens for every live sequence, so graph
    capacity must cover their product rather than either dimension alone.
    """
    speculative_tokens = int(config.get("num_speculative_tokens") or 0)
    verify_width = int(max_seqs_per_batch) * max(speculative_tokens + 1, 1)
    return max(int(max_seqs_per_batch), verify_width, 8)


def _resolve_graph_backend(config: dict) -> str:
    graph_backend = str(config.get("python_graph_backend", "off")).lower()
    graph_disabled = graph_backend in ("", "off", "none", "0")
    if graph_disabled and config.get("enable_graph", False):
        if current_platform.is_musa():
            return "musagraph"
        if current_platform.is_npu():
            return "aclgraph"
    return graph_backend


def _create_attention_backend(
    first_attention: Attention,
    device: torch.device,
    dtype: torch.dtype,
) -> AttentionBackend:
    if _is_npu_device(device):
        from xllm.python.attention.npu_paged_attention import (
            NpuPagedAttentionBackend,
        )

        return NpuPagedAttentionBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            device=device,
            dtype=dtype,
        )
    if _is_musa_device(device):
        from xllm.python.attention.musa_paged_attention import (
            MusaPagedAttentionBackend,
        )

        return MusaPagedAttentionBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            device=device,
            dtype=dtype,
        )
    if device.type == "cuda":
        from xllm.python.attention.flashinfer import FlashInferBackend

        return FlashInferBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            device=device,
            dtype=dtype,
        )
    raise NotImplementedError(
        f"No attention backend available for device type '{device.type}'"
    )


class ModelExecutor:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        max_seqs_per_batch: int,
        num_decoding_tokens: int = 1,
        acl_graph_decode_batch_size_limit: int | None = None,
    ) -> None:
        self.model = model
        self._kv_bound = False

        attention_layers = [
            module for module in model.modules() if isinstance(module, Attention)
        ]
        if not attention_layers:
            raise ValueError("Python model does not contain an Attention layer")

        first_attention = attention_layers[0]
        expected_config = self._attention_config(first_attention)
        for layer in attention_layers[1:]:
            if self._attention_config(layer) != expected_config:
                raise ValueError(
                    "Attention backend requires identical attention configuration "
                    "across all layers"
                )

        first_parameter = next(model.parameters())
        device = first_parameter.device
        self._num_attention_layers = len(attention_layers)
        self.attention_backend = _create_attention_backend(
            first_attention, device, first_parameter.dtype
        )

        execution_model = model.model
        self.eager_runner = EagerRunner(execution_model, self.attention_backend, device)
        self.eager_runner.cp_size = int(config.get("cp_size", 1))
        self.eager_runner.cp_rank = int(config.get("cp_rank", 0))
        self.decode_graph_runner = None
        self.prefill_graph_runner = None
        self.inductor_runner = None

        graph_backend = _resolve_graph_backend(config)
        logger.info(f"Python decode graph backend: {graph_backend}")
        dp_size = int(config.get("dp_size", 1))
        dp_rank = int(config.get("dp_rank", 0))
        if graph_backend in ("", "off", "none", "0"):
            pass
        elif graph_backend == "cudagraphs":
            from xllm.python.model_executor.runners.decode_cuda_graph import (
                DecodeCudaGraphRunner,
            )

            self.decode_graph_runner = DecodeCudaGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_seqs_per_batch,
                int(config.get("max_position_embeddings") or 40960),
            )
        elif graph_backend == "musagraph":
            from xllm.python.model_executor.runners.decode_musa_graph import (
                DecodeMusaGraphRunner,
            )
            from xllm.python.model_executor.runners.prefill_musa_graph import (
                PrefillMusaGraphRunner,
            )

            max_decode_tokens = _decode_graph_max_tokens(
                max_seqs_per_batch, config
            )
            logger.info(
                f"Python MUSA decode graph max tokens: {max_decode_tokens}"
            )
            self.decode_graph_runner = DecodeMusaGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_decode_tokens,
                int(config.get("max_position_embeddings") or 40960),
                lm_head=getattr(model, "lm_head", None),
            )
            # C++ can capture an 8k piecewise ladder. Python reserves a
            # per-layer GEMM workspace, so 27B OOM at that cap. Honor an
            # explicit Python override; otherwise keep the C++ flag but cap
            # it at a C=1-safe size that still covers the 384-token bucket.
            if "max_tokens_for_python_prefill_graph" in config:
                prefill_max_tokens = int(
                    config["max_tokens_for_python_prefill_graph"]
                )
            else:
                prefill_max_tokens = min(
                    int(config.get("max_tokens_for_graph_mode") or 2048),
                    512,
                )
            logger.info(
                f"Python MUSA prefill graph max tokens: {prefill_max_tokens}"
            )
            self.prefill_graph_runner = PrefillMusaGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                prefill_max_tokens,
                lm_head=getattr(model, "lm_head", None),
            )
            # Prefill GEMM/activations use GraphActivationPool. Reserve only
            # decode-width persistent buffers so 27B does not keep a 512-token
            # per-layer workspace that evicts weights.
            workspace_tokens = max_decode_tokens
            reserve_model = getattr(model, "reserve_graph_workspaces", None)
            if reserve_model is None:
                reserve_model = getattr(
                    execution_model, "reserve_graph_workspaces", None
                )
            if reserve_model is not None:
                reserve_model(workspace_tokens)
            reserve_prefill = getattr(
                self.attention_backend, "reserve_prefill_buffers", None
            )
            if reserve_prefill is not None:
                reserve_prefill(self.prefill_graph_runner.max_tokens)
        elif graph_backend == "aclgraph":
            from xllm.python.model_executor.runners.decode_acl_graph import (
                DecodeAclGraphRunner,
            )

            decode_tokens = max(1, int(num_decoding_tokens))
            decode_batch_size_limit = (
                None
                if acl_graph_decode_batch_size_limit is None
                else max(1, int(acl_graph_decode_batch_size_limit))
            )
            graph_sequence_capacity = int(max_seqs_per_batch)
            if decode_batch_size_limit is not None:
                graph_sequence_capacity = min(
                    graph_sequence_capacity,
                    decode_batch_size_limit,
                )
            max_graph_tokens = graph_sequence_capacity * decode_tokens
            logger.info(
                f"Python ACL decode graph max tokens: {max_graph_tokens}"
            )
            self.decode_graph_runner = DecodeAclGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_graph_tokens,
                int(config.get("max_position_embeddings") or 40960),
                dp_size,
                dp_rank,
                decode_batch_size_limit,
                decode_tokens,
            )
        else:
            from xllm.python.model_executor.runners.inductor import InductorRunner

            self.inductor_runner = InductorRunner(
                execution_model, self.attention_backend, device, graph_backend
            )

    @staticmethod
    def _attention_config(layer: Attention) -> tuple[int, int, int, float, int]:
        return (
            layer.num_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.scale,
            layer.sliding_window,
        )

    def bind_kv_caches(self, kv_caches: list[LayerCacheInput]) -> None:
        layer_caches = normalize_layer_caches(kv_caches)
        attention_ids = [
            module.layer_id
            for module in self.model.modules()
            if isinstance(module, Attention)
        ]
        required_layers = max(attention_ids) + 1
        if len(layer_caches) < required_layers:
            raise ValueError(
                "cache layer count does not match the model layer layout"
            )
        if self._kv_bound:
            return
        self.attention_backend.bind_kv_caches(layer_caches)
        self.eager_runner.bind_layer_caches(layer_caches)
        if self.decode_graph_runner is not None:
            self.decode_graph_runner.bind_layer_caches(layer_caches)
        if self.prefill_graph_runner is not None:
            self.prefill_graph_runner.bind_layer_caches(layer_caches)
        self._refresh_gdn_ssm_workspaces(layer_caches)
        self._kv_bound = True

    def _refresh_gdn_ssm_workspaces(
        self, layer_caches: list[LayerCache]
    ) -> None:
        for module in self.model.modules():
            refresh = getattr(module, "refresh_graph_ssm_workspace", None)
            if refresh is None:
                continue
            layer_id = getattr(module, "layer_id", None)
            if layer_id is None or int(layer_id) >= len(layer_caches):
                continue
            ssm = layer_caches[int(layer_id)].ssm
            if ssm is not None:
                refresh(ssm)

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")

        profile = os.environ.get("XLLM_PYTHON_PREFILL_PROFILE", "") == "1"
        decode_start = time.perf_counter() if profile else 0.0
        graph_runner = self.decode_graph_runner
        if graph_runner is not None and _use_decode_graph(metadata):
            graph_runner.warmup(input_ids.device, input_ids.dtype)
            if graph_runner.can_execute(input_ids, metadata):
                hidden = graph_runner.execute(input_ids, positions, metadata)
                last_logits = getattr(graph_runner, "last_logits", None)
                if last_logits is not None:
                    return hidden, last_logits
                return hidden
        decode_ms = (time.perf_counter() - decode_start) * 1000.0 if profile else 0.0
        prefill_can_start = time.perf_counter() if profile else 0.0
        prefill_runner = self.prefill_graph_runner
        if prefill_runner is not None and prefill_runner.can_execute(
            input_ids, metadata
        ):
            can_ms = (
                (time.perf_counter() - prefill_can_start) * 1000.0 if profile else 0.0
            )
            hidden = prefill_runner.execute(input_ids, positions, metadata)
            if profile and metadata.is_prefill:
                logger.info(
                    f"python execute dispatch decode_ms={decode_ms:.2f} "
                    f"prefill_can_ms={can_ms:.2f} "
                    f"tokens={int(input_ids.shape[0])}"
                )
            last_logits = getattr(prefill_runner, "last_logits", None)
            if last_logits is not None:
                return hidden, last_logits
            return hidden
        if self.inductor_runner is not None:
            return self.inductor_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                layer_synchronizer,
            )
        return self.eager_runner.execute(
            input_ids,
            positions,
            metadata,
            input_embedding,
            layer_synchronizer,
        )
