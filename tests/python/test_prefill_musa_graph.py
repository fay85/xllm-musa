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

"""CPU tests for C++-aligned Python prefill MusaGraph gating."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import torch

import xllm.python as python_pkg

_mock_kernels = MagicMock()
python_pkg.kernels = _mock_kernels
python_pkg._runtime_kernels = _mock_kernels
sys.modules["xllm.python.kernels"] = _mock_kernels
sys.modules.setdefault("xllm.python.layers.gated_delta_net", MagicMock())

from xllm.python.attention.musa_paged_attention import (  # noqa: E402
    MusaPagedAttentionBackend,
)
from xllm.python.models.qwen3_5 import (  # noqa: E402
    PartialRotaryEmbedding,
    Qwen3_5Attention,
    Qwen3_5Config,
)
from xllm.python.model_executor.runners.decode_musa_graph import (  # noqa: E402
    _ensure_static_input_embedding,
    _input_embedding_matches_tokens,
    _is_supported_prefill_conv_cache,
    _live_input_embedding,
)
from xllm.python.model_executor.runners.prefill_musa_graph import (  # noqa: E402
    _align_qwen35_bucket,
    _exact_prefill_tokens,
    _gdn_graph_tokens,
    _gdn_kkt_tokens,
    _generate_prefill_graph_tokens,
    _host_prefill_has_gdn_prefix,
    _prefill_bucket,
)


def test_prefill_bucket_matches_cpp_c1_380() -> None:
    assert _prefill_bucket(380, 8192, align_qwen35=True) == 384
    assert _prefill_bucket(380, 256, align_qwen35=True) is None


def test_prefill_bucket_rejects_large_c1_padding() -> None:
    # 16 maps to 32 on the ladder, then 64-align → padding 48 > 32.
    assert _prefill_bucket(16, 8192, align_qwen35=True) is None
    assert _prefill_bucket(16, 8192, align_qwen35=False) == 16


def test_generate_prefill_graph_tokens_includes_cpp_shoulders() -> None:
    buckets = _generate_prefill_graph_tokens(8192)
    assert 384 in buckets
    assert 2080 in buckets
    assert 2112 in buckets
    assert buckets[-1] == 8192


def test_align_qwen35_bucket_keeps_2080() -> None:
    assert _align_qwen35_bucket(2080, 8192) == 2080
    assert _align_qwen35_bucket(32, 8192) == 64


def test_gdn_kkt_tokens_aligns_to_64() -> None:
    assert _gdn_kkt_tokens(27) == 27
    assert _gdn_kkt_tokens(63) == 63
    assert _gdn_kkt_tokens(64) == 64
    assert _gdn_kkt_tokens(380) == 384
    assert _gdn_kkt_tokens(382) == 384
    assert _gdn_kkt_tokens(384) == 384


def test_exact_prefill_tokens_rejects_padding() -> None:
    assert _exact_prefill_tokens(382, 512) == 382
    assert _exact_prefill_tokens(0, 512) is None
    assert _exact_prefill_tokens(513, 512) is None


def test_input_embedding_matches_tokens() -> None:
    class _Meta:
        input_embedding = None

    assert _input_embedding_matches_tokens(_Meta(), 382) is True
    assert _live_input_embedding(_Meta()) is None
    _Meta.input_embedding = torch.zeros(0, 8)
    assert _input_embedding_matches_tokens(_Meta(), 382) is True
    _Meta.input_embedding = torch.ones(382, 8)
    assert _input_embedding_matches_tokens(_Meta(), 382) is True
    assert _input_embedding_matches_tokens(_Meta(), 384) is False


def test_static_input_embedding_reuses_storage() -> None:
    live = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    buf = _ensure_static_input_embedding(None, live, 2, torch.device("cpu"))
    assert buf is not None
    assert torch.equal(buf, live)
    live2 = live + 1
    again = _ensure_static_input_embedding(buf, live2, 2, torch.device("cpu"))
    assert again is buf
    assert torch.equal(buf, live2)
    assert _ensure_static_input_embedding(buf, None, 2, torch.device("cpu")) is buf


def test_prefill_conv_accepts_mtp_history() -> None:
    conv_dim = 8
    kernel_size = 4
    fused = torch.zeros((3, conv_dim, kernel_size - 1))
    mtp = torch.zeros((3, conv_dim, kernel_size - 1 + 1))
    assert _is_supported_prefill_conv_cache(fused, conv_dim, kernel_size)
    assert _is_supported_prefill_conv_cache(mtp, conv_dim, kernel_size)


def test_gdn_graph_uses_exact_live_length() -> None:
    assert _gdn_graph_tokens(27, 512) is None
    assert _gdn_graph_tokens(63, 512) is None
    assert _gdn_graph_tokens(64, 512) == 64
    assert _gdn_graph_tokens(380, 512) == 380
    assert _gdn_graph_tokens(382, 512) == 382
    assert _gdn_graph_tokens(384, 512) == 384
    assert _gdn_graph_tokens(513, 512) is None


def test_host_prefill_prefix_uses_cpu_kv_lengths() -> None:
    class _Meta:
        kv_seq_lens_host = torch.tensor([382], dtype=torch.int32)

    has_state = torch.tensor([False])
    assert _host_prefill_has_gdn_prefix(_Meta(), 382, has_state) is False
    _Meta.kv_seq_lens_host = torch.tensor([0, 400], dtype=torch.int32)
    assert _host_prefill_has_gdn_prefix(_Meta(), 382, has_state) is True


def _qwen35_attention_config(**overrides: object) -> Qwen3_5Config:
    config: dict[str, object] = {
        "hidden_size": 8,
        "n_layers": 1,
        "n_heads": 1,
        "n_kv_heads": 1,
        "head_dim": 8,
        "intermediate_size": 16,
        "vocab_size": 32,
        "layer_types": ["full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
    }
    config.update(overrides)
    parsed = Qwen3_5Config.from_dict(config)
    parsed.validate()
    return parsed


def _qwen35_attention(cfg: Qwen3_5Config) -> Qwen3_5Attention:
    device = torch.device("cpu")
    rotary = PartialRotaryEmbedding(
        cfg.head_dim,
        2,
        16,
        cfg.rope_theta,
        torch.float32,
        device,
    )
    return Qwen3_5Attention(cfg, 0, torch.float32, device, rotary)


def test_qwen35_full_attention_keeps_unbounded_left_window() -> None:
    attention = _qwen35_attention(_qwen35_attention_config())
    assert attention.attn.sliding_window == -1


def test_qwen35_sliding_attention_keeps_configured_window() -> None:
    attention = _qwen35_attention(
        _qwen35_attention_config(use_sliding_window=True, sliding_window=128)
    )
    assert attention.attn.sliding_window == 128


def test_musa_fa3_keeps_unbounded_left_window_sentinel() -> None:
    backend = MusaPagedAttentionBackend(
        num_heads=16,
        num_kv_heads=2,
        head_dim=256,
        scale=256**-0.5,
        sliding_window=-1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert backend._window_left == -1
