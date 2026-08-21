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

"""Opt-in Python prefill GPU attribution. Enable with XLLM_PYTHON_PREFILL_BREAKDOWN=1.

Bucket names match C++ PrefillBreakdown so C=1/382 logs can be compared.
Events are skipped while a MUSA graph is being captured.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Iterator

import torch

from scripts.logger import logger

_NESTED = frozenset(
    {
        "mate",
        "gdn_proj",
        "gdn_conv",
        "gdn_gate",
        "gdn_o_proj",
        "gdn_state_prep",
        "gdn_state_write",
        "gdn_norm",
        "full_qkv",
        "full_prep",
        "full_fa",
        "full_o_proj",
        "lm_head",
        "mlp_gate_up_quant",
        "mlp_gate_up_gemm",
    }
)

_records: list[tuple[str, object, object]] = []


def enabled() -> bool:
    return os.environ.get("XLLM_PYTHON_PREFILL_BREAKDOWN", "") == "1"


def max_samples() -> int:
    raw = os.environ.get("XLLM_PYTHON_PREFILL_BREAKDOWN_SAMPLES", "2")
    return max(int(raw), 0)


def _stream_capturing() -> bool:
    check = getattr(torch.musa, "is_current_stream_capturing", None)
    if check is None:
        return False
    return bool(check())


def begin() -> None:
    global _records
    if not enabled():
        return
    _records = []


def end_and_log(num_tokens: int, wall_ms: float, *, mode: str) -> None:
    global _records
    if not enabled():
        return
    if not _records:
        logger.info(
            f"[PREFILL_BREAKDOWN] mode={mode} n_tokens={num_tokens} "
            f"wall_ms={wall_ms:.2f} (no samples)"
        )
        return
    torch.musa.current_stream().synchronize()
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, start, end in _records:
        elapsed = float(start.elapsed_time(end))
        sums[name] = sums.get(name, 0.0) + elapsed
        counts[name] = counts.get(name, 0) + 1
    accounted = sum(
        ms for name, ms in sums.items() if name not in _NESTED
    )
    other = wall_ms - accounted
    mate_ms = sums.get("mate", 0.0)
    gdn_ms = sums.get("gdn_attn", 0.0)
    mlp_ms = (
        sums.get("mlp_gate_up", 0.0)
        + sums.get("mlp_act", 0.0)
        + sums.get("mlp_down", 0.0)
    )
    logger.info(
        f"[PREFILL_BREAKDOWN] mode={mode} n_tokens={num_tokens} "
        f"wall_ms={wall_ms:.2f} "
        f"accounted_ms={accounted:.2f} other_ms={other:.2f} "
        f"embed_ms={sums.get('embed', 0.0):.2f} "
        f"full_attn_ms={sums.get('full_attn', 0.0):.2f} "
        f"gdn_attn_ms={gdn_ms:.2f} mate_ms={mate_ms:.2f} "
        f"gdn_non_mate_ms={gdn_ms - mate_ms:.2f} "
        f"gdn_proj_ms={sums.get('gdn_proj', 0.0):.2f} "
        f"gdn_conv_ms={sums.get('gdn_conv', 0.0):.2f} "
        f"gdn_gate_ms={sums.get('gdn_gate', 0.0):.2f} "
        f"gdn_state_prep_ms={sums.get('gdn_state_prep', 0.0):.2f} "
        f"gdn_state_write_ms={sums.get('gdn_state_write', 0.0):.2f} "
        f"gdn_norm_ms={sums.get('gdn_norm', 0.0):.2f} "
        f"gdn_o_proj_ms={sums.get('gdn_o_proj', 0.0):.2f} "
        f"full_qkv_ms={sums.get('full_qkv', 0.0):.2f} "
        f"full_prep_ms={sums.get('full_prep', 0.0):.2f} "
        f"full_fa_ms={sums.get('full_fa', 0.0):.2f} "
        f"full_o_proj_ms={sums.get('full_o_proj', 0.0):.2f} "
        f"mlp_ms={mlp_ms:.2f} "
        f"mlp_gate_up_ms={sums.get('mlp_gate_up', 0.0):.2f} "
        f"mlp_gate_up_quant_ms={sums.get('mlp_gate_up_quant', 0.0):.2f} "
        f"mlp_gate_up_gemm_ms={sums.get('mlp_gate_up_gemm', 0.0):.2f} "
        f"mlp_act_ms={sums.get('mlp_act', 0.0):.2f} "
        f"mlp_down_ms={sums.get('mlp_down', 0.0):.2f} "
        f"norm_ms={sums.get('norm', 0.0):.2f} "
        f"lm_head_ms={sums.get('lm_head', 0.0):.2f} "
        f"n_full={counts.get('full_attn', 0)} "
        f"n_gdn={counts.get('gdn_attn', 0)} "
        f"n_mate={counts.get('mate', 0)}"
    )
    for name in sorted(sums):
        logger.info(
            f"[PREFILL_BREAKDOWN_BUCKET] name={name} "
            f"ms={sums[name]:.2f} calls={counts[name]} "
            f"pct={100.0 * sums[name] / wall_ms:.1f}"
        )
    _records = []


@contextmanager
def scope(name: str) -> Iterator[None]:
    if not enabled() or _stream_capturing():
        yield
        return
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        _records.append((name, start, end))
