# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parallel-mode tests for the DeepSeek-V4.1 vision encoder.

Encoder DP splits media items across the language-model TP group and
AllGathers the tower rows; encoder TP shards the tower's attention heads and
intermediate dim across that same group. Both modes must reproduce the
replicated tower's image embeddings, so the tests run one rank per thread and
emulate the TP collectives in lockstep.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from xllm.python import kernels
from xllm.python.models import deepseek_v41 as _v41_module
from xllm.python.models.deepseek_v41 import (
    DeepseekV41Aligner,
    DeepseekV41ForCausalLM,
    DeepseekV41VisionConfig,
    DeepseekV41VisionTower,
    _shard_vision_checkpoint_tensor,
)
from xllm.python.models.weight_utils import WeightLoader


def _vision_rotary(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    left, right = value.chunk(2, -1)
    rotated = torch.cat((-right, left), dim=-1)
    return (value.unsqueeze(0) * cos + rotated.unsqueeze(0) * sin).squeeze(0)


def _vision_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    actual_seq_qlen: list[int],
    actual_seq_kvlen: list[int],
    num_heads: int,
    scale: float,
    input_layout: str,
) -> torch.Tensor:
    assert input_layout == "TND"
    assert actual_seq_qlen == actual_seq_kvlen
    assert q.size(1) == num_heads
    outputs = []
    start = 0
    for end in actual_seq_qlen:
        scores = torch.einsum("thd,shd->hts", q[start:end].float(), k[start:end].float()) * scale
        outputs.append(torch.einsum("hts,shd->thd", scores.softmax(-1), v[start:end].float()).to(q.dtype))
        start = end
    return torch.cat(outputs, dim=0)


def _vision_silu_and_mul(value: torch.Tensor) -> torch.Tensor:
    gate, up = value.float().chunk(2, -1)
    return (F.silu(gate) * up).to(value.dtype)


@pytest.fixture(autouse=True)
def _cpu_vision_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "vision_rotary_mul", _vision_rotary, raising=False)
    monkeypatch.setattr(kernels, "vision_fusion_attention", _vision_attention, raising=False)
    monkeypatch.setattr(kernels, "silu_and_mul", _vision_silu_and_mul, raising=False)


def _config(**overrides: object) -> DeepseekV41VisionConfig:
    values: dict[str, object] = dict(
        n_layers=1,
        hidden_size=8,
        num_heads=2,
        intermediate_size=8,
        patch_size=2,
        num_channels=3,
        out_hidden_size=8,
        rope_theta=10000.0,
        downsample_ratio=2,
        head_dim=4,
    )
    values.update(overrides)
    return DeepseekV41VisionConfig(**values)


def _make_holder(cfg: DeepseekV41VisionConfig) -> torch.nn.Module:
    """Duck-typed stand-in for ``DeepseekV41ForCausalLM`` (unbound ``encode``)."""
    holder = torch.nn.Module()
    holder.vision_cfg = cfg
    holder.vision_model = DeepseekV41VisionTower(cfg, torch.float32, torch.device("cpu"))
    holder.aligner = DeepseekV41Aligner(cfg, torch.float32, torch.device("cpu"))
    for name in ("image_start", "image_end", "image_newline"):
        holder.register_parameter(
            name,
            torch.nn.Parameter(torch.zeros(cfg.out_hidden_size, dtype=torch.float32), requires_grad=False),
        )
    holder.requires_grad_(False)
    return holder


def _copy_state(source: torch.nn.Module, target: torch.nn.Module) -> None:
    state = {name: parameter.detach().clone() for name, parameter in source.named_parameters()}
    for name, parameter in target.named_parameters():
        parameter.data.copy_(state[name])


def _copy_sharded_state(source: torch.nn.Module, target: torch.nn.Module) -> None:
    state = {name: parameter.detach().clone() for name, parameter in source.named_parameters()}
    cfg = target.vision_cfg
    for name, parameter in target.named_parameters():
        if cfg.encoder_parallel == "tp" and name.endswith(("attn.wo.weight", "mlp.w2.weight")):
            local = WeightLoader(target, [], cfg.tp_size, cfg.tp_rank).shard(state[name], dim=1)
        else:
            local = _shard_vision_checkpoint_tensor(name, state[name], cfg)
        assert local.shape == parameter.shape, name
        parameter.data.copy_(local)


class _ThreadGroup:
    """Emulate one TP collective stream across rank threads in lockstep.

    Every rank executes the same collective sequence, so a per-thread call
    counter identifies matching calls. The barrier keeps ranks aligned; a
    timeout turns a mismatched sequence into a test failure instead of a hang.
    """

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size, timeout=60)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.pending: dict[tuple[int, int], torch.Tensor] = {}

    def bind(self, rank: int) -> None:
        self.local.rank = rank
        self.local.calls = 0

    def _slot(self) -> int:
        slot = self.local.calls
        self.local.calls += 1
        return slot

    def all_reduce(self, tensor: torch.Tensor) -> None:
        slot = self._slot()
        with self.lock:
            self.pending[(slot, self.local.rank)] = tensor.detach().clone()
        self.barrier.wait()
        with self.lock:
            total = self.pending[(slot, 0)].clone()
            for rank in range(1, self.world_size):
                total += self.pending[(slot, rank)]
        tensor.copy_(total)
        self.barrier.wait()

    def all_gather(self, tensor: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
        assert world_size == self.world_size and dim == 0
        slot = self._slot()
        with self.lock:
            self.pending[(slot, self.local.rank)] = tensor.detach().clone()
        self.barrier.wait()
        with self.lock:
            gathered = torch.cat(
                [self.pending[(slot, rank)] for rank in range(self.world_size)],
                dim=0,
            )
        self.barrier.wait()
        return gathered


def _run_ranks(holders: list[torch.nn.Module], group: _ThreadGroup, encode) -> list[torch.Tensor]:
    executor = ThreadPoolExecutor(max_workers=len(holders))
    futures = []

    def task(rank: int, holder: torch.nn.Module) -> torch.Tensor:
        group.bind(rank)
        return encode(holder)

    try:
        for rank, holder in enumerate(holders):
            futures.append(executor.submit(task, rank, holder))
        return [future.result(timeout=120) for future in futures]
    finally:
        executor.shutdown(wait=True)


def test_vision_config_selects_encoder_parallel_mode() -> None:
    base = {
        "mm_hidden_size": 8,
        "mm_num_attention_heads": 2,
        "mm_head_dim": 4,
        "mm_intermediate_size": 8,
        "mm_num_hidden_layers": 1,
        "mm_patch_size": 2,
        "mm_num_channels": 3,
        "mm_projection_dim": 8,
        "mm_downsample_ratio": 2,
    }
    default = DeepseekV41VisionConfig.from_dict(dict(base))
    assert (default.encoder_parallel, default.tp_size, default.tp_rank) == ("tp", 1, 0)

    dp = DeepseekV41VisionConfig.from_dict({**base, "enable_mm_encoder_dp": True, "tp_size": 4, "tp_rank": 2})
    assert (dp.encoder_parallel, dp.tp_size, dp.tp_rank) == ("dp", 4, 2)

    tp = DeepseekV41VisionConfig.from_dict({**base, "enable_mm_encoder_dp": False, "tp_size": 4, "tp_rank": 1})
    assert (tp.encoder_parallel, tp.tp_size, tp.tp_rank) == ("tp", 4, 1)

    implicit_tp = DeepseekV41VisionConfig.from_dict({**base, "tp_size": 4, "tp_rank": 1})
    assert (implicit_tp.encoder_parallel, implicit_tp.tp_size, implicit_tp.tp_rank) == ("tp", 4, 1)


def test_encoder_tp_validation_rejects_indivisible_layouts() -> None:
    _config(encoder_parallel="tp", tp_size=2, tp_rank=0).validate()
    # hidden_size must be num_heads * head_dim.
    with pytest.raises(ValueError, match="num_heads \\* head_dim"):
        replace(_config(encoder_parallel="tp", tp_size=2), num_heads=4, head_dim=1).validate()
    # Heads must divide evenly across ranks.
    with pytest.raises(ValueError, match="heads"):
        replace(_config(encoder_parallel="tp", tp_size=3), num_heads=4, head_dim=2).validate()
    # The fused gate|up intermediate dim must divide evenly too.
    with pytest.raises(ValueError, match="intermediate"):
        replace(_config(encoder_parallel="tp", tp_size=2), intermediate_size=9).validate()
    with pytest.raises(ValueError, match="encoder_parallel"):
        replace(_config(), encoder_parallel="data").validate()


@pytest.mark.parametrize("mode", ["tp", "dp"])
@pytest.mark.parametrize("malformed", [False, True])
def test_vision_weight_loading_preserves_encoder_parallel_mode(mode: str, malformed: bool) -> None:
    reference = _make_holder(_config())
    candidate = _make_holder(_config(encoder_parallel=mode, tp_size=2, tp_rank=1))
    checkpoint = {}
    for name, parameter in reference.named_parameters():
        key = name.replace("vision_model.", "vision.")
        if key == "vision.patch_embed.weight":
            key = "vision.patch_embed.proj.weight"
        elif key == "vision.patch_embed.bias":
            key = "vision.patch_embed.proj.bias"
        checkpoint[key] = parameter.detach().clone()

    class Loader(WeightLoader):
        def load_tensor(self, name: str) -> torch.Tensor:
            return checkpoint[name]

    if malformed:
        checkpoint["vision.blocks.0.attn.wqkv.weight"] = torch.zeros(30, 8)
        with pytest.raises((ValueError, RuntimeError), match="shape|size"):
            DeepseekV41ForCausalLM._load_vision_weights(candidate, Loader(candidate, [], 2, 1))
        return
    DeepseekV41ForCausalLM._load_vision_weights(candidate, Loader(candidate, [], 2, 1))
    expected = _make_holder(candidate.vision_cfg)
    _copy_sharded_state(reference, expected)
    for name, parameter in candidate.named_parameters():
        torch.testing.assert_close(parameter, dict(expected.named_parameters())[name])


def test_nonpacked_images_are_aligned_before_next_tower_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    holder = _make_holder(_config())
    original_tower = holder.vision_model.forward_batch
    original_aligner = holder.aligner.forward
    events = []

    def tower(patches: torch.Tensor, grids: list[tuple[int, int]]) -> torch.Tensor:
        events.append("tower")
        return original_tower(patches, grids)

    def aligner(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
        events.append("aligner")
        return original_aligner(value, height, width)

    monkeypatch.setattr(_v41_module, "_VISION_BATCH_MAX_TOTAL_PATCHES", 1)
    monkeypatch.setattr(holder.vision_model, "forward_batch", tower)
    monkeypatch.setattr(holder.aligner, "forward", aligner)
    DeepseekV41ForCausalLM.encode(holder, torch.randn(8 * 4, 3, 2, 2), torch.tensor([[2, 2]] * 8))
    assert events == ["tower", "aligner"] * 8


def test_encoder_tp_checkpoint_slicing_matches_rank_shards() -> None:
    cfg = _config(encoder_parallel="tp", tp_size=2, tp_rank=1)

    wqkv = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)
    expected = torch.cat([wqkv[4:8], wqkv[12:16], wqkv[20:24]], dim=0)
    local = _shard_vision_checkpoint_tensor("vision.blocks.0.attn.wqkv.weight", wqkv, cfg)
    assert torch.equal(local, expected)

    bias = torch.arange(24, dtype=torch.float32)
    expected_bias = torch.cat([bias[4:8], bias[12:16], bias[20:24]])
    local_bias = _shard_vision_checkpoint_tensor("vision.blocks.0.attn.wqkv.bias", bias, cfg)
    assert torch.equal(local_bias, expected_bias)

    wo = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
    local_wo = WeightLoader(torch.nn.Module(), [], cfg.tp_size, cfg.tp_rank).shard(wo, dim=1)
    assert torch.equal(local_wo, wo[:, 4:8])
    # The output-projection bias is replicated, not sliced.
    local_wo_bias = _shard_vision_checkpoint_tensor("vision.blocks.0.attn.wo.bias", bias[:8], cfg)
    assert torch.equal(local_wo_bias, bias[:8])

    w1 = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    local_w1 = _shard_vision_checkpoint_tensor("vision.blocks.0.mlp.w1.weight", w1, cfg)
    assert torch.equal(local_w1, torch.cat([w1[4:8], w1[12:16]], dim=0))

    w2 = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
    local_w2 = WeightLoader(torch.nn.Module(), [], cfg.tp_size, cfg.tp_rank).shard(w2, dim=1)
    assert torch.equal(local_w2, w2[:, 4:8])

    norm = torch.arange(8, dtype=torch.float32)
    assert torch.equal(
        _shard_vision_checkpoint_tensor("vision.blocks.0.norm1.weight", norm, cfg),
        norm,
    )


def test_encoder_tp_matches_replicated(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(7)
    full = _make_holder(_config())
    for parameter in full.parameters():
        parameter.data.normal_(0.0, 0.05)

    world_size = 2
    shards = []
    for rank in range(world_size):
        cfg = replace(full.vision_cfg, encoder_parallel="tp", tp_size=world_size, tp_rank=rank)
        holder = _make_holder(cfg)
        _copy_sharded_state(full, holder)
        shards.append(holder)

    group = _ThreadGroup(world_size)
    monkeypatch.setattr(_v41_module.distributed, "tp_all_reduce", group.all_reduce, raising=False)
    monkeypatch.setattr(_v41_module.distributed, "tp_all_gather", group.all_gather, raising=False)

    pixels = torch.randn(16, 3, 2, 2)
    grid = torch.tensor([[4, 4]])
    expected = DeepseekV41ForCausalLM.encode(full, pixels, grid)
    actual = _run_ranks(
        shards,
        group,
        lambda holder: DeepseekV41ForCausalLM.encode(holder, pixels, grid),
    )
    for output in actual[1:]:
        torch.testing.assert_close(output, actual[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("grids", [[(4, 4)], [(4, 4), (2, 4), (4, 2)]])
def test_encoder_dp_matches_replicated(
    monkeypatch: pytest.MonkeyPatch,
    grids: list[tuple[int, int]],
) -> None:
    torch.manual_seed(11)
    full = _make_holder(_config())
    for parameter in full.parameters():
        parameter.data.normal_(0.0, 0.05)

    world_size = 4
    holders = []
    for rank in range(world_size):
        cfg = replace(full.vision_cfg, encoder_parallel="dp", tp_size=world_size, tp_rank=rank)
        holder = _make_holder(cfg)
        _copy_state(full, holder)
        holders.append(holder)

    group = _ThreadGroup(world_size)
    monkeypatch.setattr(_v41_module.distributed, "tp_all_reduce", group.all_reduce, raising=False)
    monkeypatch.setattr(_v41_module.distributed, "tp_all_gather", group.all_gather, raising=False)

    pixels = torch.cat(
        [torch.randn(n_h * n_w, 3, 2, 2) for n_h, n_w in grids],
        dim=0,
    )
    grid = torch.tensor(grids)
    expected = DeepseekV41ForCausalLM.encode(full, pixels, grid)
    actual = _run_ranks(
        holders,
        group,
        lambda holder: DeepseekV41ForCausalLM.encode(holder, pixels, grid),
    )
    for output in actual[1:]:
        torch.testing.assert_close(output, actual[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected, rtol=1e-4, atol=1e-5)
