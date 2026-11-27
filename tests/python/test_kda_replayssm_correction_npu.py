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

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from xllm.python.kernels_npu.kda import (
    KdaSafeGateParameters,
    build_kda_replayssm_state_indices,
    kda_replayssm_correction_fold,
    kda_replayssm_verify_record,
    recurrent_kda_verify,
)


def _operator_library() -> str | None:
    return os.environ.get("XLLM_REPLAYSSM_FOLD_RUNTIME_TEST_LIBRARY") or os.environ.get("XLLM_EXPORT_PATH")


def _load_operator_library(library_path: str) -> None:
    # FLA prepends an OPP root. Prioritize vendor directories when the
    # xLLM resolver snapshots ASCEND_CUSTOM_OPP_PATH at library load.
    opp_env = "ASCEND_CUSTOM_OPP_PATH"
    entries = [entry for entry in os.environ.get(opp_env, "").split(os.pathsep) if entry]
    entries.sort(key=lambda entry: not (Path(entry) / "op_api/lib/libcust_opapi.so").is_file())
    with pytest.MonkeyPatch.context() as env:
        if entries:
            env.setenv(opp_env, os.pathsep.join(entries))
        torch.ops.load_library(library_path)


@pytest.fixture(scope="module")
def _replayssm_npu() -> None:
    device_name = os.environ.get("XLLM_KDA_TEST_NPU_DEVICE")
    if device_name is None:
        pytest.skip("requires an explicitly selected NPU")
    pytest.importorskip("torch_npu")
    device = torch.device(device_name)
    torch.npu.set_device(device)
    soc_name = torch.npu.get_device_name(device)
    if not soc_name.lower().startswith("ascend910_93"):
        pytest.skip(f"ReplaySSM requires an Ascend 910_93 NPU, got {soc_name}")
    pytest.importorskip("fla_npu.ops.ascendc")
    library_path = _operator_library()
    if library_path:
        _load_operator_library(library_path)
    for name in ("kda_replayssm_verify_record", "kda_replayssm_correction_fold"):
        assert hasattr(torch.ops.xllm_ops, name), f"{name} requires the rebuilt xLLM operator library"


def test_replayssm_state_indices_require_distinct_live_slots() -> None:
    with pytest.raises(ValueError, match="duplicate ReplaySSM slot 1"):
        build_kda_replayssm_state_indices([1, 1])

    indices = build_kda_replayssm_state_indices([1, None, None, 0])
    assert torch.equal(indices, torch.tensor([1, -1, -1, 0], dtype=torch.int32))


@pytest.mark.usefixtures("_replayssm_npu")
@pytest.mark.parametrize("graph_mode", (False, True), ids=("dense-eager", "packed-graph"))
@torch.inference_mode()
def test_correction_records_match_native_outputs_and_accepted_checkpoints(
    graph_mode: bool,
) -> None:
    batch = 9
    generator = torch.Generator().manual_seed(20261115)
    heads, dimension, window, slots = 8, 128, 8, batch + 2
    tokens, features = batch * window, heads * dimension
    packed = torch.randn(tokens, 3 * features, generator=generator).bfloat16().npu()
    qkv = [tensor.view(tokens, heads, dimension) for tensor in packed.split(features, dim=-1)]
    query, key, value = qkv if graph_mode else [tensor.contiguous() for tensor in qkv]
    query[0].zero_()
    key[0].zero_()
    key[1].zero_()
    key[1, :, dimension // 2] = 2**-14
    key[2].fill_(2**-80)
    gate = torch.randn(tokens, heads, dimension, generator=generator).bfloat16().npu()
    beta = torch.randn(tokens, heads, generator=generator).bfloat16().npu()
    a_log = torch.randn(heads, generator=generator).npu()
    bias = torch.randn(features, generator=generator).npu()
    original = (torch.randn(slots, heads, dimension, dimension, generator=generator) * 0.01).npu()
    readonly_state = original.clone()
    folded_state = original.clone()
    native_pool = original.repeat_interleave(window, dim=0)
    record_shape = (slots, window, heads, dimension)
    normalized_keys, corrections, decays = [torch.full(record_shape, 42.0, device="npu") for _ in range(3)]
    indices = build_kda_replayssm_state_indices(range(1, batch + 1)).npu()
    accepted = torch.ones(batch, dtype=torch.int32, device="npu")
    checkpoints = indices[:, None] * window + torch.arange(window, dtype=torch.int32, device="npu")[None, :]
    starts = (torch.arange(batch + 1, dtype=torch.int32) * window).npu()
    prior_accepted = torch.ones_like(accepted)
    safe_gate = KdaSafeGateParameters(a_log, bias, -5.0)

    def verify() -> torch.Tensor:
        return kda_replayssm_verify_record(
            query,
            key,
            value,
            gate,
            beta,
            readonly_state,
            a_log,
            bias,
            indices,
            normalized_keys,
            corrections,
            decays,
        )

    def native() -> torch.Tensor:
        return recurrent_kda_verify(
            query,
            key,
            value,
            gate,
            beta,
            native_pool,
            starts,
            checkpoints,
            prior_accepted,
            safe_gate=safe_gate,
        )

    def fold() -> None:
        kda_replayssm_correction_fold([folded_state], [normalized_keys], [corrections], [decays], indices, accepted)

    graphs: dict[str, object] = {}
    outputs: dict[str, torch.Tensor] = {}
    if graph_mode:
        for name, launch in (("verify", verify), ("native", native), ("fold", fold)):
            launch()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = launch()
                if result is not None:
                    outputs[name] = result
            graphs[name] = graph
    for iteration in range(4):
        original.copy_(torch.randn(original.shape, generator=generator) * 0.01)
        readonly_state.copy_(original)
        folded_state.copy_(original)
        native_pool.copy_(original.repeat_interleave(window, dim=0))
        selected = [(slot + iteration) % slots for slot in range(batch, 0, -1)]
        indices.copy_(build_kda_replayssm_state_indices(selected))
        checkpoints.copy_(indices[:, None] * window + torch.arange(window, dtype=torch.int32, device="npu")[None, :])
        accepted.copy_((torch.arange(batch, dtype=torch.int32) + iteration) % (window + 1))
        beta.copy_(torch.randn(beta.shape, generator=generator).bfloat16())
        beta[0].fill_(-30)
        beta[1].fill_(30)
        for record in (normalized_keys, corrections, decays):
            record.fill_(42.0)
        for name, launch in (("verify", verify), ("native", native), ("fold", fold)):
            if graph_mode:
                graphs[name].replay()
            else:
                result = launch()
                if result is not None:
                    outputs[name] = result
        torch.npu.synchronize()
        assert torch.equal(
            outputs["verify"].cpu().view(torch.int16).reshape(-1),
            outputs["native"].cpu().view(torch.int16).reshape(-1),
        )
        assert torch.equal(readonly_state.cpu().view(torch.int32), original.cpu().view(torch.int32))
        committed = accepted > 0
        committed_indices = indices[committed].to(torch.int64)
        source_rows = committed_indices * window + accepted[committed].to(torch.int64) - 1
        expected = original.clone()
        expected.index_copy_(0, committed_indices, native_pool.index_select(0, source_rows))
        assert torch.equal(folded_state.cpu().view(torch.int32), expected.cpu().view(torch.int32))
        untouched = [slot for slot in range(slots) if slot not in selected]
        for record in (normalized_keys, corrections, decays):
            assert torch.all(record[untouched].cpu() == 42.0)

    readonly_state.copy_(original)
    folded_state.copy_(original)
    native_pool.copy_(original.repeat_interleave(window, dim=0))
    prior_accepted.fill_(1)
    for chain_step in range(2):
        accepted.copy_((torch.arange(batch, dtype=torch.int32) + chain_step + 2) % window + 1)
        for name, launch in (("verify", verify), ("native", native), ("fold", fold)):
            if graph_mode:
                graphs[name].replay()
            else:
                result = launch()
                if result is not None:
                    outputs[name] = result
        torch.npu.synchronize()
        assert torch.equal(
            outputs["verify"].cpu().view(torch.int16).reshape(-1),
            outputs["native"].cpu().view(torch.int16).reshape(-1),
        )
        source_rows = indices.to(torch.int64) * window + accepted.to(torch.int64) - 1
        expected = readonly_state.clone()
        expected.index_copy_(0, indices.to(torch.int64), native_pool.index_select(0, source_rows))
        assert torch.equal(folded_state.cpu().view(torch.int32), expected.cpu().view(torch.int32))
        readonly_state.copy_(folded_state)
        prior_accepted.copy_(accepted)


@pytest.mark.usefixtures("_replayssm_npu")
@torch.inference_mode()
def test_two_layer_correction_fold_matches_native_checkpoints_and_slot_bounds() -> None:
    heads, dimension, window = 8, 128, 8
    slot_capacities = (3, 5)
    indices_cpu = build_kda_replayssm_state_indices([2, 0, 3, None, 5])
    accepted_cpu = torch.tensor([window + 2, -1, 3, window, 1], dtype=torch.int32)
    indices, accepted = indices_cpu.npu(), accepted_cpu.npu()
    batch = indices_cpu.numel()
    states: list[torch.Tensor] = []
    state_buffers: list[torch.Tensor] = []
    expected_buffers: list[torch.Tensor] = []
    records_by_layer: list[list[torch.Tensor]] = []
    for layer, slots in enumerate(slot_capacities):
        generator = torch.Generator().manual_seed(20261116 + layer)
        state_buffer = (torch.randn(slots + 1, heads, dimension, dimension, generator=generator) * 0.01).npu()
        state = state_buffer[:slots]
        original = state.clone()
        states.append(state)
        state_buffers.append(state_buffer)
        query, key, value, gate = [
            torch.randn(batch * window, heads, dimension, generator=generator).bfloat16().npu() for _ in range(4)
        ]
        beta = torch.randn(batch * window, heads, generator=generator).bfloat16().npu()
        a_log = torch.randn(heads, generator=generator).npu()
        bias = torch.randn(heads * dimension, generator=generator).npu()
        records = [torch.full((slots, window, heads, dimension), 42.0 + layer, device="npu") for _ in range(3)]
        records_by_layer.append(records)
        output = kda_replayssm_verify_record(
            query,
            key,
            value,
            gate,
            beta,
            original,
            a_log,
            bias,
            indices,
            *records,
        )
        valid = (indices_cpu >= 0) & (indices_cpu < slots)
        assert torch.count_nonzero(output.view(batch, window, heads, dimension).cpu()[~valid]) == 0
        assert torch.equal(state.cpu().view(torch.int32), original.cpu().view(torch.int32))
        valid_slots = indices_cpu[valid]
        untouched = [slot for slot in range(slots) if slot not in valid_slots.tolist()]
        for record in records:
            assert torch.all(record.cpu()[untouched] == 42.0 + layer)
        token_rows = torch.arange(batch * window).view(batch, window)[valid].reshape(-1).npu()
        native_pool = original.repeat_interleave(window, dim=0)
        checkpoints = (valid_slots[:, None] * window + torch.arange(window, dtype=torch.int32)[None, :]).npu()
        recurrent_kda_verify(
            *[tensor.index_select(0, token_rows) for tensor in (query, key, value, gate, beta)],
            native_pool,
            (torch.arange(valid_slots.numel() + 1, dtype=torch.int32) * window).npu(),
            checkpoints,
            torch.ones(valid_slots.numel(), dtype=torch.int32, device="npu"),
            safe_gate=KdaSafeGateParameters(a_log, bias, -5.0),
        )
        committed = valid & (accepted_cpu > 0)
        committed_slots = indices_cpu[committed].to(torch.int64)
        checkpoint_rows = committed_slots * window + accepted_cpu[committed].clamp(max=window).to(torch.int64) - 1
        expected = state_buffer.clone()
        expected.index_copy_(0, committed_slots.npu(), native_pool.index_select(0, checkpoint_rows.npu()))
        expected_buffers.append(expected)

    records = [record for layer_records in records_by_layer for record in layer_records]
    saved_records = [record.clone() for record in records]
    keys, corrections, decays = zip(*records_by_layer, strict=True)
    kda_replayssm_correction_fold(states, keys, corrections, decays, indices, accepted)
    empty = indices[:0]
    kda_replayssm_correction_fold(states, keys, corrections, decays, empty, empty)
    torch.npu.synchronize()
    for actual, expected in zip(state_buffers, expected_buffers):
        assert torch.equal(actual.cpu().view(torch.int32), expected.cpu().view(torch.int32))
    for actual, expected in zip(records, saved_records):
        assert torch.equal(actual.cpu().view(torch.int32), expected.cpu().view(torch.int32))


def _launch_with_blocked_state(operation: str, library_path: str) -> None:
    import resource

    import torch_npu

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    _load_operator_library(library_path)
    torch.npu.set_device(torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"]))
    torch.npu.config.allow_internal_format = True
    nc1hwc0 = 3
    state = torch_npu.empty_with_format((2, 8, 128, 128), dtype=torch.float32, device="npu", acl_format=nc1hwc0)
    assert state.is_contiguous() and torch_npu.get_npu_format(state) == nc1hwc0
    keys, corrections, decays = [torch.empty((2, 8, 8, 128), device="npu") for _ in range(3)]
    indices = torch.tensor([1], dtype=torch.int32, device="npu")
    if operation == "verify":
        qkv = torch.empty((8, 8, 128), dtype=torch.bfloat16, device="npu")
        beta = torch.empty((8, 8), dtype=torch.bfloat16, device="npu")
        a_log = torch.empty(8, device="npu")
        bias = torch.empty(8 * 128, device="npu")
        kda_replayssm_verify_record(qkv, qkv, qkv, qkv, beta, state, a_log, bias, indices, keys, corrections, decays)
    else:
        kda_replayssm_correction_fold([state], [keys], [corrections], [decays], indices, indices)
    torch.npu.synchronize()


@pytest.mark.usefixtures("_replayssm_npu")
@pytest.mark.parametrize("operation", ("verify", "fold"))
def test_replayssm_rejects_blocked_state(operation: str, tmp_path: Path) -> None:
    library_path = _operator_library()
    if not library_path:
        pytest.skip("requires the native operator library path for an isolated CHECK failure")
    log_path = tmp_path / "blocked-state.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy, sys; runpy.run_path(sys.argv[1])['_launch_with_blocked_state'](*sys.argv[2:])",
                str(Path(__file__).resolve()),
                operation,
                library_path,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == -signal.SIGABRT, log_text
    assert "ReplaySSM requires base NPU storage format" in log_text
