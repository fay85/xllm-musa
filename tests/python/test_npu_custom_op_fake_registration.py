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

"""Fake-op registration must stay resilient to stale xLLM binaries.

``xllm/python/kernels_npu/_custom_op.py`` is imported from the source tree at
runtime, while the ``xllm_ops`` C++ schemas come from the compiled binary. A
stale binary (newer Python source, older build) used to make the whole model
runtime fail at boot with ``operator 'xllm_ops::...' is not registered``. The
loader now skips missing operators with a warning so every model can still
start; graph capture through a genuinely missing operator keeps failing.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from xllm.python.kernels_npu import _custom_op, sparse_attention
from xllm.python.kernels_npu.kda import kda_replayssm_correction_fold, kda_replayssm_verify_record


def test_register_fake_missing_operator_with_namespace_warns_and_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale binary (namespace present, schema missing) must warn, not raise."""

    monkeypatch.setattr(_custom_op, "_is_registered", lambda qualname: False)
    monkeypatch.setattr(_custom_op, "_loaded_xllm_ops_runtime", lambda: True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _custom_op.register_fake("xllm_ops::definitely_missing_op_xyz", lambda tensor: tensor)

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "xllm_ops::definitely_missing_op_xyz" in message
    assert "not registered" in message


def test_register_fake_without_namespace_skips_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone/test interpreters (no xLLM ops loaded) must not warn."""

    monkeypatch.setattr(_custom_op, "_is_registered", lambda qualname: False)
    monkeypatch.setattr(_custom_op, "_loaded_xllm_ops_runtime", lambda: False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _custom_op.register_fake("xllm_ops::definitely_missing_op_xyz", lambda tensor: tensor)

    assert caught == []


@pytest.mark.parametrize("layout", ["TND", "BSND"])
def test_smla_fake_rejects_nondivisible_query_heads(layout: str) -> None:
    with FakeTensorMode():
        shape = (3, 15, 512) if layout == "TND" else (1, 3, 15, 512)
        q = torch.empty(shape, dtype=torch.bfloat16)
        kv = torch.empty((1, 128, 2, 512), dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="query heads must be divisible"):
            _custom_op._smla_va_fake(q, ori_kv=kv, layout_q=layout, layout_kv="PA_BBND", return_softmax_lse=True)


@pytest.mark.parametrize(
    "operation",
    [
        "dsa_kv_offloading_update_nope_out",
        "dsa_kv_offloading_update_rope_with_plan_out",
        "dsa_kv_offloading_serve_rope_planned_out",
    ],
)
def test_stable_resident_wrapper_preserves_reserved_payload_and_fake_aliases(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, capacity = 2, 4
    width = sparse_attention.SELECTION_CAPACITY
    page = sparse_attention.RESIDENT_PAGE_SIZE
    host = torch.zeros((2, page, 1, 512), dtype=torch.bfloat16)
    block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    descriptor = torch.tensor([[0, 1, page, 0], [1, 1, page, 3]], dtype=torch.int32)
    current_map = torch.full((rows, capacity), -1, dtype=torch.int32)
    current_map[:, 0] = torch.arange(rows, dtype=torch.int32)
    topk = torch.full((rows, 1, width), -1, dtype=torch.int32)
    append = torch.tensor([page - 1, 2 * page - 1], dtype=torch.int32)
    current = torch.zeros((rows, 1, 512), dtype=torch.bfloat16)
    selection = torch.zeros((capacity, width, 1, 512), dtype=torch.bfloat16)
    indices = torch.zeros((rows, 1, width), dtype=torch.int32)
    lengths = torch.zeros(rows, dtype=torch.int32)
    table = torch.zeros((rows, width // page), dtype=torch.int32)
    resident = torch.full((capacity, 1, width), -1, dtype=torch.int32)
    slots = torch.zeros_like(resident)
    count = torch.zeros(capacity, dtype=torch.int32)
    compact = torch.zeros_like(indices)
    host_rope = torch.zeros((2, page, 1, 64), dtype=torch.bfloat16)
    current_rope = torch.zeros((rows, 1, 64), dtype=torch.bfloat16)
    selection_rope = torch.zeros((capacity, width, 1, 64), dtype=torch.bfloat16)
    plan_tokens = torch.zeros_like(indices)
    plan_slots = torch.zeros_like(indices)
    calls: list[tuple[torch.Tensor, ...]] = []
    fake = getattr(_custom_op, f"_{operation}_fake")

    def native_call(*args: torch.Tensor) -> object:
        calls.append(args)
        result = fake(*args)
        if operation == "dsa_kv_offloading_update_nope_out":
            assert all(value is args[index] for value, index in zip(result, (7, 14, 9, 10)))
        else:
            assert result is None
        return result

    monkeypatch.setattr(torch.ops.xllm_ops, operation, native_call, raising=False)
    update_args = (
        host,
        block_table,
        descriptor,
        current_map,
        topk,
        append,
        current,
        selection,
        indices,
        lengths,
        table,
        resident,
        slots,
        count,
    )
    wrapper = getattr(sparse_attention, operation)
    if operation == "dsa_kv_offloading_update_nope_out":
        partial_payload = list(update_args)
        partial_payload[7] = torch.zeros(capacity * width * 512 + 1, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="selection payload must contain whole resident rows"):
            wrapper(*partial_payload, compact)
        mismatched = list(update_args)
        mismatched[13] = torch.zeros(capacity - 1, dtype=torch.int32)
        with pytest.raises(RuntimeError, match="selection_valid_count"):
            wrapper(*mismatched, compact)
        for map_shape in ((rows - 1, capacity), (rows, 1), (rows * capacity,)):
            invalid_map = list(update_args)
            invalid_map[3] = torch.zeros(map_shape, dtype=torch.int32)
            with pytest.raises(RuntimeError, match="current row map") as error:
                wrapper(*invalid_map, compact)
            assert f"got {map_shape}" in str(error.value)
        assert not calls
    elif operation == "dsa_kv_offloading_update_rope_with_plan_out":
        bad_topk = torch.zeros((rows, 2, width), dtype=torch.int32)
        with pytest.raises(RuntimeError, match="padded top-k"):
            wrapper(
                host,
                block_table,
                descriptor,
                current_map,
                bad_topk,
                append,
                current,
                selection,
                indices,
                lengths,
                table,
                resident,
                slots,
                count,
                host_rope,
                current_rope,
                selection_rope,
                plan_tokens,
                plan_slots,
            )
    else:
        bad_plan = torch.zeros((rows, 2, width), dtype=torch.int32)
        with pytest.raises(RuntimeError, match="padded top-k"):
            wrapper(
                host,
                host_rope,
                block_table,
                descriptor,
                current_map,
                bad_plan,
                plan_slots,
                append,
                current,
                current_rope,
                selection,
                selection_rope,
            )
    if operation == "dsa_kv_offloading_update_nope_out":
        result = wrapper(*update_args, compact)
        assert all(value is expected for value, expected in zip(result, (selection, compact, lengths, table)))
    elif operation == "dsa_kv_offloading_update_rope_with_plan_out":
        assert wrapper(*update_args, host_rope, current_rope, selection_rope, plan_tokens, plan_slots) is None
    else:
        assert (
            wrapper(
                host,
                host_rope,
                block_table,
                descriptor,
                current_map,
                plan_tokens,
                plan_slots,
                append,
                current,
                current_rope,
                selection,
                selection_rope,
            )
            is None
        )
    assert len(calls) == 1
    args = calls[0]
    payload = args[10] if operation == "dsa_kv_offloading_serve_rope_planned_out" else args[7]
    assert payload.shape == (capacity * (width // page), page, 1, 512)
    assert payload.data_ptr() == selection.data_ptr()
    row_map = args[4] if operation == "dsa_kv_offloading_serve_rope_planned_out" else args[3]
    assert row_map.shape == (rows, rows)
    assert torch.equal(row_map, current_map[:, :rows])
    if operation != "dsa_kv_offloading_serve_rope_planned_out":
        assert args[11].shape == args[12].shape == (capacity, 1, width)
        assert args[13].shape == (capacity,)
    if operation == "dsa_kv_offloading_update_rope_with_plan_out":
        assert args[14] is host_rope
        assert args[16].shape == (capacity * (width // page), page, 1, 64)
        assert args[17].shape == args[18].shape == (rows, 1, width)


def _check_replayssm_fake_dispatch(*, require_native: bool = False) -> None:
    specifications = (
        (
            "kda_replayssm_verify_record",
            _custom_op._kda_replayssm_verify_record_fake,
            ("normalized_keys", "corrections", "decays"),
        ),
        ("kda_replayssm_correction_fold", _custom_op._kda_replayssm_correction_fold_fake, ("states",)),
    )
    native = (
        require_native
        or _custom_op._loaded_xllm_ops_runtime()
        or any(hasattr(torch.ops.xllm_ops, name) for name, _, _ in specifications)
    )
    if not native:
        source_path = Path(__file__).resolve().parents[2] / "xllm/core/kernels/npu/npu_ops_library.cpp"
        source = source_path.read_text(encoding="utf-8")
    with torch.library._scoped_library("xllm_ops", "FRAGMENT") as library, FakeTensorMode():
        for name, fake_impl, written in specifications:
            qualified = f"xllm_ops::{name}"
            if native:
                assert torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "PrivateUse1"), qualified
            else:
                definition = re.search(rf'm\.def\(\s*("{name}\([^;]+?)\s*\);', source)
                assert definition is not None
                schema_text = "".join(
                    ast.literal_eval(literal) for literal in re.findall(r'"[^"]*"', definition.group(1))
                )
                library.define(schema_text)
                torch.library.register_fake(qualified, fake_impl, lib=library)
            assert torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "Meta"), qualified
            schema = getattr(torch.ops.xllm_ops, name).default._schema
            assert tuple(arg.name for arg in schema.arguments if arg.alias_info and arg.alias_info.is_write) == written
            if name == "kda_replayssm_correction_fold":
                assert "Tensor(a!)[] states" in str(schema)

        query = torch.empty((16, 3, 8, 128), dtype=torch.bfloat16)[:, 0]
        state = torch.empty((4, 8, 128, 128), dtype=torch.float32)
        records = [torch.empty((4, 8, 8, 128), dtype=torch.float32) for _ in range(3)]
        indices = torch.empty(2, dtype=torch.int32)
        output = kda_replayssm_verify_record(
            query,
            query,
            query,
            query.contiguous(),
            torch.empty((16, 8), dtype=torch.bfloat16),
            state,
            torch.empty(8),
            torch.empty(8 * 128),
            indices,
            *records,
        )
        assert output.shape == query.shape
        assert output.dtype == query.dtype
        assert output.device == query.device
        assert output.is_contiguous()
        assert kda_replayssm_correction_fold([state], *([record] for record in records), indices, indices) is None


def test_replayssm_schemas_dispatch_with_fake_tensors() -> None:
    library_path = os.environ.get("XLLM_REPLAYSSM_FOLD_RUNTIME_TEST_LIBRARY") or os.environ.get("XLLM_EXPORT_PATH")
    if not library_path:
        _check_replayssm_fake_dispatch()
        return

    # Runtime initialization assumes that native schemas are loaded before
    # _custom_op is first imported. Pytest collection may have imported it.
    script = """
import runpy
import sys

import torch
import torch_npu

torch.ops.load_library(sys.argv[1])
if hasattr(torch.ops.xllm_ops, "rms_norm"):
    from xllm.python.kernels_npu import _initialize_runtime

    _initialize_runtime()
else:
    # An operator-only test library supplies just the ReplaySSM schemas.
    from xllm.python.kernels_npu import _custom_op

runpy.run_path(sys.argv[2])["_check_replayssm_fake_dispatch"](require_native=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, library_path, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
