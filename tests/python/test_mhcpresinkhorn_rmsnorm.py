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

"""CPU-side tests for the PyPTO-Pro mhcpresinkhorn_rmsnorm host wrapper.

The kernel itself needs torch_npu and pypto_pro (see
test_mhcpresinkhorn_rmsnorm_npu.py). This module covers the parts that must
hold on any machine:

* tiling / layout constants stay mutually consistent,
* every argument-contract violation raises ValueError before device work,
* the pinned and scratch workspace caches follow the current policy,
* the kernel is launched with the documented buffer contract, including
  the t=1024 prefill shape.

torch_npu, pypto_pro and the tensors are replaced by lightweight stubs so the
module can be imported without an Ascend device. The wrapper only reads dtype,
shape, dim(), device, is_contiguous(), data_ptr(), to(), view(), reshape() and
a leading slice from its inputs.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

_MODULE_NAME = "xllm.python.kernels_npu.pypto_pro_mhcpresinkhorn_rmsnorm"
_SOURCE = (Path(__file__).parents[2] / "xllm" / "python" / "kernels_npu"
           / "pypto-pro" / "mhcpresinkhorn_rmsnorm.py")

# torch.device("npu:0") is only parsed once the privateuse1 backend is
# renamed; torch_npu does that when it is imported for real.
torch.utils.backend_registration._rename_privateuse1_backend("npu")


class _StubTensor:
    """Stand-in for a device tensor: shape/dtype/device/contiguity only."""

    _next_ptr = 1

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype,
                 device: torch.device, *, contiguous: bool = True,
                 view_of: "_StubTensor | None" = None) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous
        self._view_of = view_of
        if view_of is None:
            self._ptr = _StubTensor._next_ptr
            _StubTensor._next_ptr += 1
        else:
            self._ptr = view_of.data_ptr()

    def dim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def data_ptr(self) -> int:
        return self._ptr

    def to(self, device: torch.device) -> "_StubTensor":
        if self.device == device:
            return self
        return _StubTensor(self.shape, self.dtype, device,
                           contiguous=self._contiguous, view_of=self)

    def view(self, *shape: int) -> "_StubTensor":
        return _StubTensor(shape, self.dtype, self.device,
                           contiguous=self._contiguous, view_of=self)

    def reshape(self, *shape: int) -> "_StubTensor":
        return _StubTensor(shape, self.dtype, self.device,
                           contiguous=self._contiguous, view_of=self)

    def t(self) -> "_StubTensor":
        return _StubTensor(tuple(reversed(self.shape)), self.dtype, self.device,
                           contiguous=False, view_of=self)

    def __getitem__(self, item: slice) -> "_StubTensor":
        if not isinstance(item, slice):
            raise TypeError("only slice views are supported")
        start, stop, step = item.indices(self.shape[0])
        length = len(range(start, stop, step))
        return _StubTensor((length,) + self.shape[1:], self.dtype, self.device,
                           contiguous=self._contiguous, view_of=self)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_StubTensor({self.shape}, {self.dtype}, {self.device})"


def _tensor(shape: tuple[int, ...], dtype: torch.dtype,
            device: torch.device, *, contiguous: bool = True) -> _StubTensor:
    return _StubTensor(shape, dtype, device, contiguous=contiguous)


def _npu_device(index: int = 0) -> torch.device:
    return torch.device("npu", index)


class _FakeStream:
    def __init__(self, stream_id: int) -> None:
        self.stream_id = stream_id


class _FakeNpu:
    """Minimal torch.npu stand-in used by the host wrapper."""

    def __init__(self) -> None:
        self.current_device_id = 0
        self.capturing = False
        self.allocations: list[tuple[tuple[int, ...], torch.dtype,
                                     torch.device]] = []
        self.synchronized: list[Any] = []
        self.streams: dict[int, _FakeStream] = {}

    def set_device(self, device: torch.device) -> None:
        self.current_device_id = (device.index if device.index is not None
                                  else 0)

    def current_device(self) -> int:
        return self.current_device_id

    def current_stream(self, device: torch.device | None = None) -> _FakeStream:
        if device is None or device.index is None:
            index = self.current_device_id
        else:
            index = device.index
        return self.streams.setdefault(index, _FakeStream(index))

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    def empty(self, shape: tuple[int, ...], dtype: torch.dtype,
              device: torch.device) -> _StubTensor:
        self.allocations.append((tuple(shape), dtype, device))
        return _tensor(tuple(shape), dtype, device)

    def synchronize(self, device: Any = None) -> None:
        self.synchronized.append(device)


class _FakeKernel:
    """Records kernel launches so the wrapper's call contract is checkable."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...]]] = []
        self.grid: Any = None

    def __getitem__(self, grid: Any) -> "_FakeKernel":
        self.grid = grid
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((self.grid, args))


class _FakeAnnotation:
    """Anything pypto_pro.language exposes: subscriptable and callable."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __getitem__(self, item: Any) -> "_FakeAnnotation":
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return _FakeAnnotation(self.name)

    def __getattr__(self, name: str) -> "_FakeAnnotation":
        if name.startswith("__"):
            raise AttributeError(name)
        return _FakeAnnotation(f"{self.name}.{name}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_FakeAnnotation({self.name})"


class _FakeLanguage(types.ModuleType):
    """pypto_pro.language stub: every attribute resolves to a fake."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return _FakeAnnotation(name)


@pytest.fixture()
def op(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the operator module against stubbed torch_npu/pypto_pro."""
    fake_npu = _FakeNpu()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "empty", fake_npu.empty, raising=False)
    monkeypatch.setenv("TILE_FWK_DEVICE_ID", "0")
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))

    language = _FakeLanguage("pypto_pro.language")
    language.Vf = _FakeLanguage("pypto_pro.language.Vf")
    pypto_pro = types.ModuleType("pypto_pro")
    pypto_pro.language = language
    monkeypatch.setitem(sys.modules, "pypto_pro", pypto_pro)
    monkeypatch.setitem(sys.modules, "pypto_pro.language", language)
    monkeypatch.setitem(sys.modules, "pypto_pro.language.Vf", language.Vf)

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    module._WS_CACHE.clear()
    module._WS_SCRATCH_CACHE.clear()
    module._fake_npu = fake_npu
    module._fake_kernel = _FakeKernel()
    module.mhcpresinkhorn_rmsnorm_kernel_k = module._fake_kernel
    _StubTensor._next_ptr = 1
    yield module
    sys.modules.pop(_MODULE_NAME, None)


def _valid_args(device: torch.device, t: int = 4) -> dict[str, Any]:
    return dict(
        x=_tensor((t, 4, 4096), torch.bfloat16, device),
        hc_fn=_tensor((24, 16384), torch.bfloat16, device),
        hc_scale=_tensor((3,), torch.float32, device),
        hc_base=_tensor((24,), torch.float32, device),
        gamma=_tensor((4096,), torch.bfloat16, device),
    )


def test_module_constants_are_mutually_consistent(op: Any) -> None:
    assert op.HC == 4
    assert op.D == 4096
    assert op.HC_D == op.HC * op.D == 16384
    assert op.MIX_HC == (2 + op.HC) * op.HC == 24
    assert op.SINKHORN_ITERS == 20
    assert op.EPS == 1e-6
    assert op.RMS_EPS == 1e-6
    assert op.INV_HC_D == 1.0 / op.HC_D
    assert op.INV_D == 1.0 / op.D
    assert op.NDC == op.D // op.DC == 8
    assert op.NDC * op.DC == op.D
    assert op.PWC == op.MIX_HC == 24
    assert op.NAIV == 2 * op.NB == 56
    assert op.TPC == op.GPCH * op.TG
    assert op.SGRP > 0 and op.NT > 0
    assert op.KLAST == op.HC_D - (op.NB - 1) * op.KA == 400
    assert (op.NB - 1) * op.KA + op.KLAST == op.HC_D
    assert op.KA % 16 == 0 and op.KLAST % 16 == 0
    assert op.MIXC >= op.MIX_HC
    assert op.TPC == 512
    assert op._WS_SCRATCH_MAX == 8


def test_comb16_rows_cover_every_kernel_write(op: Any) -> None:
    """The host allocation must cover the last chunk's padded AIV window."""

    def max_written_row(t: int) -> int:
        ng = (t + op.TG - 1) // op.TG
        nch = (ng + op.GPCH - 1) // op.GPCH
        end = 0
        for c in range(nch):
            jc = c * op.TPC
            tc = min(op.TPC, t - jc)
            wc = (tc + op.NAIV - 1) // op.NAIV
            for b in range((wc + op.NT - 1) // op.NT):
                nt = min(op.NT, wc - b * op.NT)
                for gidx in range(op.NAIV):
                    end = max(end, jc + gidx * wc + b * op.NT + nt)
        return end

    # The old NAIV-aligned allocation missed the chunk-start misalignment for
    # these t values (t=1024 wrote rows 1064..1071 past a 1064-row buffer).
    assert op._comb16_rows(1024) == 1072
    assert op._comb16_rows(2048) == 2096
    for t in list(range(1, 1200)) + [2048, 4096, 8192, 16384]:
        tpad = op._comb16_rows(t)
        assert tpad >= t
        assert tpad >= max_written_row(t)


def test_ub_layout_does_not_overlap(op: Any) -> None:
    """Every UB buffer must fit without overlapping the next one."""
    layout = [
        ("SSQR_UB", op.SSQR_UB, 1 * 8 * 4),
        ("SCALE3_UB", op.SCALE3_UB, 1 * 8 * 4),
        ("SCALE24_UB", op.SCALE24_UB, 1 * 32 * 4),
        ("PMUL24_UB", op.PMUL24_UB, 1 * 32 * 4),
        ("EPS24_UB", op.EPS24_UB, 1 * 32 * 4),
        ("BASE_UB", op.BASE_UB, 1 * 32 * 4),
        ("ZBAT_UB", op.ZBAT_UB, op.NT * op.MIXC * 4),
        ("ZSD_UB", op.ZSD_UB, op.MIXC * op.NT * 4),
        ("CBBAT_UB", op.CBBAT_UB, 16 * op.NT * 4),
        ("CBTT_UB", op.CBTT_UB, op.NT * 16 * 4),
        ("PRE_UB", op.PRE_UB, 1 * 8 * 4),
        ("POST_UB", op.POST_UB, 1 * 8 * 4),
        ("XSUB_UB", op.XSUB_UB, 2 * op.HC * op.D * 2),
        ("YUB_UB", op.YUB_UB, 2 * 1 * op.D * 2),
        ("YROW_UB", op.YROW_UB, 1 * op.D * 2),
        ("GAMMA_UB", op.GAMMA_UB, 1 * op.D * 2),
        ("RSTD_UB", op.RSTD_UB, 1 * 64 * 4),
        ("SLAB_UB", op.SLAB_UB, op.NB * op.SGRP * op.PWC * 4),
        ("MIXR_UB", op.MIXR_UB, 1 * op.SGRP * op.PWC * 4),
    ]
    spans = sorted((start, start + size, name)
                   for name, start, size in layout)
    for (_, end, name), (next_start, _, next_name) in zip(spans, spans[1:]):
        assert end <= next_start, f"{name} overlaps {next_name}"
    assert spans[-1][1] <= 248 * 1024


def test_get_workspaces_shapes_and_views(op: Any) -> None:
    device = _npu_device()
    pmix3, pmix3f, r_ws = op._get_workspaces(40, device)
    ng = (40 + op.TG - 1) // op.TG
    rows = ng * op.TG + 64
    assert tuple(pmix3.shape) == (op.NB, rows, op.PWC)
    assert tuple(pmix3f.shape) == (op.NB, rows * op.PWC)
    assert tuple(r_ws.shape) == (40, 8)
    assert pmix3.dtype == pmix3f.dtype == r_ws.dtype == torch.float32
    assert pmix3f._view_of is pmix3
    assert pmix3.device == device and r_ws.device == device
    assert op._fake_npu.allocations == [
        ((40, 8), torch.float32, device),
        ((op.NB, rows, op.PWC), torch.float32, device),
    ]


def test_get_workspaces_reuses_entry(op: Any) -> None:
    device = _npu_device()
    first = op._get_workspaces(40, device)
    allocations = len(op._fake_npu.allocations)
    second = op._get_workspaces(40, device)
    assert second is first
    assert len(op._fake_npu.allocations) == allocations


def test_ws_key_includes_device_and_stream(op: Any) -> None:
    device = _npu_device(1)
    op._fake_npu.streams[1] = _FakeStream(7)
    assert op._ws_key(32, device) == (32, 1, 7)
    op._fake_npu.streams[1] = _FakeStream(9)
    assert op._ws_key(32, device) == (32, 1, 9)


def test_eager_shapes_use_bounded_lru(op: Any) -> None:
    device = _npu_device()
    buckets = (5, 10, 40, 80, 160, 240, 320, 384, 400)
    for t in buckets:
        op._get_workspaces(t, device)
    assert op._WS_CACHE == {}
    assert len(op._WS_SCRATCH_CACHE) == op._WS_SCRATCH_MAX
    assert (5, 0, 0) not in op._WS_SCRATCH_CACHE
    assert (384, 0, 0) in op._WS_SCRATCH_CACHE


def test_scratch_cache_is_bounded(op: Any) -> None:
    device = _npu_device()
    for t in (2048, 4096, 8192):
        op._get_workspaces(t, device)
    assert len(op._WS_SCRATCH_CACHE) == 3
    assert (2048, 0, 0) in op._WS_SCRATCH_CACHE
    assert (4096, 0, 0) in op._WS_SCRATCH_CACHE
    assert (8192, 0, 0) in op._WS_SCRATCH_CACHE
    assert op._WS_CACHE == {}
    for t in range(10000, 10000 + op._WS_SCRATCH_MAX):
        op._get_workspaces(t, device)
    assert len(op._WS_SCRATCH_CACHE) == op._WS_SCRATCH_MAX
    assert (2048, 0, 0) not in op._WS_SCRATCH_CACHE
    assert (10000 + op._WS_SCRATCH_MAX - 1, 0, 0) in op._WS_SCRATCH_CACHE


def test_scratch_cache_is_true_lru(op: Any) -> None:
    device = _npu_device()
    for t in range(100, 100 + op._WS_SCRATCH_MAX):
        op._get_workspaces(t, device)
    op._get_workspaces(100, device)  # hit: 100 becomes most recently used
    op._get_workspaces(200, device)  # miss: evicts 101, not 100
    assert (101, 0, 0) not in op._WS_SCRATCH_CACHE
    assert (100, 0, 0) in op._WS_SCRATCH_CACHE
    assert (200, 0, 0) in op._WS_SCRATCH_CACHE


def test_capture_pins_large_t(op: Any) -> None:
    device = _npu_device()
    op._fake_npu.capturing = True
    op._get_workspaces(2048, device)
    assert (2048, 0, 0) in op._WS_CACHE
    assert op._WS_SCRATCH_CACHE == {}


def test_t1024_wrapper_launch_and_workspace(op: Any) -> None:
    """t=1024 is the prefill case; the comb16 allocation must cover kernel writes."""
    device = _npu_device()
    args = _valid_args(device, t=1024)
    y, rstd, post, comb = op.mhcpresinkhorn_rmsnorm(**args)

    assert len(op._fake_kernel.calls) == 1
    grid, call_args = op._fake_kernel.calls[0]
    assert grid == (None, op.NB)
    assert len(call_args) == 13

    ng = (1024 + op.TG - 1) // op.TG
    rows = ng * op.TG + 64
    tpad = op._comb16_rows(1024)
    assert tpad == 1072

    assert tuple(call_args[0].shape) == (1024, op.HC_D)
    assert tuple(call_args[1].shape) == (1024, op.HC, op.D)
    assert tuple(call_args[2].shape) == (op.MIX_HC, op.HC_D)
    assert tuple(call_args[3].shape) == (1, 3)
    assert tuple(call_args[4].shape) == (1, op.MIX_HC)
    assert tuple(call_args[5].shape) == (1, op.D)
    assert tuple(call_args[6].shape) == (1024, op.D)
    assert tuple(call_args[7].shape) == (1024, 1)
    assert tuple(call_args[8].shape) == (1024, op.HC)
    assert tuple(call_args[9].shape) == (tpad, op.HC * op.HC)
    assert tuple(call_args[10].shape) == (op.NB, rows, op.PWC)
    assert tuple(call_args[11].shape) == (op.NB, rows * op.PWC)
    assert tuple(call_args[12].shape) == (1024, 8)

    assert (tuple(y.shape), y.dtype) == ((1024, op.D), torch.bfloat16)
    assert (tuple(rstd.shape), rstd.dtype) == ((1024, 1), torch.float32)
    assert (tuple(post.shape), post.dtype) == ((1024, op.HC), torch.float32)
    assert (tuple(comb.shape), comb.dtype) == ((1024, op.HC, op.HC),
                                               torch.float32)
    assert (1024, 0, 0) in op._WS_SCRATCH_CACHE
    assert op._WS_CACHE == {}


def test_wrapper_launch_for_decode_shape(op: Any) -> None:
    device = _npu_device()
    args = _valid_args(device, t=20)
    y, rstd, post, comb = op.mhcpresinkhorn_rmsnorm(**args)

    assert len(op._fake_kernel.calls) == 1
    grid, call_args = op._fake_kernel.calls[0]
    assert grid == (None, op.NB)
    ng = (20 + op.TG - 1) // op.TG
    rows = ng * op.TG + 64
    tpad = op._comb16_rows(20)
    assert tuple(call_args[9].shape) == (tpad, op.HC * op.HC)
    assert tuple(call_args[10].shape) == (op.NB, rows, op.PWC)
    assert tuple(call_args[12].shape) == (20, 8)
    assert (tuple(y.shape), y.dtype) == ((20, op.D), torch.bfloat16)
    assert (tuple(comb.shape), comb.dtype) == ((20, op.HC, op.HC),
                                               torch.float32)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hc_mult": 2}, "hc_mult"),
        ({"sinkhorn_iters": 10}, "sinkhorn_iters"),
        ({"eps": 1e-5}, "eps"),
        ({"rms_eps": 1e-5}, "rms_eps"),
    ],
)
def test_rejects_unsupported_scalar_options(op: Any, kwargs: dict[str, Any],
                                            message: str) -> None:
    device = _npu_device()
    with pytest.raises(ValueError, match=message):
        op.mhcpresinkhorn_rmsnorm(**_valid_args(device), **kwargs)
    assert op._fake_npu.allocations == []
    assert op._fake_kernel.calls == []


@pytest.mark.parametrize(
    "field, shape, dtype, message",
    [
        ("x", (4, 4, 4096), torch.float32, "x must be"),
        ("x", (4, 2, 4096), torch.bfloat16, "x must be"),
        ("x", (4, 4, 2048), torch.bfloat16, "x must be"),
        ("hc_fn", (24, 16384), torch.float32, "hc_fn must be"),
        ("hc_fn", (23, 16384), torch.bfloat16, "hc_fn must be"),
        ("hc_scale", (3,), torch.bfloat16, "hc_scale must be"),
        ("hc_scale", (4,), torch.float32, "hc_scale must be"),
        ("hc_base", (24,), torch.bfloat16, "hc_base must be"),
        ("hc_base", (23,), torch.float32, "hc_base must be"),
        ("gamma", (4096,), torch.float32, "gamma must be"),
        ("gamma", (2048,), torch.bfloat16, "gamma must be"),
    ],
)
def test_rejects_bad_tensor_contract(op: Any, field: str,
                                     shape: tuple[int, ...],
                                     dtype: torch.dtype, message: str) -> None:
    device = _npu_device()
    args = _valid_args(device)
    args[field] = _tensor(shape, dtype, device)
    with pytest.raises(ValueError, match=message):
        op.mhcpresinkhorn_rmsnorm(**args)
    assert op._fake_kernel.calls == []


def test_rejects_bad_x_rank(op: Any) -> None:
    device = _npu_device()
    args = _valid_args(device)
    args["x"] = _tensor((4, 16384), torch.bfloat16, device)
    with pytest.raises(ValueError, match="x must be"):
        op.mhcpresinkhorn_rmsnorm(**args)
    assert op._fake_kernel.calls == []


def test_rejects_non_contiguous_checked_tensors(op: Any) -> None:
    device = _npu_device()
    for field in ("x", "hc_fn", "hc_scale", "hc_base", "gamma"):
        args = _valid_args(device)
        args[field] = _tensor(args[field].shape, args[field].dtype, device,
                              contiguous=False)
        with pytest.raises(ValueError, match=f"{field} must be contiguous"):
            op.mhcpresinkhorn_rmsnorm(**args)
    assert op._fake_kernel.calls == []


def test_rejects_cpu_input(op: Any) -> None:
    with pytest.raises(ValueError, match="must be on an npu device"):
        op.mhcpresinkhorn_rmsnorm(**_valid_args(torch.device("cpu")))
    assert op._fake_kernel.calls == []


def test_moves_cross_device_inputs_to_tile_fwk_device(op: Any) -> None:
    args = _valid_args(_npu_device(1))
    y, rstd, post, comb = op.mhcpresinkhorn_rmsnorm(**args)

    target = _npu_device(0)
    for tensor in (y, rstd, post, comb):
        assert tensor.device == target
    grid, call_args = op._fake_kernel.calls[0]
    for tensor in call_args[:6]:
        assert tensor.device == target


def test_rejects_empty_token_dimension(op: Any) -> None:
    with pytest.raises(ValueError, match="t >= 1"):
        op.mhcpresinkhorn_rmsnorm(**_valid_args(_npu_device(), t=0))
    assert op._fake_kernel.calls == []


def test_rejects_unknown_keyword(op: Any) -> None:
    args = _valid_args(_npu_device())
    with pytest.raises(TypeError, match="gammar"):
        op.mhcpresinkhorn_rmsnorm(**args, gammar=args["gamma"])
    assert op._fake_kernel.calls == []
