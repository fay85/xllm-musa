#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export and validate stride-aware Mate GDN prefill TVM-FFI modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import tvm_ffi


def _dtype_suffix(dtype: torch.dtype) -> str:
  return {
      torch.bfloat16: "bf16",
      torch.float16: "f16",
  }[dtype]


def _build_prefill_kernel(
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    is_varlen: bool,
) -> Any:
  from mate.gdn_kernels.tilelang.gdn_prefill import (
      tilelang_fused_chunk_gdn_prefill,
  )

  return tilelang_fused_chunk_gdn_prefill(
      H=num_v_heads,
      Hg=num_q_heads,
      DK=head_dim,
      DV=head_dim,
      chunk_size=64,
      scale=head_dim**-0.5,
      accum_dtype="float32",
      qkva_dtype=dtype,
      g_dtype=torch.float32,
      b_dtype=torch.float32,
      h0_dtype=torch.float32,
      ht_dtype=torch.float32,
      o_dtype=dtype,
      seqlen_dtype=torch.int32,
      use_initial_state=True,
      store_final_state=True,
      is_varlen=is_varlen,
      is_log_space=True,
  )


def _build_kkt_kernel(
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    is_varlen: bool,
) -> Any:
  from mate.gdn_kernels.tilelang.gdn_kkt_solve import tilelang_kkt_solve

  return tilelang_kkt_solve(
      H=num_v_heads,
      Hg=num_q_heads,
      DK=head_dim,
      chunk_size=64,
      accum_dtype="float32",
      qkva_dtype=dtype,
      b_dtype=torch.float32,
      seqlen_dtype=torch.int32,
      is_varlen=is_varlen,
  )


def _make_strided_qkv(
    batch_size: int,
    num_tokens: int,
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  fused_width = (2 * num_q_heads + num_v_heads) * head_dim
  fused = torch.randn(
      (batch_size, num_tokens, fused_width),
      device=torch.device("musa"),
      dtype=dtype,
  )
  query = fused.narrow(
      -1, 0, num_q_heads * head_dim
  ).view(batch_size, num_tokens, num_q_heads, head_dim)
  key = fused.narrow(
      -1, num_q_heads * head_dim, num_q_heads * head_dim
  ).view(batch_size, num_tokens, num_q_heads, head_dim)
  value = fused.narrow(
      -1, 2 * num_q_heads * head_dim, num_v_heads * head_dim
  ).view(batch_size, num_tokens, num_v_heads, head_dim)
  if query.is_contiguous() or key.is_contiguous() or value.is_contiguous():
    raise RuntimeError("test Q/K/V tensors must exercise leading strides")
  return query, key, value


def _validate_prefill_kernel(
    kernel: Any,
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
  torch.manual_seed(20260801)
  batch_size = 1
  num_tokens = 64
  query, key, value = _make_strided_qkv(
      batch_size,
      num_tokens,
      num_q_heads,
      num_v_heads,
      head_dim,
      dtype,
  )
  a = torch.randn(
      (batch_size, num_tokens, num_v_heads, 64),
      device=query.device,
      dtype=dtype,
  )
  g = -0.01 * torch.rand(
      (batch_size, num_tokens, num_v_heads),
      device=query.device,
      dtype=torch.float32,
  )
  beta = torch.rand_like(g)
  initial_state = torch.zeros(
      (batch_size, num_v_heads, head_dim, head_dim),
      device=query.device,
      dtype=torch.float32,
  )
  cu_seqlens = torch.tensor(
      [0, num_tokens], device=query.device, dtype=torch.int32
  )
  strided_output = torch.empty_like(value.contiguous())
  contiguous_output = torch.empty_like(strided_output)
  strided_state = torch.empty_like(initial_state)
  contiguous_state = torch.empty_like(initial_state)

  kernel(
      query,
      key,
      value,
      a,
      g,
      beta,
      initial_state,
      cu_seqlens,
      strided_output,
      strided_state,
  )
  kernel(
      query.contiguous(),
      key.contiguous(),
      value.contiguous(),
      a,
      g,
      beta,
      initial_state,
      cu_seqlens,
      contiguous_output,
      contiguous_state,
  )
  torch.musa.synchronize()
  torch.testing.assert_close(
      strided_output, contiguous_output, rtol=0.0, atol=0.0
  )
  torch.testing.assert_close(
      strided_state, contiguous_state, rtol=0.0, atol=0.0
  )


def _validate_kkt_kernel(
    kernel: Any,
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    is_varlen: bool,
) -> None:
  torch.manual_seed(20260802)
  batch_size = 1
  num_tokens = 64
  _query, key, _value = _make_strided_qkv(
      batch_size,
      num_tokens,
      num_q_heads,
      num_v_heads,
      head_dim,
      dtype,
  )
  beta = torch.rand(
      (batch_size, num_tokens, num_v_heads),
      device=key.device,
      dtype=torch.float32,
  )
  cu_seqlens = torch.tensor(
      [0, num_tokens], device=key.device, dtype=torch.int32
  )
  strided_output = torch.empty(
      (batch_size, num_tokens, num_v_heads, 64),
      device=key.device,
      dtype=dtype,
  )
  contiguous_output = torch.empty_like(strided_output)

  if is_varlen:
    kernel(key, beta, cu_seqlens, strided_output)
    kernel(key.contiguous(), beta, cu_seqlens, contiguous_output)
  else:
    num_chunks = batch_size * (num_tokens // 64)
    kernel(key, beta, strided_output, num_chunks)
    kernel(key.contiguous(), beta, contiguous_output, num_chunks)
  torch.musa.synchronize()
  torch.testing.assert_close(
      strided_output, contiguous_output, rtol=0.0, atol=0.0
  )


def _export_kernel(kernel: Any, destination: Path) -> str:
  destination.parent.mkdir(parents=True, exist_ok=True)
  libpath = vars(kernel.adapter).get("libpath")
  if libpath and os.path.exists(libpath):
    shutil.copy2(libpath, destination)
  else:
    executable = getattr(kernel.adapter, "executable", None)
    if executable is None:
      raise RuntimeError(
          "TileLang adapter exposes neither libpath nor executable"
      )
    executable.export_library(str(destination))

  module = tvm_ffi.load_module(str(destination))
  module.get_function("main")
  return hashlib.sha256(destination.read_bytes()).hexdigest()


def _module_uri(
    kind: str,
    num_q_heads: int,
    num_v_heads: int,
    dtype: torch.dtype,
    is_varlen: bool,
) -> str:
  varlen = "varlen_" if is_varlen else ""
  return (
      f"mate_{kind}_{varlen}strided_hq{num_q_heads}_"
      f"hv{num_v_heads}_{_dtype_suffix(dtype)}"
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--ops-path",
      default=os.environ.get(
          "FLASHINFER_OPS_PATH", "/workspace/mate_cached_ops"
      ),
  )
  parser.add_argument("--hq", type=int, default=16)
  parser.add_argument("--hv", type=int, default=48)
  parser.add_argument("--head-dim", type=int, default=128)
  parser.add_argument("--dtype", choices=("bf16", "f16"), default="bf16")
  args = parser.parse_args()

  dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
  ops_root = Path(args.ops_path)
  manifest: dict[str, object] = {
      "mate_version": __import__("mate").__version__,
      "torch_version": torch.__version__,
      "num_q_heads": args.hq,
      "num_v_heads": args.hv,
      "head_dim": args.head_dim,
      "dtype": args.dtype,
      "modules": {},
  }

  for is_varlen in (False, True):
    prefill_kernel = _build_prefill_kernel(
        args.hq, args.hv, args.head_dim, dtype, is_varlen
    )
    _validate_prefill_kernel(
        prefill_kernel, args.hq, args.hv, args.head_dim, dtype
    )
    prefill_uri = _module_uri(
        "gdn_prefill_full", args.hq, args.hv, dtype, is_varlen
    )
    prefill_path = ops_root / prefill_uri / f"{prefill_uri}.so"
    manifest["modules"][prefill_uri] = {
        "path": str(prefill_path),
        "sha256": _export_kernel(prefill_kernel, prefill_path),
    }

    kkt_kernel = _build_kkt_kernel(
        args.hq, args.hv, args.head_dim, dtype, is_varlen
    )
    _validate_kkt_kernel(
        kkt_kernel,
        args.hq,
        args.hv,
        args.head_dim,
        dtype,
        is_varlen,
    )
    kkt_uri = _module_uri(
        "kkt_solve", args.hq, args.hv, dtype, is_varlen
    )
    kkt_path = ops_root / kkt_uri / f"{kkt_uri}.so"
    manifest["modules"][kkt_uri] = {
        "path": str(kkt_path),
        "sha256": _export_kernel(kkt_kernel, kkt_path),
    }

  manifest_path = (
      ops_root
      / (
          f"mate_gdn_stride_aware_hq{args.hq}_hv{args.hv}_"
          f"{_dtype_suffix(dtype)}.json"
      )
  )
  manifest_path.write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(f"Exported and validated stride-aware modules: {manifest_path}")


if __name__ == "__main__":
  main()
