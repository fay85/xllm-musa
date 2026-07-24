#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Export Mate TileLang KKT solve as a TVM-FFI module for xLLM."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch


def export_one(num_q_heads: int, num_v_heads: int, dtype: torch.dtype, out_root: Path) -> Path:
  from mate.gdn_kernels.tilelang.gdn_kkt_solve import tilelang_kkt_solve

  dtype_suffix = {torch.bfloat16: "bf16", torch.float16: "f16"}[dtype]
  uri = f"mate_kkt_solve_hq{num_q_heads}_hv{num_v_heads}_{dtype_suffix}"
  out_dir = out_root / uri
  out_dir.mkdir(parents=True, exist_ok=True)
  out_so = out_dir / f"{uri}.so"

  kn = tilelang_kkt_solve(
      num_v_heads,
      num_q_heads,
      128,
      64,
      qkva_dtype=dtype,
      b_dtype=torch.float32,
      seqlen_dtype="int32",
      accum_dtype="float32",
      is_varlen=False,
  )
  src = kn.adapter.libpath
  if not src or not os.path.exists(src):
    raise RuntimeError(f"TileLang KKT adapter libpath missing for {uri}: {src}")
  shutil.copy2(src, out_so)
  print(f"exported {out_so} <- {src}")
  return out_so


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--ops-path",
      default=os.environ.get("FLASHINFER_OPS_PATH", "/workspace/mate_cached_ops"),
  )
  parser.add_argument("--hq", type=int, default=16)
  parser.add_argument("--hv", type=int, default=48)
  parser.add_argument("--dtype", choices=["bf16", "f16"], default="bf16")
  args = parser.parse_args()

  dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
  device = torch.device("musa")
  k = torch.randn(1, 64, args.hq, 128, device=device, dtype=dtype)
  b = torch.rand(1, 64, args.hv, device=device, dtype=torch.float32)
  from mate.gdn_kernels.tilelang.gdn_kkt_solve import kkt_solve

  _ = kkt_solve(k=k, b=b)
  export_one(args.hq, args.hv, dtype, Path(args.ops_path))


if __name__ == "__main__":
  main()
