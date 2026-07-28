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
"""Export and validate a Mate GDN MTP TVM-FFI module for xLLM."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import tvm_ffi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.logger import logger


def _build_kernel(
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> Any:
    from mate.gdn_kernels.tilelang.gdn_mtp import (
        _get_mtp_config,
        _get_mtp_fp32_vk_smem_kernel,
    )

    batch_size = 1
    seq_len = 2
    tile_v, _vec_size, ilp_rows = _get_mtp_config(
        batch_size=batch_size,
        seq_len=seq_len,
        num_v_heads=num_v_heads,
        v_dim=head_dim,
        cache_intermediate_states=True,
    )
    dtype_name = str(dtype).split(".")[-1]
    return _get_mtp_fp32_vk_smem_kernel(
        seq_len=seq_len,
        qk_head=num_q_heads,
        head=num_v_heads,
        dim_k=head_dim,
        dim_v=head_dim,
        input_dtype=dtype_name,
        output_dtype=dtype_name,
        dt_bias_dtype="float32",
        state_dtype="float32",
        use_qk_l2norm=True,
        cache_intermediate_states=True,
        disable_state_update=True,
        use_identity_state_indices=False,
        tile_v=tile_v,
        ilp_rows=ilp_rows,
    )


def _reference_mtp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta_logits: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_q_heads, _head_dim = query.shape
    num_v_heads = value.size(2)
    heads_per_query = num_v_heads // num_q_heads
    query_f32 = torch.nn.functional.normalize(query.float(), dim=-1) * scale
    key_f32 = torch.nn.functional.normalize(key.float(), dim=-1)
    output = torch.empty_like(value, dtype=torch.float32)
    intermediate = torch.empty(
        (
            batch_size,
            seq_len,
            num_v_heads,
            value.size(-1),
            query.size(-1),
        ),
        device=query.device,
        dtype=torch.float32,
    )

    for batch_idx in range(batch_size):
        state_slot = int(state_indices[batch_idx].item())
        current = state[state_slot].clone()
        for token_idx in range(seq_len):
            query_token = query_f32[batch_idx, token_idx].repeat_interleave(
                heads_per_query, dim=0
            )
            key_token = key_f32[batch_idx, token_idx].repeat_interleave(
                heads_per_query, dim=0
            )
            alpha = torch.exp(
                -torch.exp(a_log.float())
                * torch.nn.functional.softplus(
                    a[batch_idx, token_idx].float() + dt_bias.float()
                )
            )
            beta = torch.sigmoid(beta_logits[batch_idx, token_idx].float())
            current = current * alpha[:, None, None]
            prediction = torch.einsum("hvk,hk->hv", current, key_token)
            delta = (value[batch_idx, token_idx].float() - prediction) * beta[
                :, None
            ]
            current = current + torch.einsum("hv,hk->hvk", delta, key_token)
            intermediate[batch_idx, token_idx].copy_(current)
            output[batch_idx, token_idx] = torch.einsum(
                "hvk,hk->hv", current, query_token
            )

    return output, intermediate


def _compile_and_validate(
    kernel: Any,
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260727)
    device = torch.device("musa")
    batch_size = 1
    seq_len = 2
    pool_size = 2
    query = torch.randn(
        (batch_size, seq_len, num_q_heads, head_dim),
        device=device,
        dtype=dtype,
    )
    key = torch.randn_like(query)
    value = torch.randn(
        (batch_size, seq_len, num_v_heads, head_dim),
        device=device,
        dtype=dtype,
    )
    a_log = torch.randn((num_v_heads,), device=device, dtype=torch.float32)
    a = torch.randn(
        (batch_size, seq_len, num_v_heads), device=device, dtype=dtype
    )
    dt_bias = torch.randn(
        (num_v_heads,), device=device, dtype=torch.float32
    )
    beta_logits = torch.randn_like(a)
    state_indices = torch.tensor([1], device=device, dtype=torch.int32)
    state = torch.randn(
        (pool_size, num_v_heads, head_dim, head_dim),
        device=device,
        dtype=torch.float32,
    )
    state_before = state.clone()
    intermediate = torch.empty(
        (batch_size, seq_len, num_v_heads, head_dim, head_dim),
        device=device,
        dtype=torch.float32,
    )
    output = torch.empty_like(value)
    scale = head_dim**-0.5
    expected_output, expected_intermediate = _reference_mtp(
        query,
        key,
        value,
        a_log,
        a,
        dt_bias,
        beta_logits,
        state_indices,
        state,
        scale,
    )

    kernel(
        query,
        key,
        value,
        a_log,
        a,
        dt_bias,
        beta_logits,
        scale,
        state,
        state_indices,
        intermediate,
        output,
    )
    torch.musa.synchronize()
    torch.testing.assert_close(
        output.float(), expected_output, rtol=0.05, atol=0.05
    )
    torch.testing.assert_close(
        intermediate, expected_intermediate, rtol=0.01, atol=0.01
    )
    torch.testing.assert_close(state, state_before, rtol=0.0, atol=0.0)


def _export_module(
    kernel: Any,
    destination: Path,
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    libpath = vars(kernel.adapter).get("libpath")
    if libpath and os.path.exists(libpath):
        shutil.copy2(libpath, destination)
        return

    from mate.gdn_kernels.tilelang.gdn_mtp import (
        _get_mtp_fp32_vk_smem_kernel,
    )

    _get_mtp_fp32_vk_smem_kernel.cache_clear()
    kernel = _build_kernel(num_q_heads, num_v_heads, head_dim, dtype)
    libpath = vars(kernel.adapter).get("libpath")
    if libpath and os.path.exists(libpath):
        shutil.copy2(libpath, destination)
        return

    executable = getattr(kernel.adapter, "executable", None)
    if executable is None:
        raise RuntimeError(
            "TileLang MTP adapter exposes neither libpath nor executable"
        )
    executable.export_library(str(destination))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ops-path",
        default=os.environ.get(
            "FLASHINFER_OPS_PATH", "/workspace/mate_cached_ops"
        ),
    )
    parser.add_argument("--hq", type=int, default=16)
    parser.add_argument("--hv", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "f16"), default="bf16")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    dtype_suffix = {torch.bfloat16: "bf16", torch.float16: "f16"}[dtype]
    ops_root = Path(args.ops_path)
    uri = f"mate_gdn_mtp_hq{args.hq}_hv{args.hv}_{dtype_suffix}"
    destination_dir = ops_root / uri

    kernel = _build_kernel(args.hq, args.hv, args.head_dim, dtype)
    _compile_and_validate(kernel, args.hq, args.hv, args.head_dim, dtype)

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{uri}.so"
    _export_module(
        kernel,
        destination,
        args.hq,
        args.hv,
        args.head_dim,
        dtype,
    )
    stale_kernel_library = destination_dir / "kernel_lib.so"
    stale_kernel_library.unlink(missing_ok=True)

    module = tvm_ffi.load_module(str(destination))
    module.get_function("main")
    logger.info(f"Exported validated MTP TVM-FFI module to {destination}")


if __name__ == "__main__":
    main()
