#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Export batch-tuned Mate GDN decode TVM-FFI modules for xLLM."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch


_BATCH_BUCKETS = (("", 2), ("b4", 4), ("b16", 16), ("blarge", 17))


def _build_kernel(
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    batch_size: int,
):
    from mate.gdn_kernels.tilelang.gdn_decode import (
        _get_decode_fp32_vk_kernel,
        _resolve_autotuned_kernel_config,
    )

    config = _resolve_autotuned_kernel_config(batch_size)
    dtype_name = str(dtype).split(".")[-1]
    return _get_decode_fp32_vk_kernel(
        qk_head=num_q_heads,
        head=num_v_heads,
        dim_k=head_dim,
        dim_v=head_dim,
        input_dtype=dtype_name,
        gate_batch_dtype=dtype_name,
        dt_bias_dtype="float32",
        output_dtype=dtype_name,
        use_qk_l2norm=True,
        v_tile=config["v_tile"],
        num_blocks_per_state=config["num_blocks_per_state"],
        stage=config["stage"],
        use_identity_state_indices=False,
    )


def _reference_step(
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
    query_f32 = torch.nn.functional.normalize(query.float(), dim=-1) * scale
    key_f32 = torch.nn.functional.normalize(key.float(), dim=-1)
    value_f32 = value.float()
    alpha = torch.exp(
        -torch.exp(a_log.float()).unsqueeze(0)
        * torch.nn.functional.softplus(a.float() + dt_bias.float().unsqueeze(0))
    )
    beta = torch.sigmoid(beta_logits.float())
    reference_state = state.clone()
    output = torch.empty_like(value_f32)
    heads_per_query = value.size(1) // query.size(1)
    for batch_idx in range(query.size(0)):
        state_slot = int(state_indices[batch_idx].item())
        for value_head in range(value.size(1)):
            query_head = value_head // heads_per_query
            current = reference_state[state_slot, value_head]
            current = current * alpha[batch_idx, value_head]
            prediction = torch.mv(current, key_f32[batch_idx, query_head])
            delta = (
                value_f32[batch_idx, value_head] - prediction
            ) * beta[batch_idx, value_head]
            current = current + torch.outer(
                delta, key_f32[batch_idx, query_head]
            )
            reference_state[state_slot, value_head] = current
            output[batch_idx, value_head] = torch.mv(
                current, query_f32[batch_idx, query_head]
            )
    return output, reference_state


def _compile_and_validate(
    kernel,
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    device = torch.device("musa")
    pool_size = batch_size + 1
    query = torch.randn(
        (batch_size, num_q_heads, head_dim), device=device, dtype=dtype
    )
    key = torch.randn_like(query)
    value = torch.randn(
        (batch_size, num_v_heads, head_dim), device=device, dtype=dtype
    )
    a_log = torch.randn((num_v_heads,), device=device, dtype=torch.float32)
    a = torch.randn((batch_size, num_v_heads), device=device, dtype=dtype)
    dt_bias = torch.randn(
        (num_v_heads,), device=device, dtype=torch.float32
    )
    beta_logits = torch.randn_like(a)
    state_indices = torch.arange(
        1, batch_size + 1, device=device, dtype=torch.int32
    )
    state = torch.randn(
        (pool_size, num_v_heads, head_dim, head_dim),
        device=device,
        dtype=torch.float32,
    )
    scale = head_dim**-0.5
    expected_output, expected_state = _reference_step(
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
    output = torch.empty_like(value)
    kernel(
        query,
        key,
        value,
        a_log,
        a,
        dt_bias,
        beta_logits,
        scale,
        state_indices,
        state,
        output,
    )
    torch.musa.synchronize()
    torch.testing.assert_close(
        output.float(), expected_output, rtol=0.05, atol=0.05
    )
    selected_state = state.index_select(0, state_indices.long())
    expected_selected_state = expected_state.index_select(
        0, state_indices.long()
    )
    torch.testing.assert_close(
        selected_state, expected_selected_state, rtol=0.01, atol=0.01
    )


def _export_bucket(
    num_q_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    suffix: str,
    batch_size: int,
    ops_root: Path,
) -> Path:
    dtype_suffix = {torch.bfloat16: "bf16", torch.float16: "f16"}[dtype]
    uri = (
        f"mate_gdn_decode_tvmffi_hq{num_q_heads}_hv{num_v_heads}_"
        f"{dtype_suffix}{'_' + suffix if suffix else ''}"
    )
    destination = ops_root / uri / f"{uri}.so"
    destination.parent.mkdir(parents=True, exist_ok=True)

    kernel = _build_kernel(
        num_q_heads, num_v_heads, head_dim, dtype, batch_size
    )
    _compile_and_validate(
        kernel, num_q_heads, num_v_heads, head_dim, dtype, batch_size
    )
    libpath = vars(kernel.adapter).get("libpath")
    if not libpath:
        # A freshly compiled TileLang adapter may expose libpath only after it
        # is reconstructed from the disk cache.
        from mate.gdn_kernels.tilelang.gdn_decode import (
            _get_decode_fp32_vk_kernel,
        )

        _get_decode_fp32_vk_kernel.cache_clear()
        kernel = _build_kernel(
            num_q_heads, num_v_heads, head_dim, dtype, batch_size
        )
        libpath = vars(kernel.adapter).get("libpath")
    if libpath and os.path.exists(libpath):
        shutil.copy2(libpath, destination)
        source = libpath
    else:
        executable = getattr(kernel.adapter, "executable", None)
        if executable is None:
            raise RuntimeError(
                "TileLang adapter exposes neither libpath nor executable for "
                f"batch {batch_size}"
            )
        executable.export_library(str(destination))
        source = "in-memory TileLang executable"
    print(f"exported batch<={batch_size}: {destination} <- {source}")
    return destination


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
    ops_root = Path(args.ops_path)
    for suffix, batch_size in _BATCH_BUCKETS:
        _export_bucket(
            args.hq,
            args.hv,
            args.head_dim,
            dtype,
            suffix,
            batch_size,
            ops_root,
        )


if __name__ == "__main__":
    main()
