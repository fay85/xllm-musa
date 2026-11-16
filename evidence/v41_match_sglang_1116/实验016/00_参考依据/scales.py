"""Scale layout conversions and validation for MUSA W4A8."""

from __future__ import annotations

import torch


def interleave_cutlass_scales(scales: torch.Tensor) -> torch.Tensor:
    """Interleave scales in groups of four for the CUTLASS runtime layout."""
    shape = scales.shape
    alignment = 4 if shape[2] % 4 == 0 else 1
    return (
        scales.reshape(shape[0], shape[1], shape[2] // alignment, alignment)
        .permute(0, 2, 1, 3)
        .reshape(shape[0], shape[2] // alignment, shape[1] * alignment)
        .contiguous()
    )


def convert_cutlass_scale_to_mate(scales: torch.Tensor, n_logical: int) -> torch.Tensor:
    """Convert checkpoint/CUTLASS scales to ``[E, N, K // 128]``."""
    if scales.ndim != 3:
        raise ValueError("W4A8 checkpoint scales must be rank 3")
    _, dim1, dim2 = scales.shape
    if dim1 == n_logical:
        return scales
    if dim2 % n_logical:
        raise ValueError(
            f"cannot convert scale shape {tuple(scales.shape)} for N={n_logical}"
        )
    alignment = dim2 // n_logical
    groups = dim1
    if alignment == 1:
        return scales.transpose(1, 2).contiguous()
    return (
        scales.reshape(scales.shape[0], groups, n_logical, alignment)
        .permute(0, 2, 1, 3)
        .reshape(scales.shape[0], n_logical, groups * alignment)
        .contiguous()
    )


def validate_mate_scale(scales: torch.Tensor, n_logical: int, k_logical: int) -> None:
    if k_logical % 128:
        raise ValueError("MUSA W4A8 requires K to be divisible by 128")
    expected = (scales.shape[0], n_logical, k_logical // 128)
    if scales.ndim != 3 or tuple(scales.shape) != expected:
        raise ValueError(
            f"MUSA W4A8 scale must use MATE layout [E, N, K//128], got {tuple(scales.shape)}"
        )
