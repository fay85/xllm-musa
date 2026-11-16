import torch
import triton
import triton.language as tl

@triton.jit
def _reduce_block_fp8_split_k(
    Parts, Out, ELEMENTS: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLITS)
    values = tl.load(
        Parts + splits[:, None] * ELEMENTS + offsets[None, :],
        offsets[None, :] < ELEMENTS,
        0.0,
    )
    tl.store(Out + offsets, tl.sum(values, axis=0), offsets < ELEMENTS)


@triton.jit
def _quant_gemm_kernel(inputs, weight, weight_scale, partials,
                      rows: tl.constexpr, columns: tl.constexpr, hidden: tl.constexpr,
                      split_count: tl.constexpr):
    block_column = tl.program_id(0)
    split = tl.program_id(1)
    input_rows = tl.arange(0, 32) % rows
    output_columns = block_column * 32 + tl.arange(0, 32)
    contraction = tl.arange(0, 32)
    accumulator = tl.zeros((32, 32), tl.float32)
    tiles_per_split = hidden // 32 // split_count
    for local_tile in range(tiles_per_split):
        tile = split * tiles_per_split + local_tile
        offsets = tile * 32 + contraction
        values = tl.load(inputs + input_rows[:, None] * hidden + offsets[None, :]).to(tl.float32)
        maximum = tl.maximum(tl.max(tl.abs(values), 1), 1e-4)
        raw_scale = maximum * (1.0 / 448.0)
        scale_bits = raw_scale.to(tl.int32, bitcast=True)
        scale = ((scale_bits + 0x7FFFFF) & -0x800000).to(tl.float32, bitcast=True)
        quantized = tl.clamp(values * (1.0 / scale)[:, None], -448.0, 448.0).to(weight.dtype.element_ty)
        weights = tl.load(weight + output_columns[None, :] * hidden + offsets[:, None])
        scales = tl.load(weight_scale + (output_columns // 32) * (hidden // 32) + tile)
        accumulator += tl.dot(quantized, weights) * scale[:, None] * scales[None, :]
    store_rows = tl.arange(0, 32)
    tl.store(partials + split * rows * columns + store_rows[:, None] * columns + output_columns[None, :], accumulator, store_rows[:, None] < rows)


def fused_decode_quant_gemm(inputs, weight, scales, *, split_count=None):
    splits = split_count if split_count is not None else (16 if inputs.shape[1] == 5120 else 4)
    rows, hidden = inputs.shape
    columns = weight.shape[0]
    assert hidden % (32 * splits) == 0 and columns % 32 == 0
    partials = torch.empty((splits, rows, columns), device=inputs.device, dtype=torch.float32)
    output = torch.empty((rows, columns), device=inputs.device, dtype=torch.bfloat16)
    _quant_gemm_kernel[(columns // 32, splits)](inputs, weight, scales, partials, rows, columns, hidden, splits, num_warps=4, num_stages=1)
    _reduce_block_fp8_split_k[(triton.cdiv(rows * columns, 256),)](partials, output, rows * columns, splits, 256)
    return output

