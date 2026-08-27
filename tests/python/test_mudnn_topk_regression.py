#!/usr/bin/env python3
"""Direct TorchMUSA/muDNN TopK regression test.

The dimensions match Qwen3-0.6B software beam search:
  batch = beam_width = 128
  vocab = 151936
  k = 2 * beam_width = 256
"""

from __future__ import annotations

import os

import pytest
import torch


BATCH = int(os.getenv("MUDNN_TOPK_BATCH", "128"))
VOCAB = int(os.getenv("MUDNN_TOPK_VOCAB", "151936"))
K = int(os.getenv("MUDNN_TOPK_K", "256"))
ITERATIONS = int(os.getenv("MUDNN_TOPK_ITERATIONS", "1000"))
SEED = int(os.getenv("MUDNN_TOPK_SEED", "1234"))


def _assert_valid_topk(
    source_cpu: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    reference_indices: torch.Tensor,
    iteration: int,
) -> None:
    host_values = values.float().cpu()
    host_indices = indices.cpu()

    assert host_indices.dtype == torch.int64
    assert tuple(host_indices.shape) == (BATCH, K)

    minimum = int(host_indices.min())
    maximum = int(host_indices.max())
    assert minimum >= 0, (
        f"iteration={iteration}: negative TopK index: {minimum}"
    )
    assert maximum < VOCAB, (
        f"iteration={iteration}: TopK index {maximum} >= vocab {VOCAB}"
    )

    # An index tensor containing score bytes instead of int64 token indices
    # fails this relation even if a corrupted value happens to be in range.
    gathered = source_cpu.gather(-1, host_indices)
    torch.testing.assert_close(host_values, gathered, rtol=0, atol=0)

    # Compare sets so equal-score tie ordering is not treated as an error.
    actual_set = host_indices.sort(dim=-1).values
    expected_set = reference_indices.sort(dim=-1).values
    assert torch.equal(actual_set, expected_set), (
        f"iteration={iteration}: MUSA and CPU TopK candidate sets differ"
    )


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="MUSA device is unavailable",
)
def test_mudnn_topk_indices_are_int64_and_in_range() -> None:
    torch.musa.set_device(0)
    torch.manual_seed(SEED)

    # Float32 matches the log_softmax tensor passed to the beam top-logprobs
    # TopK call. Build one fixed input so failures are not hidden by changing
    # random data between iterations.
    source = torch.randn(
        (BATCH, VOCAB), device="musa", dtype=torch.float32
    )
    source_cpu = source.cpu()
    reference_indices = torch.topk(
        source_cpu, K, dim=-1, largest=True, sorted=True
    ).indices

    for iteration in range(ITERATIONS):
        values, indices = torch.topk(
            source, K, dim=-1, largest=True, sorted=True
        )
        torch.musa.synchronize()
        _assert_valid_topk(
            source_cpu, values, indices, reference_indices, iteration
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
