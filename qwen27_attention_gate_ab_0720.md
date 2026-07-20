# Qwen3.5-27B attention gate A/B (2026-07-20)

## Scope

This experiment evaluates the Qwen3.5-27B attention output gate on the MUSA
decode path.  The candidate replaces the decomposed in-place sequence
`gate.sigmoid_(); out.mul_(gate)` with the capture-safe
`mul_sigmoid_gate_inplace(out, gate)` kernel.  The fused form avoids the
temporary sigmoid result and does not mutate `gate`.

The test was run with the BF16 Qwen3.5-27B weights, FA3 enabled, CUDA graph
enabled, piece-wise graph enabled, prefix cache disabled, GPU 7, C=1
(`max_concurrent_requests=1`), ISL=20k, OSL=300, four warmup requests and ten
measured requests.  Both cases used the same MUSA container and server/client
protocol.  The benchmark client was fixed to count Qwen3.5
`reasoning_content` as generated output; that harness-only change is not part
of the source commit.

## Results

| Variant | Mean TTFT (ms) | Mean TPOT (ms) | Successful measured requests |
| --- | ---: | ---: | ---: |
| Generic sigmoid + multiply | 3493.55 | 51.00 | 10/10 |
| Fused `mul_sigmoid_gate_inplace` | 3494.30 | 50.01 | 10/10 |

Relative to the generic path, fused gating changes TTFT by +0.75 ms (+0.02%),
which is noise-level, and reduces TPOT by 0.99 ms (-1.94%).  The candidate is
therefore accepted as a decode TPOT optimization, but it should not be claimed
as a TTFT improvement.

## Validation notes

- The source was rebuilt inside `xllm-musa2.9.1-sdk5.1-dev` with
  `MAX_JOBS=16 NINJA_TARGET=xllm ./_build_cuda_graph_musa.sh`.
- A graph-mode smoke request completed successfully after the rebuild.
- The full A/B runs completed with 10/10 successful measured requests.
- The experimental GQA6 dynamic FA3 split (`num_splits=0`) was not committed.
- A null-stream MUSA stream-semantics optimization caused illegal graph-path
  accesses and was restored to the working pool-stream behavior; it is not
  included in this commit.
- Fused GDN was tested separately and did not produce a net improvement, so it
  is not included here.
