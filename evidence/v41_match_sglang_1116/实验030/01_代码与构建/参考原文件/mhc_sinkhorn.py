"""Fused hc=4 Sinkhorn matching the corrected MUSA Torch operation order."""

from functools import lru_cache

import torch

_SOURCE = r"""
__device__ __forceinline__ int dsv4_sinkhorn(
    const float* x, const float* scale, const float* base,
    float* pre, float* post, float* comb) {
  float c[16];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    pre[i] = 1.0f / (1.0f + __expf(-(x[i] * scale[0] + base[i]))) + 1e-6f;
    post[i] = 2.0f * (1.0f / (1.0f + __expf(-(x[i+4] * scale[1] + base[i+4]))));
  }
  #pragma unroll
  for (int i = 0; i < 16; ++i) c[i] = x[i+8] * scale[2] + base[i+8];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float maximum = fmaxf(fmaxf(c[i*4], c[i*4+1]), fmaxf(c[i*4+2], c[i*4+3]));
    #pragma unroll
    for (int j = 0; j < 4; ++j) c[i*4+j] = __expf(c[i*4+j] - maximum);
    float denominator = ((c[i*4] + c[i*4+1]) + c[i*4+2]) + c[i*4+3];
    #pragma unroll
    for (int j = 0; j < 4; ++j) c[i*4+j] = c[i*4+j] / denominator + 1e-6f;
  }
  for (int it = 0; it < 20; ++it) {
    if (it > 0) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float denominator = ((c[i*4] + c[i*4+1]) + c[i*4+2]) + c[i*4+3] + 1e-6f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) c[i*4+j] = c[i*4+j] / denominator;
      }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float denominator = ((c[j] + c[4+j]) + c[8+j]) + c[12+j] + 1e-6f;
      #pragma unroll
      for (int i = 0; i < 4; ++i) c[i*4+j] = c[i*4+j] / denominator;
    }
  }
  #pragma unroll
  for (int i = 0; i < 16; ++i) comb[i] = c[i];
  return 0;
}
"""


@lru_cache(maxsize=1)
def _kernel():
    import tilelang
    import tilelang.language as T

    @tilelang.jit(
        target="musa",
        out_idx=[3, 4, 5],
        compile_flags=["-ffp-contract=off", "-fno-fast-math"],
    )
    def build():
        n = T.symbolic("n")

        @T.prim_func
        def kernel(
            X: T.Tensor((n, 24), T.float32),
            S: T.Tensor((3,), T.float32),
            B: T.Tensor((24,), T.float32),
            P: T.Tensor((n, 4), T.float32),
            Q: T.Tensor((n, 4), T.float32),
            C: T.Tensor((n, 4, 4), T.float32),
        ):
            with T.Kernel(T.ceildiv(n, 64), threads=64) as bx:
                T.import_source(_SOURCE)
                i = bx * 64 + T.get_thread_binding()
                if i < n:
                    T.evaluate(
                        T.call_extern(
                            "int32",
                            "dsv4_sinkhorn",
                            T.address_of(X[i, 0]),
                            T.address_of(S[0]),
                            T.address_of(B[0]),
                            T.address_of(P[i, 0]),
                            T.address_of(Q[i, 0]),
                            T.address_of(C[i, 0, 0]),
                        )
                    )

        return kernel

    return build()


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Return contiguous FP32 pre/post/comb for the V4/V4.1 MUSA contract.

    __expf, ordered additions and separate FP32 multiply/add rounding are
    intentional. Changing any of them requires exact model-tensor replay.
    """
    if hc_mult != 4 or sinkhorn_iters != 20 or eps != 1e-6:
        raise ValueError("MUSA Sinkhorn requires hc=4, 20 iterations and eps=1e-6")
    if (
        mixes.ndim != 3
        or mixes.shape[-1] != 24
        or hc_scale.shape != (3,)
        or hc_base.shape != (24,)
    ):
        raise ValueError("MUSA Sinkhorn expects [B,S,24], [3] and [24] tensors")
    if mixes.device.type != "musa" or any(
        t.dtype != torch.float32 or t.device != mixes.device or not t.is_contiguous()
        for t in (mixes, hc_scale, hc_base)
    ):
        raise ValueError("MUSA Sinkhorn expects contiguous FP32 inputs on one device")
    b, s, _ = mixes.shape
    if mixes.numel() == 0:
        return (
            mixes.new_empty((b, s, 4)),
            mixes.new_empty((b, s, 4)),
            mixes.new_empty((b, s, 4, 4)),
        )
    pre, post, comb = _kernel()(mixes.view(-1, 24), hc_scale, hc_base)
    return pre.view(b, s, 4), post.view(b, s, 4), comb.view(b, s, 4, 4)
