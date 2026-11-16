"""Candidate only: process the valid expert prefixes from DeepEP LL."""
import logging
import torch
import triton
import triton.language as tl
from sglang.srt.layers.moe.musa_w4a8_moe import (
    _silu_and_mul_quant_fp8_elementwise_for_musa,
)

@triton.jit
def quant_kernel(X,Y,Counts,Scale,K:tl.constexpr,CAP:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0)
    col=tl.program_id(1)*B+tl.arange(0,B)
    count=tl.load(Counts+e)
    scale=tl.load(Scale).to(tl.float32)
    for row in range(count):
        offset=(e*CAP+row)*K+col
        x=tl.load(X+offset,col<K,other=0).to(tl.float32)
        q=tl.minimum(tl.maximum(tl.div_rn(x,scale),-448.),448.)
        tl.store(Y+offset,q.to(Y.dtype.element_ty),col<K)

@triton.jit
def silu_kernel(X,Y,Counts,Scale,K:tl.constexpr,CAP:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0)
    col=tl.program_id(1)*B+tl.arange(0,B)
    count=tl.load(Counts+e)
    inv_scale=1./tl.load(Scale).to(tl.float32)
    for row in range(count):
        _silu_and_mul_quant_fp8_elementwise_for_musa(
            X,Y,(e*CAP+row)*K*2+col,(e*CAP+row)*K+col,
            K,col,col<K,inv_scale,448.)

def quant(x,scale,counts):
    if not getattr(quant, '_logged', False):
        logging.getLogger(__name__).info(
            'MUSA W4A8 DeepEP LL masked preparation active: shape=%s static scales retained',
            tuple(x.shape),
        )
        quant._logged = True
    e,cap,k=x.shape
    y=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    quant_kernel[(e,triton.cdiv(k,512))](x,y,counts,scale,k,cap,512,num_warps=4)
    return y

def silu(x,scale,counts):
    e,cap,kn=x.shape;k=kn//2
    y=torch.empty((e,cap,k),device=x.device,dtype=torch.float8_e4m3fn)
    silu_kernel[(e,triton.cdiv(k,512))](x,y,counts,scale,k,cap,512,num_warps=4)
    return y
