# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Validate original FP4/block32 primitives; no full-model speed claim."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
import torch.nn.functional as F

from scripts.logger import logger


def _quant_oracle(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Independent CPU implementation: FP32 amax/scaling as official TileLang,
    # FP64 log2 only to calculate the exact enclosing power of two.
    value = x.cpu().float().reshape(x.shape[0], x.shape[1] // 32, 32)
    raw = value.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    scales = torch.pow(2.0, raw.double().log2().ceil()).float()
    quant = (value / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    qdq = (quant.float() * scales.unsqueeze(-1)).to(torch.bfloat16).reshape(x.shape)
    return qdq, scales


def _weight_oracle(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    table = torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float64)
    raw = packed.cpu()
    v = torch.stack((table[(raw & 15).long()], table[(raw >> 4).long()]), -1).flatten(-2)
    # Byte 0 is 2^-127. FF is deliberately invalid and not used by assets.
    scales = torch.pow(2., scale.cpu().double() - 127).repeat_interleave(32,-1)
    return (v * scales).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    import musa_python_kernel_test_module as native
    torch.manual_seed(111105)
    result = {'status':'RUNNING','checks':[], 'scope':'GPU primitives only; no full-model acceptance or performance result'}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(result, indent=2))

    def check_quant(label: str, value: torch.Tensor) -> None:
        actual, scale = native.musa_v41_block32_qdq(value.to('musa').contiguous())
        expected, expected_scale = _quant_oracle(value)
        torch.testing.assert_close(scale.cpu(), expected_scale, rtol=0, atol=0)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        result['checks'].append({'name':label,'status':'PASS','shape':list(value.shape),'exact_qdq_and_scale':True})
        save()

    try:
        with torch.inference_mode():
            for rows in (0,1,17,256,2048):
                for inner in (32,64,2304,5120):
                    # Substantial per-block range; per-token quant is incorrect.
                    x = torch.randn(rows,inner,dtype=torch.bfloat16)
                    factor = torch.pow(2.,torch.randint(-20,12,(rows,inner//32)).float())
                    x = (x.float()*factor.repeat_interleave(32,-1)).bfloat16()
                    check_quant(f'random-{rows}-{inner}',x)
            for value in (0.,1e-8,1e-4,448.,896.,-448.,-896.):
                check_quant(f'constant-{value}',torch.full((1,32),value,dtype=torch.bfloat16))
            # Values on E4M3 rounding midpoints. A large final element fixes scale=1.
            ties = torch.tensor([1.0625,1.1875,-1.0625,-1.1875,.00390625,-.00390625,448.,-448.],dtype=torch.bfloat16).repeat(4).reshape(1,32)
            check_quant('round-to-nearest-even',ties)
            for rows in (0,1,2,8):
                for columns,inner in ((7,32),(65,64),(128,256),(2304,5120),(5120,2304)):
                    a = torch.randn(rows,inner,dtype=torch.bfloat16)
                    packed = torch.randint(0,256,(columns,inner//2),dtype=torch.uint8)
                    scale = torch.randint(123,131,(columns,inner//32),dtype=torch.uint8)
                    for quantized in (False,True):
                        aq = _quant_oracle(a)[0] if quantized else a
                        expected = (aq.double() @ _weight_oracle(packed,scale).double().T).bfloat16()
                        actual = native.musa_v41_packed_fp4_gemv(aq.to('musa'),packed.to('musa'),scale.to('musa'))
                        torch.testing.assert_close(actual.cpu(),expected,rtol=1/128,atol=1e-4)
                        diff = (actual.cpu().float()-expected.float()).abs()
                        result['checks'].append({'name':f'gemv-{rows}-{columns}-{inner}-qdq{quantized}','status':'PASS','shape':[rows,columns,inner],'rtol':1/128,'atol':1e-4,'max_abs':diff.max().item() if diff.numel() else 0.})
                        save()
            # Cover every nibble, signs, negative zero, and scale-byte zero.
            packed = torch.tensor([i | (i << 4) for i in range(16)],dtype=torch.uint8).reshape(16,1).repeat(1,16)
            scale = torch.zeros(16,1,dtype=torch.uint8)
            a = torch.full((1,32),2.**120,dtype=torch.bfloat16)
            actual = native.musa_v41_packed_fp4_gemv(a.to('musa'),packed.to('musa'),scale.to('musa'))
            expected = (a.double() @ _weight_oracle(packed,scale).double().T).bfloat16()
            torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
            result['checks'].append({'name':'all-nibbles-ue8m0-byte-zero','status':'PASS'})
            stream = torch.musa.Stream()
            x = torch.randn(2,64,dtype=torch.bfloat16,device='musa')
            packed = torch.randint(0,256,(65,32),dtype=torch.uint8,device='musa')
            scale = torch.full((65,2),127,dtype=torch.uint8,device='musa')
            stream.wait_stream(torch.musa.current_stream())
            with torch.musa.stream(stream):
                q,qs = native.musa_v41_block32_qdq((x+1).contiguous())
                y = native.musa_v41_packed_fp4_gemv(q,packed,scale)
            stream.synchronize()
            expected_q,expected_s = _quant_oracle((x+1).cpu())
            torch.testing.assert_close(q.cpu(),expected_q,rtol=0,atol=0)
            torch.testing.assert_close(qs.cpu(),expected_s,rtol=0,atol=0)
            expected_y = (expected_q.double() @ _weight_oracle(packed,scale).double().T).bfloat16()
            torch.testing.assert_close(y.cpu(),expected_y,rtol=1/128,atol=1e-4)
            result['checks'].append({'name':'nondefault-stream','status':'PASS'})
        # Exercise the actual model wrapper and the native routed expert.
        sys.modules['xllm_runtime'] = native
        from xllm.python import initialize_runtime
        initialize_runtime()
        from xllm.python.kernels_musa.fp4 import select_fp4_linear
        from xllm.python.models.deepseek_v41 import dequant_fp4_group, _swiglu_clamped, _PackedFP4Linear
        with torch.inference_mode():
            for mode in ('packed_bf16','block32'):
                os.environ['XLLM_DSV41_MUSA_FP4_LINEAR'] = mode
                op = select_fp4_linear(torch.device('musa:0'),torch.bfloat16,True,dequant_fp4_group)
                for rows in (0,1,8,17,256):
                    x = torch.randn(rows,64,dtype=torch.bfloat16)
                    w = torch.randint(0,256,(96,32),dtype=torch.uint8)
                    sc = torch.randint(120,127,(96,2),dtype=torch.uint8)
                    q = _quant_oracle(x)[0] if mode=='block32' else x
                    expected = (q.double() @ _weight_oracle(w,sc).double().T).bfloat16()
                    actual = op(x.to('musa'),w.to('musa'),sc.to('musa'))
                    torch.testing.assert_close(actual.cpu(),expected,rtol=1/128,atol=1e-4)
                    result['checks'].append({'name':f'wrapper-{mode}-{rows}','status':'PASS','large_m_implementation':'BF16 GEMM after explicit GPU QDQ' if mode=='block32' and rows>8 else 'selected packed GEMV or BF16 reference'})
                    save()
                for rows in (0,1,2,8):
                    # All expert linears obey QDQ, including the weighted DOWN input.
                    x = torch.randn(rows,128,dtype=torch.bfloat16,device='musa')
                    route = torch.linspace(.137,.913,rows,dtype=torch.float32,device='musa')
                    weights = [torch.randint(0,256,shape,dtype=torch.uint8,device='musa') for shape in ((64,64),(64,64),(128,32))]
                    scales = [torch.randint(120,123,(z.shape[0],z.shape[1]//16),dtype=torch.uint8,device='musa') for z in weights]
                    gate = op(x,weights[0],scales[0]);up = op(x,weights[1],scales[1])
                    down = _PackedFP4Linear(op,weights[2],scales[2])
                    python_out = _swiglu_clamped(x,gate,up,down,10.,route)
                    cpp_op = native.NativeV41PackedRoutedMlp(weights[0],scales[0],weights[1],scales[1],weights[2],scales[2],10.,mode=='block32')
                    cpp_out = cpp_op.forward(x,route)
                    torch.testing.assert_close(cpp_out.cpu(),python_out.cpu(),rtol=0,atol=0)
                    # Separate CPU oracle does not call the model wrapper or kernel.
                    xc = x.cpu();qc = _quant_oracle(xc)[0] if mode=='block32' else xc
                    wc = [_weight_oracle(a,b) for a,b in zip(weights,scales)]
                    g = (qc.double()@wc[0].double().T).bfloat16().float().clamp_max(10.)
                    u = (qc.double()@wc[1].double().T).bfloat16().float().clamp(-10.,10.)
                    h = (F.silu(g)*u*route.cpu().unsqueeze(-1)).bfloat16()
                    h = _quant_oracle(h)[0] if mode=='block32' else h
                    expected = (h.double()@wc[2].double().T).bfloat16()
                    torch.testing.assert_close(cpp_out.cpu(),expected,rtol=1/64,atol=5e-4)
                    result['checks'].append({'name':f'python-cpp-routed-expert-{mode}-{rows}','status':'PASS','python_cpp_bitwise_equal':True,'cpu_oracle_rtol':1/64,'cpu_oracle_atol':5e-4})
                    save()
        from xllm.python.models.deepseek_v41 import DeepseekV41Config, DeepseekV41MoE
        with torch.inference_mode():
            os.environ['XLLM_DSV41_MUSA_GATE']='reference'
            os.environ['XLLM_DSV41_MUSA_SHARED_SWIGLU']='reference'
            for mode in ('packed_bf16','block32'):
                os.environ['XLLM_DSV41_MUSA_FP4_LINEAR']=mode
                cfg=DeepseekV41Config(hidden_size=64,moe_intermediate_size=32,n_routed_experts=384,n_activated_experts=6,routed_scaling_factor=1.5,swiglu_limit=10.)
                model=DeepseekV41MoE(cfg,0,torch.bfloat16,torch.device('musa:0'),packed_fp4=True)
                model.gate.weight.zero_()
                model.e_score_correction_bias.copy_(torch.arange(384,device='musa',dtype=torch.float32))
                for name in ('w1','w2','w3'):
                    weight=getattr(model,'experts_'+name);scales=getattr(model,'experts_'+name+'_scale')
                    weight.copy_(torch.randint(0,256,weight.shape,dtype=torch.uint8,device='musa'))
                    scales.copy_(torch.randint(120,123,scales.shape,dtype=torch.uint8,device='musa'))
                    sw=getattr(model,'shared_'+name).weight
                    sw.copy_(torch.randn_like(sw)*.025)
                for rows in (0,1,17):
                    x=torch.randn(rows,64,dtype=torch.bfloat16,device='musa')
                    actual=model(x).cpu();xc=x.cpu();expected=torch.zeros(rows,64,dtype=torch.float32)
                    def cpu_linear(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
                        return (a.double()@weight.cpu().double().T).bfloat16()
                    for expert in range(378,384):
                        q=_quant_oracle(xc)[0] if mode=='block32' else xc
                        ws=[_weight_oracle(getattr(model,'experts_'+name)[expert],getattr(model,'experts_'+name+'_scale')[expert]) for name in ('w1','w3','w2')]
                        gate=cpu_linear(q,ws[0]).float().clamp_max(10.)
                        up=cpu_linear(q,ws[1]).float().clamp(-10.,10.)
                        hidden=(F.silu(gate)*up*.25).bfloat16()
                        hidden=_quant_oracle(hidden)[0] if mode=='block32' else hidden
                        expected+=cpu_linear(hidden,ws[2]).float()
                    gate=cpu_linear(xc,model.shared_w1.weight).float().clamp_max(10.)
                    up=cpu_linear(xc,model.shared_w3.weight).float().clamp(-10.,10.)
                    expected+=cpu_linear((F.silu(gate)*up).bfloat16(),model.shared_w2.weight).float()
                    expected=expected.bfloat16()
                    torch.testing.assert_close(actual,expected,rtol=1/64,atol=5e-4)
                    result['checks'].append({'name':f'model-moe-384-top6-EP1-{mode}-{rows}','status':'PASS','scope':'single-GPU real model MoE forward; not EP8 communication','rtol':1/64,'atol':5e-4})
                    save()
        result.update(status='PASS',count=len(result['checks']),device=torch.musa.get_device_name(0),torch_version=torch.__version__)
        save()
        logger.info('Original FP4/block32 primitive checks passed: %s',len(result['checks']))
    except Exception:
        result['status']='FAIL'
        save()
        raise


if __name__ == '__main__':
    main()
