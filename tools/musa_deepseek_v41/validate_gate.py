# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Check V4.1 gate semantics and native C++ gate/shared MLP on one idle GPU."""
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
from xllm.python import initialize_runtime
from xllm.python.kernels_musa.gating import select_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose one verified idle GPU')
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime'] = native
    initialize_runtime()
    os.environ['XLLM_DSV41_MUSA_GATE'] = 'v41_fused'
    os.environ['XLLM_DSV41_MUSA_SHARED_SWIGLU'] = 'v4_fused'
    from xllm.python.models.deepseek_v41 import DeepseekV41Config, DeepseekV41MoE, DeepseekV41TorchMLP
    torch.manual_seed(111013)
    checks = []
    result = {'status': 'RUNNING', 'checks': checks, 'scope': 'GPU gate and native sublayers; not full-model or EP communication acceptance'}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(result, indent=2))

    def check_gate(label: str, logits: torch.Tensor, bias: torch.Tensor, k: int, norm: bool, stable: bool = False) -> None:
        op = select_gate(torch.device('musa:0'), logits.shape[1], k, norm, 1.5)
        weights, ids = op(logits, bias)
        scores = F.softplus(logits).sqrt()
        # Independent oracle for the fused mathematical function. The
        # captured TorchMUSA softplus(-20/-80) returned zero, whereas CPU
        # FP32/FP64 and the stable fused implementation remain positive.
        oracle_scores = F.softplus(logits.cpu().double()).sqrt()
        selected = oracle_scores.float() + bias.cpu()
        if stable:
            reference_ids = torch.argsort(selected, dim=-1, descending=True, stable=True)[:, :k]
        else:
            reference_ids = selected.topk(k, dim=-1).indices
        ref_weights = oracle_scores.gather(1, ids.cpu())
        legacy_weights = scores.gather(1, ids)
        if norm and k > 1:
            ref_weights = ref_weights / (ref_weights.sum(-1, keepdim=True) + 1e-20)
            legacy_weights = legacy_weights / (legacy_weights.sum(-1,keepdim=True)+1e-20)
        ref_weights = (ref_weights * 1.5).float()
        legacy_weights = legacy_weights * 1.5
        try:
            torch.testing.assert_close(ids.cpu(), reference_ids, rtol=0, atol=0)
            torch.testing.assert_close(weights.cpu(), ref_weights.cpu(), rtol=2e-5, atol=1e-6, equal_nan=True)
        except AssertionError:
            result['status'] = 'FAIL'
            result['failed_case'] = label
            torch.save({'logits':logits.cpu(),'bias':bias.cpu(),'ids':ids.cpu(),'reference_ids':reference_ids,'weights':weights.cpu(),'reference_weights':ref_weights.cpu()}, args.output.with_suffix('.failure.pt'))
            save()
            raise
        checks.append({'name':label, 'status':'PASS', 'tokens':logits.shape[0], 'experts':logits.shape[1], 'topk':k, 'normalize':norm,'stable_tie_policy':stable,'legacy_torch_max_abs_error':((legacy_weights.cpu()-ref_weights).abs().max().item() if ref_weights.numel() else 0.0),'oracle':'CPU FP64 fused function; selection rounded to FP32 before bias'})
        save()

    with torch.inference_mode():
        for experts in (4,128,256,384,512,1024):
            for tokens in (0,1,17,256):
                logits = torch.randn(tokens,experts*2,device='musa')[:,::2]
                for k in (1,min(6,experts)):
                    for norm in (False,True):
                        for modal in (False,True):
                            bias = torch.randn(tokens,experts*2,device='musa')[:,::2]*.1 if modal else torch.randn(experts,device='musa')*.1
                            check_gate(f'random_E{experts}_T{tokens}_K{k}_norm{norm}_modal{modal}',logits,bias,k,norm)
        for label,logits,bias in (
            ('ties',torch.zeros(3,384,device='musa'),torch.zeros(384,device='musa')),
            ('large_bias',torch.randn(3,384,device='musa'),torch.full((384,),1e10,device='musa')),
            ('tiny_scores',torch.full((3,384),-80.,device='musa'),torch.zeros(384,device='musa')),
            ('subnormal_scores',torch.full((3,384),-100.,device='musa'),torch.zeros(384,device='musa')),
        ):
            check_gate(label,logits,bias,6,True,stable=True)
        for hidden in (64,5120):
            cfg=DeepseekV41Config(hidden_size=hidden,moe_intermediate_size=32,n_routed_experts=384,n_activated_experts=6,routed_scaling_factor=1.5)
            for tokens in (0,1,17):
                py=DeepseekV41MoE(cfg,0,torch.bfloat16,torch.device('musa:0'))
                py.gate.weight.normal_(0,.02)
                py.e_score_correction_bias.normal_(0,.1)
                py.gate_bias_vl.normal_(0,.1)
                x=torch.randn(tokens,hidden,device='musa',dtype=torch.bfloat16)
                ids=torch.arange(tokens,device='musa')%2+cfg.image_token_id
                for modal in (False,True):
                    py.gate_bias_vl_loaded=modal
                    cpp=native.NativeV41Gate(py.gate.weight,py.e_score_correction_bias,py.gate_bias_vl if modal else None,cfg.image_token_id,6,True,1.5)
                    actual=cpp.forward(x,ids if modal else None)
                    logits,bias=py._routing_logits_and_bias(x,ids if modal else None)
                    expected=py._musa_gate(logits,bias)
                    torch.testing.assert_close(actual[1],expected[1],rtol=0,atol=0)
                    torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
                    checks.append({'name':f'native_gate_H{hidden}_T{tokens}_modal{modal}','status':'PASS'})
                    save()
        for hidden,inter in ((64,32),(5120,2304)):
            cfg=DeepseekV41Config(hidden_size=hidden,moe_intermediate_size=inter)
            py=DeepseekV41TorchMLP(cfg,torch.bfloat16,torch.device('musa:0'))
            for p in py.parameters():p.normal_(0,.02)
            cpp=native.NativeV41SharedMlp(py.w1.weight,py.w3.weight,py.w2.weight,cfg.swiglu_limit)
            for tokens in (0,1,17,256):
                for amplitude in (1.,30.):
                    x=torch.randn(tokens,hidden*2,device='musa',dtype=torch.bfloat16)[:,::2]*amplitude
                    actual,expected=cpp.forward(x),py(x)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    checks.append({'name':f'native_shared_H{hidden}_I{inter}_T{tokens}_amp{amplitude}','status':'PASS','bitwise_equal':True})
                    save()
        stream=torch.musa.Stream()
        with torch.musa.stream(stream):
            check_gate('nondefault_stream',torch.randn(17,384,device='musa'),torch.randn(17,384,device='musa')*.1,6,True)
        stream.synchronize()
    result['status']='PASS'
    save()
    logger.info('Gate/native sublayer checks passed: %s',len(checks))


if __name__ == '__main__':
    main()
