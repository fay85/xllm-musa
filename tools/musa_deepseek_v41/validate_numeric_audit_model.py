# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Actual FP8 checkpoint loader plus six-layer combined MUSA execution fixture."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401

from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bmm-so', type=Path, required=True)
    parser.add_argument('--engram-gate', choices=('reference', 'original'), default='reference')
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    for name in ('SHARED_SWIGLU', 'ROPE', 'HC_POST', 'HC_PRE', 'HC_METADATA'):
        os.environ['XLLM_DSV41_MUSA_' + name] = 'v4_fused'
    os.environ.update(XLLM_DSV41_MUSA_GATE='v41_fused', XLLM_DSV41_MUSA_ATTENTION_QUANT='original_qdq',
                      XLLM_DSV41_MUSA_DENSE_QUANT='original_qdq', XLLM_DSV41_MUSA_FP4_LINEAR='partition32',
                      XLLM_DSV41_MUSA_FP4_MOE='v41_decode', XLLM_DSV41_MATE_BMM_SO=str(args.bmm_so),
                      XLLM_TILELANG_LIB='/usr/local/lib/python3.10/dist-packages/tilelang/lib/libtilelang.so')
    os.environ['XLLM_DSV41_MUSA_ENGRAM_GATE'] = args.engram_gate
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime'] = native
    from xllm.python import initialize_runtime
    initialize_runtime()
    from tests.python.test_deepseek_v41_model import (
        _DEEPSEEK_V41_FORWARD_CONFIG, _FakeStateDict, _V41ForwardHarness, _build_tiny_checkpoint,
    )
    from xllm.python.kernels_musa.dense_quant import V41OriginalFp8Dense
    from xllm.python.models.deepseek_v41 import DeepseekV41ForCausalLM

    fixture = args.output.with_suffix('.fixture')
    fixture.mkdir(parents=True, exist_ok=False)
    config = {**_DEEPSEEK_V41_FORWARD_CONFIG, 'device': 'musa:0', 'dtype': 'bfloat16',
              'model_path': str(fixture), 'enable_graph': False, 'num_speculative_tokens': 0,
              'quantization_config': {'quant_method': 'fp8', 'weight_block_size': [32, 32],
                                      'scale_fmt': 'ue8m0', 'expert_dtype': 'fp4'}}
    (fixture/'config.json').write_text(json.dumps(config, indent=2))
    torch.manual_seed(111121)
    with torch.inference_mode():
        os.environ['XLLM_DSV41_NUMERIC_AUDIT_DIR'] = str(args.output.with_suffix('.audit'))
        model = DeepseekV41ForCausalLM(config)
        tensors = _build_tiny_checkpoint(model.cfg)
        for name, value in tensors.items():
            if value.is_floating_point() and 'norm.weight' not in name:
                value.mul_(.02)
            if name.endswith('engram.embed.weight'):
                value.copy_(torch.randint(0, 112, value.shape, dtype=torch.uint8))
            if name.endswith('engram.embed.scale'):
                value.fill_(120)
        dense_prefixes = []
        for index in range(model.cfg.n_layers):
            base = f'layers.{index}.'
            dense_prefixes.extend(base+'attn.'+key for key in ('wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b'))
            dense_prefixes.extend(base+'ffn.shared_experts.'+key for key in ('w1', 'w2', 'w3'))
            if base+'attn.indexer.wq_b.weight' in tensors:
                dense_prefixes.append(base+'attn.indexer.wq_b')
            if index in model.cfg.engram_layer_ids:
                dense_prefixes.append(base+'engram.wkv')
            for expert in range(model.cfg.n_routed_experts):
                for key in ('w1', 'w2', 'w3'):
                    prefix=base+f'ffn.experts.{expert}.{key}'
                    n,k=tensors[prefix+'.weight'].shape
                    tensors[prefix+'.weight']=torch.randint(0,256,(n,k//2),dtype=torch.uint8)
                    tensors[prefix+'.scale']=torch.full((n,k//32),120,dtype=torch.uint8)
        for prefix in dense_prefixes:
            weight=tensors[prefix+'.weight'];n,k=weight.shape
            tensors[prefix+'.weight']=weight.to(torch.float8_e4m3fn)
            tensors[prefix+'.scale']=torch.full((-(n//-32),-(k//-32)),127,dtype=torch.uint8)
        model.load_weights([_FakeStateDict(tensors)],0,1)
        wrapped=[name for name,module in model.named_modules() if isinstance(module,V41OriginalFp8Dense)]
        assert len(wrapped)==len(dense_prefixes)-model.cfg.n_layers
        device=torch.device('musa:0');nseq=1;length=model.cfg.window_size+1
        ids=torch.randint(0,model.cfg.vocab_size,(nseq,length),device=device)
        full=_V41ForwardHarness(model,config,nseq,torch.bfloat16,device,original_attention_qdq=True)
        hidden=full.step(ids.flatten(),torch.arange(length,device=device).repeat(nseq),[length]*nseq,[length]*nseq,is_prefill=True)
        expected=model.lm_head(hidden).float().reshape(nseq,length,-1)
        selected=model.compute_logits(hidden,torch.tensor([length-1],device=device))
        torch.testing.assert_close(selected.float(),expected[:,-1],rtol=0,atol=0)
        assert bool(torch.isfinite(expected).all()) and float(expected.abs().max())>1e-4
        assert float((expected[:,1:]-expected[:,:-1]).abs().max())>1e-5
        stepwise=_V41ForwardHarness(model,config,nseq,torch.bfloat16,device,original_attention_qdq=True)
        outputs=[];errors=[]
        for position in range(length):
            h=stepwise.step(ids[:,position],torch.full((nseq,),position,device=device),[position+1]*nseq,[1]*nseq,is_prefill=False)
            actual=stepwise.logits(h);errors.append(float((actual-expected[:,position]).abs().max()))
            torch.testing.assert_close(actual,expected[:,position],rtol=0,atol=.02)
            outputs.append(actual.cpu())
        torch.save({'ids':ids.cpu(),'prefill_logits':expected.cpu(),'decode_logits':torch.stack(outputs,dim=1)},args.output.with_suffix('.tensors.pt'))
    result={'status':'PASS','scope':'six-layer synthetic FP8/FP4 checkpoint loaded through real loader; single MUSA GPU; not1017fullmodel or networkPD',
            'wrapped_dense_modules':wrapped,'engram_layers':model.cfg.engram_layer_ids,'tokens_per_request':length,
            'sequences':nseq,'max_abs_logits_by_position':errors,'max_abs_logits':float(expected.abs().max()),
            'activation_modes':{'dense':'original_qdq','attention':'original_qdq','expert':'partition32/v41_decode','engram_gate':args.engram_gate},'logits_atol':.02}
    root=args.output.with_suffix('.audit')
    frames=sorted(root.glob('*.hidden.pt'));logits_files=sorted(root.glob('*.logits.pt'))
    assert len(frames)==length+1 and len(logits_files)==length+1
    first=torch.load(frames[0],weights_only=True);last=torch.load(frames[-1],weights_only=True)
    for key in first['layers']:
        for field in ('hidden','pre_mix'):
            torch.testing.assert_close(first['layers'][key][field],last['layers'][key][field],rtol=0,atol=0)
    full_logits=torch.load(logits_files[0],weights_only=True);step_logits=torch.load(logits_files[-1],weights_only=True)
    torch.testing.assert_close(full_logits['logits'],step_logits['logits'],rtol=0,atol=0)
    result.update(audit_hidden_frames=len(frames),audit_logits_frames=len(logits_files),audit_layers=len(first['layers']),
                  audit_last_position=full_logits['position'],audit_full_vocab=full_logits['logits'].shape[-1],
                  audit_full_prefill_vs_decode_lastrow='bitwise hidden/pre_mix/logits')
    args.output.write_text(json.dumps(result,indent=2))
    logger.info('Loaded dense/attention/expert combined fixture passed; maxlogitdifference=%s',max(errors))


if __name__=='__main__':
    main()
