# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""EP8 regression for original packed experts, with empty and busy ranks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401
import torch.distributed as dist
import torch.nn.functional as F

from scripts.logger import logger
from tools.musa_deepseek_v41.validate_fp4 import _quant_oracle, _weight_oracle


def _weight(expert: int, branch: int, columns: int, inner: int) -> tuple[torch.Tensor, torch.Tensor]:
    byte = (torch.arange(columns * inner // 2).reshape(columns,inner//2) + expert * 13 + branch * 7) % 256
    return byte.to(torch.uint8), torch.full((columns,inner//32),120 + expert % 3,dtype=torch.uint8)


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.double() @ weight.double().T).bfloat16()


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    rank=args.rank;size=8
    torch.musa.set_device(rank)
    device=torch.device(f'musa:{rank}')
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime']=native
    from xllm.python import initialize_runtime
    initialize_runtime()
    from xllm.python.distributed.collectives import init_process_group
    from xllm.python.models.deepseek_v41 import DeepseekV41MoE,DeepseekV41Config
    init_process_group('moe_ep','127.0.0.1',args.port,rank,size,str(device),rank,size,0,backend='mccl')
    result={'rank':rank,'status':'RUNNING','checks':[],'scope':'single-host EP8 via real framework Python process group/MCCL; not native serving communicator or cross-host PD'}
    out=args.output_dir/f'rank{rank}.json';out.parent.mkdir(exist_ok=True,parents=True)
    def save() -> None:
        out.write_text(json.dumps(result,indent=2))
    save()
    try:
        with torch.inference_mode():
            os.environ['XLLM_DSV41_MUSA_GATE']='v41_fused'
            os.environ['XLLM_DSV41_MUSA_SHARED_SWIGLU']='v4_fused'
            for mode in ('packed_bf16','block32'):
                os.environ['XLLM_DSV41_MUSA_FP4_LINEAR']=mode
                cfg=DeepseekV41Config(hidden_size=64,moe_intermediate_size=32,n_routed_experts=384,n_activated_experts=6,routed_scaling_factor=1.5,swiglu_limit=10.)
                cfg.ep_size=size;cfg.ep_rank=rank
                model=DeepseekV41MoE(cfg,0,torch.bfloat16,device,packed_fp4=True)
                model.gate.weight.zero_()
                for i,name in enumerate(('w1','w3','w2')):
                    w=getattr(model,'experts_'+name);sc=getattr(model,'experts_'+name+'_scale')
                    for slot in range(48):
                        weight,scale=_weight(rank*48+slot,i,w.shape[1],w.shape[2]*2)
                        w[slot].copy_(weight);sc[slot].copy_(scale)
                    getattr(model,'shared_'+name).weight.fill_(.003)
                for pattern,selected in [('concentrated',list(range(378,384))),('spread',[0,48,96,144,192,240])]:
                    model.e_score_correction_bias.zero_()
                    for i,expert in enumerate(selected):model.e_score_correction_bias[expert]=i+10
                    for rows in (0,1,17):
                        torch.manual_seed(71+rows)
                        xc=torch.randn(rows,64,dtype=torch.bfloat16)
                        actual=model(xc.to(device)).cpu()
                        expected=torch.zeros(rows,64,dtype=torch.float32)
                        for expert in selected:
                            weights=[_weight(expert,i,n,k) for i,(n,k) in enumerate(((32,64),(32,64),(64,32)))]
                            w=[_weight_oracle(a,b) for a,b in weights]
                            q=_quant_oracle(xc)[0] if mode=='block32' else xc
                            gate=_linear(q,w[0]).float().clamp_max(10.)
                            up=_linear(q,w[1]).float().clamp(-10.,10.)
                            h=(F.silu(gate)*up*.25).bfloat16()
                            h=_quant_oracle(h)[0] if mode=='block32' else h
                            expected+=_linear(h,w[2]).float()
                        w1=model.shared_w1.weight.cpu();w3=model.shared_w3.weight.cpu();w2=model.shared_w2.weight.cpu()
                        gate=_linear(xc,w1).float().clamp_max(10.);up=_linear(xc,w3).float().clamp(-10.,10.)
                        expected+=_linear((F.silu(gate)*up).bfloat16(),w2).float()
                        torch.testing.assert_close(actual,expected.bfloat16(),rtol=1/64,atol=5e-4)
                        result['checks'].append({'mode':mode,'pattern':pattern,'tokens':rows,'status':'PASS','selected_global_experts':selected,'local_selected':sum(rank*48<=e<(rank+1)*48 for e in selected),'shared_count':1,'rtol':1/64,'atol':5e-4})
                        save()
        result['status']='PASS';save()
        logger.info('EP rank %s passed %s cases',rank,len(result['checks']))
    except Exception:
        result['status']='FAIL';save();raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
