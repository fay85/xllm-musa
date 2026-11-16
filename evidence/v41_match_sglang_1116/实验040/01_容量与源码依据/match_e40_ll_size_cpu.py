from pathlib import Path
import json,os,subprocess
r=Path('/expt');env=json.loads((r/'diagnostics/e38_deepep_w4a8_v3/environment.json').read_text());env.update(MUSA_VISIBLE_DEVICES='-1',LD_PRELOAD='');code="""import json,torch
from deep_ep import Buffer
sizes=[{'experts':e,'capacity':c,'num_rdma_bytes':Buffer.get_low_latency_rdma_size_hint(c,5120,8,e)} for e in [384,128] for c in [16,32]]
x=torch.empty((2304,5120),dtype=torch.bfloat16);a=x.narrow(0,0,288).contiguous();y=torch.empty((5120,2304),dtype=torch.bfloat16);b=y.narrow(1,0,288).contiguous()
print(json.dumps({'rdma_hints':sizes,'cpu_storage_check':{'gate_logical_bytes':a.numel()*2,'gate_storage_bytes':a.untyped_storage().nbytes(),'gate_aliases_original':a.data_ptr()==x.data_ptr(),'down_logical_bytes':b.numel()*2,'down_storage_bytes':b.untyped_storage().nbytes(),'down_aliases_original':b.data_ptr()==y.data_ptr()},'scope':'CPU only; no Buffer allocated, hints do not include all mtshmem metadata'}))
"""
p=subprocess.run(['/opt/venv/bin/python3','-c',code],env=env,text=True,capture_output=True,timeout=40);out=r/'diagnostics/e40_dp8_memory_budget/ll_size_cpu.json';out.write_text(json.dumps({'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2));print(p.stdout);print(p.stderr);assert p.returncode==0
