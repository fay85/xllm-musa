import sys,os,json
from pathlib import Path
import torch,torch_musa
import musa_python_kernel_test_module as native
sys.modules['xllm_runtime']=native
from xllm.python import initialize_runtime
initialize_runtime()
from xllm.python.models.deepseek_v41 import _hc_split_sinkhorn
out=Path('/work/20261110_ops/stage2/evidence');torch.manual_seed(111002)
# Reproduce the RNG sequence used by the first unit run exactly.
for t in (0,1,17,256,2048):
 r=torch.randn(t,4,10240,device='musa',dtype=torch.bfloat16)
 x=torch.randn(t,10240,device='musa',dtype=torch.bfloat16)
 mix=torch.randn(t,48,device='musa');base=torch.randn(24,device='musa')*.1
mix=torch.randn(17,24,device='musa')*30;scale=torch.ones(3,device='musa');base=torch.zeros(24,device='musa')
records=[];values={'mix':mix.cpu(),'scale':scale.cpu(),'base':base.cpu()}
for it in (1,2,20):
 got=native.musa_hc_metadata(mix,scale,base,it,1e-6)
 gpu=_hc_split_sinkhorn(mix,scale,base,4,it,1e-6)
 cpu=_hc_split_sinkhorn(mix.cpu(),scale.cpu(),base.cpu(),4,it,1e-6)
 f64=_hc_split_sinkhorn(mix.cpu().double(),scale.cpu().double(),base.cpu().double(),4,it,1e-6)
 for k,a,b,c,d in zip(('pre','post','comb'),got,gpu,cpu,f64):
  a,b=a.cpu(),b.cpu();values[f'{it}_{k}']={'native':a,'gpu_torch':b,'cpu':c,'f64':d}
  records.append({'iters':it,'part':k,'native_vs_gpu':float((a-b).abs().max()),'native_vs_f64':float((a-d).abs().max()),'gpu_vs_f64':float((b-d).abs().max()),'cpu_vs_f64':float((c-d).abs().max())})
torch.save(values,out/'mhc_diagnostic.pt');(out/'mhc_diagnostic.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
print('worst example',values['1_comb']['native'][0],values['1_comb']['gpu_torch'][0],values['1_comb']['f64'][0])
a=mix[:,8:].reshape(17,4,4)
row_gpu=torch.softmax(a,-1);row_cpu=torch.softmax(a.cpu().double(),-1)
row_eps=row_gpu+1e-6
sum_gpu=row_eps.sum(1,keepdim=True)
col_gpu=row_eps/(sum_gpu+1e-6)
stages={'softmax':float((row_gpu.cpu().double()-row_cpu).abs().max()),'sum':float((sum_gpu.cpu().double()-row_eps.cpu().double().sum(1,keepdim=True)).abs().max()),'divide':float((col_gpu.cpu().double()-row_eps.cpu().double()/(sum_gpu.cpu().double()+1e-6)).abs().max())}
(out/'mhc_stage_diagnostic.json').write_text(json.dumps(stages,indent=2));print(stages)
a=(mix[:,8:].unflatten(-1,(4,4))*scale[2]+base[8:].unflatten(-1,(4,4)))
p=torch.softmax(a,dim=-1);q=p+1e-6;c=q/(q.sum(1,keepdim=True)+1e-6)
bad=(c.cpu().double()-values['1_comb']['f64']).abs();idx=torch.unravel_index(bad.argmax(),bad.shape);t=int(idx[0]);print('worst token',t,'error',float(bad.max()));print('logits',a[t].cpu());print('gpu softmax',p[t].cpu());print('cpu softmax',torch.softmax(a[t].cpu().double(),-1));print('col actual',c[t].cpu());print('col oracle',values['1_comb']['f64'][t]);print('standalone-vs-function',float((c.cpu()-values['1_comb']['gpu_torch']).abs().max()))
print('q',q[t].cpu());print('q exact',p[t].cpu().double()+1e-6);sg=q.sum(1,keepdim=True);print('sum GPU',sg[t].cpu());print('sum CPU',q[t].cpu().double().sum(0,keepdim=True));print('denom GPU',(sg+1e-6)[t].cpu());print('denom CPU',sg[t].cpu().double()+1e-6)
torch.save({'logits':a.cpu(),'softmax':p.cpu(),'eps_added':q.cpu(),'column_sum':sg.cpu(),'denominator':(sg+1e-6).cpu(),'normalized':c.cpu()},out/'mhc_stage_diagnostic.pt')
