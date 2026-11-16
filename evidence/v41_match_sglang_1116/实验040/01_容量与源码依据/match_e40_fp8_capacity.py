from pathlib import Path
import json,math,datetime
out=Path('/expt/diagnostics/e40_dp8_memory_budget');data=json.loads((out/'checkpoint_header_inventory.json').read_text());t=data['tensors'];G=1024**3;rows=[]
for kind,count in [('layers',40),('mtp',3)]:
 for layer in range(count):
  for suffix in ['attn.wq_a','attn.wq_b','attn.wkv','attn.wo_b','ffn.shared_experts.w1','ffn.shared_experts.w3','ffn.shared_experts.w2']:
   key=f'{kind}.{layer}.{suffix}.weight';scale=key[:-6]+'scale';v=t[key];sv=t[scale];assert v['dtype'] in ['F8_E4M3','F8_E4M3FN'] and len(v['shape'])==2,(key,v)
   n,k=v['shape'];assert sv['shape']==[n//32,k//32]
   bf=n*k*2;fp=n*k+math.prod(sv['shape'])*4
   rows.append({'key':key,'shape':[n,k],'BF16_bytes':bf,'FP8_FP32scale_bytes':fp,'saved_bytes':bf-fp,'category':kind+'_'+('shared' if 'shared_experts' in suffix else 'attention')})
groups={k:sum(x['saved_bytes'] for x in rows if x['category']==k)/G for k in sorted({x['category'] for x in rows})};saved=sum(x['saved_bytes'] for x in rows)/G;prev=json.loads((out/'parameter_budget.json').read_text());full=prev['scenarios']['target_and_draft_dp8']['rank0']['total_parameter_storage_GiB'];v={'at':datetime.datetime.now().astimezone().isoformat(),'assumption':'Only listed 2D linears retain original FP8 and decoded FP32 scales; grouped O-A remains BF16; no duplicate BF16 carriers kept','groups_saved_GiB':groups,'total_saved_GiB':saved,'full_DP_parameter_storage_GiB_before':full,'full_DP_parameter_storage_GiB_after':full-saved,'excluded':'Graph/cache/runtime/load peaks still need verification, not allocation permission','rows':rows};(out/'fp8_capacity_delta.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:x for k,x in v.items() if k!='rows'}))
