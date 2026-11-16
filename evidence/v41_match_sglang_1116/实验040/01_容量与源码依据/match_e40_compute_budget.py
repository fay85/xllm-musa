from pathlib import Path
import json,math,struct,collections,datetime
out=Path('/expt/diagnostics/e40_dp8_memory_budget');data=json.loads((out/'checkpoint_header_inventory.json').read_text());t=data['tensors'];c=data['config']['text_config'];GiB=1024**3
model=Path('/work/artifacts/model_view_w4afp8_mt_1017');hashfile=model/'engram_hash_state.native'
with hashfile.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));p=h['primes'];assert p['dtype']=='I64';f.seek(8+n+p['data_offsets'][0]);primes=struct.unpack('<'+'q'*math.prod(p['shape']),f.read(p['data_offsets'][1]-p['data_offsets'][0]))
assert p['shape']==[2,3,8]
def budget(rank,full_target=False,full_draft=False):
 groups=collections.defaultdict(int);entries=[]
 def add(key,group,size=2,split=1,storage_split=None):
  assert key in t,key
  count=math.prod(t[key]['shape']);assert count%split==0
  b=count*size//(storage_split if storage_split is not None else split);groups[group]+=b;entries.append({'key':key,'category':group,'bytes':b,'logical_split':split,'storage_split':storage_split if storage_split is not None else split})
 for kind,layers,full in [('layers',40,full_target),('mtp',3,full_draft)]:
  for l in range(layers):
   prefix=f'{kind}.{l}.';attn=prefix+'attn.';cat=kind+'_attention';atp=1 if full else 8
   for name in ['wq_a','wq_b','wkv','wo_a','wo_b']:add(attn+name+'.weight',cat,split=atp if name in ['wq_b','wo_a','wo_b'] else 1)
   for name in ['q_norm.weight','kv_norm.weight']:add(attn+name,cat)
   add(attn+'attn_sink',cat,4,atp)
   if kind=='layers' and l in c['index_source_layer_ids']:
    for name in ['wq_b.weight','weights_proj.weight']:add(attn+'indexer.'+name,kind+'_index_compression')
   if kind=='layers' and l in c['kv_source_layer_ids']:
    for name in ['wk.weight','k_norm.weight']:add(attn+'indexer.'+name,kind+'_index_compression')
    add(attn+'compressor.wkv.weight',kind+'_index_compression',4 if c['compress_ratios'][l]==2 else 2)
    add(attn+'compressor.norm.weight',kind+'_index_compression')
    if c['compress_ratios'][l]==2:add(attn+'compressor.wgate.weight',kind+'_index_compression',4)
   for name in ['attn_norm.weight','ffn_norm.weight']:add(prefix+name,kind+'_norm_hc_gate')
   for part in ['attn','ffn']:
    for term in ['fn','scale','base']:add(prefix+'hc_'+part+'_'+term,kind+'_norm_hc_gate',4)
   for name in ['weight','bias','bias_vl']:
    key=prefix+'ffn.gate.'+name
    if key in t:add(key,kind+'_norm_hc_gate',2 if name=='weight' else 4)
   for name in ['w1','w3','w2']:
    # Current load does full GPU H2D before narrow. dim0 contiguous keeps full storage.
    split=1 if full else 8;storage=split if name=='w2' else 1
    add(prefix+'ffn.shared_experts.'+name+'.weight',kind+'_shared',split=split,storage_split=storage)
   for key,v in t.items():
    if not key.startswith(prefix+'ffn.experts.'):continue
    expert=int(key[len(prefix+'ffn.experts.'):].split('.')[0]);experts=c['n_routed_experts'] if kind=='layers' else c['dspark_n_routed_experts']
    if not rank*(experts//8)<=expert<(rank+1)*(experts//8):continue
    if key.endswith('.weight'):add(key,kind+'_routed_weight',1)
    elif key.endswith('.weight_scale_inv'):add(key,kind+'_routed_scale',2)
   if kind=='layers' and l in c['engram_layer_ids']:
    idx=c['engram_layer_ids'].index(l);bucketcount=sum(primes[idx*24+rank*3:idx*24+(rank+1)*3]);assert bucketcount>0;groups['engram_sharded_table']+=bucketcount*264
    for name in ['wkv.weight','q_weight','k_weight']:add(prefix+'engram.'+name,'engram_projection')
 for key in ['embed.weight','head.weight']:add(key,'target_embed_head',split=8)
 add('norm.weight','target_embed_head')
 for key,fallback in [('mtp.0.embed.weight','embed.weight'),('mtp.2.head.weight','head.weight')]:add(key if key in t else fallback,'draft_embed_head',split=8)
 for key in ['mtp.0.main_proj.weight','mtp.0.main_norm.weight','mtp.2.norm.weight','mtp.2.markov_head.embed.weight','mtp.2.markov_head.head.weight']:add(key,'draft_other')
 for key in t:
  if key.startswith('mtp.2.confidence_head.'):add(key,'draft_other')
 return {'rank':rank,'groups_bytes':dict(groups),'groups_GiB':{k:v/GiB for k,v in groups.items()},'total_parameter_storage_bytes':sum(groups.values()),'total_parameter_storage_GiB':sum(groups.values())/GiB,'entries':entries}
# Rank differences are Engram bucket rows; other expert shapes identical, evaluated rank0 fully then adjust.
r0=budget(0);scenarios={}
for name,a,b in [('current_tp8',False,False),('target_dp8_draft_tp8',True,False),('target_and_draft_dp8',True,True)]:
 v=r0 if name=='current_tp8' else budget(0,a,b);v.pop('entries',None);ranks=[]
 for rank in range(8):
  en=sum(sum(primes[idx*24+rank*3:idx*24+(rank+1)*3])*264 for idx in range(2));total=v['total_parameter_storage_bytes']-v['groups_bytes']['engram_sharded_table']+en;ranks.append({'rank':rank,'engram_GiB':en/GiB,'parameter_storage_GiB':total/GiB})
 scenarios[name]={'rank0':v,'ranks':ranks,'extra_GiB_vs_current':v['total_parameter_storage_GiB']-r0['total_parameter_storage_GiB']}
result={'at':datetime.datetime.now().astimezone().isoformat(),'scenarios':scenarios,'not_included':['Rope and hash metadata','KV/draft/Engram recurrent state','allocator slack, library/workspace, custom AR','DeepEP LL RDMA/NVL buffers','Graph pools and capture/load transient peak'],'warnings':['Parameter storage only, not allocation permission or full model fit proof','current shared gate/up narrow(dim0).contiguous retains whole original GPU storage; down dim1 makes copy','DP layout assumes complete local shared expert and same TP8 head/Engram partition; owner/idle routing not implemented','packed QKV/shared gate-up optional duplicate storage must be checked against effective environment'],'scope':'CPU header-only analysis; no GPU allocation or model change'}
(out/'parameter_budget.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
