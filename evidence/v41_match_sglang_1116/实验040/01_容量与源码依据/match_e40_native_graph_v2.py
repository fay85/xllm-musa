from pathlib import Path
base=Path('/expt/diagnostics/e40_prequant_native_gpu_v2/probe.py').read_text()
exec(compile(base.split('specs=')[0],'/expt/diagnostics/e40_prequant_native_gpu_v2/probe.py','exec'))
r=Path('/expt/diagnostics/e40_prequant_native_graph_v2');records=[];comparisons=[]
def sg_prequant(x,w,s,splits):
 m,k=x.shape;n=w.shape[0];assert k%(32*splits)==0 and n%32==0
 a,ascale=sglang_per_token_group_quant_fp8(x,32,eps=1e-4,scale_ue8m0=True)
 assert a.is_contiguous() and ascale.is_contiguous() and ascale.dtype==torch.float32
 partials=torch.empty((splits,m,n),device=x.device,dtype=torch.float32);out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
 _prequantized_decode_gemm_kernel[(n//32,splits)](a,w,ascale,s,partials,m,n,k,splits,num_warps=4,num_stages=1)
 _reduce_block_fp8_split_k[(triton.cdiv(m*n,256),)](partials,out,m*n,splits,256)
 return out


specs=[('qa',['layers.0.attn.wq_a'],16),('kv',['layers.0.attn.wkv'],16),('qb_h64',['layers.0.attn.wq_b'],4),('ob_h64',['layers.0.attn.wo_b'],4),('shared_gate_up',['layers.0.ffn.shared_experts.w1','layers.0.ffn.shared_experts.w3'],8),('shared_down',['layers.0.ffn.shared_experts.w2'],8)]
with torch.inference_mode():
 for name,prefixes,split in specs:
  parts=[load(p) for p in prefixes];wc=torch.cat([p[0] for p in parts],0).contiguous() if len(parts)>1 else parts[0][0];sc=torch.cat([p[1] for p in parts],0).contiguous() if len(parts)>1 else parts[0][1]
  w=wc.to('musa');s=sc.to('musa');carrier=(w.float()*s.repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16();ref=native.NativeV41Dense(carrier,True);n,k=w.shape
  for m in [4,6]:
   actual_split=4 if name=='shared_gate_up' and m==6 else split;x=torch.randn(m,k,device='musa',dtype=torch.bfloat16);stream=torch.musa.Stream();stream.wait_stream(torch.musa.current_stream())
   with torch.musa.stream(stream):
    for _ in range(3):prequant(x,w,s,actual_split)
    stream.synchronize();graph=torch.musa.MUSAGraph()
    with torch.musa.graph(graph,stream=stream):out=prequant(x,w,s,actual_split)
    for amplitude in [1.,1e-6,.3]:
     changed=torch.randn_like(x)*amplitude;x.copy_(changed);graph.replay();reference=ref.forward(x);stream.synchronize();er=error(reference,out);assert er<.01,(name,m,amplitude,er);records.append({'name':name,'rows':m,'amplitude':amplitude,'rms':er,'nondefault_stream':True,'input_address':x.data_ptr(),'output_address':out.data_ptr()});(r/'partial.json').write_text(json.dumps(records,indent=2))
   if name in ['qb_h64','ob_h64']:
    a=sg_prequant(x,w,s,actual_split);b=prequant(x,w,s,actual_split);torch.musa.synchronize();er=error(a,b);assert er<.001;comparisons.append({'name':name,'rows':m,'rms_vs_sg':er,'sg':graph_time(lambda:sg_prequant(x,w,s,actual_split)),'native':graph_time(lambda:prequant(x,w,s,actual_split)),'sg_reverse':graph_time(lambda:sg_prequant(x,w,s,actual_split))})
   del graph,x,out,reference,changed
  del ref,carrier,w,s,wc,sc,parts;gc.collect();torch.musa.empty_cache()
(r/'result.json').write_text(json.dumps({'status':'PASS','checks':records,'same_process_sg_comparison':comparisons,'scope':'Actual real-weight graph capture and replay after three different input values, not model integration'},indent=2));print('PASS',len(records))
