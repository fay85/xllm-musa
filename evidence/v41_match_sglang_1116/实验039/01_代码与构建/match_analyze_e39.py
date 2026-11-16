from pathlib import Path
import json,sqlite3,base64,pickle,io,hashlib,re,statistics,collections
b=Path('王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它/02_实验记录/实验_039')
class DataOnly(pickle.Unpickler):
 def find_class(self,module,name):raise pickle.UnpicklingError('No global objects accepted')
result={}
for mode in ['off','on']:
 raw=b/('原始证据_'+mode);db=next((raw/'formal_client/formal').rglob('benchmark_data.db'));con=sqlite3.connect(db);con.row_factory=sqlite3.Row;rows=[dict(x) for x in con.execute('select * from result')];assert len(rows)==5
 server=(raw/'roles/sj10/rank0.log').read_text(errors='replace');parsed=[]
 for row in rows:
  assert row['success']==1 and row['prompt_tokens']==256 and row['completion_tokens']==256
  chunks=DataOnly(io.BytesIO(base64.b64decode(row['response_messages']))).load();assert isinstance(chunks,list)
  content='';usage=None;rid=None;finish=None
  for chunk in chunks:
   rid=chunk.get('id') or rid;usage=chunk.get('usage') or usage
   for ch in chunk.get('choices',[]):content+=ch.get('delta',{}).get('content') or '';finish=ch.get('finish_reason') or finish
  assert finish=='length'
  if usage:assert usage['prompt_tokens']==256 and usage['completion_tokens']==256
  terminal=[x for x in server.splitlines() if 'record_type: sequence_terminal' in x and 'request_id: '+rid+',' in x];assert len(terminal)==1
  assert 'prompt_tokens: 256, generated_tokens: 256' in terminal[0] and 'finish_reason: length' in terminal[0]
  m=re.search(r'speculative_accepted_tokens: (\d+), speculative_proposed_tokens: (\d+)',terminal[0]);assert m;a,p=map(int,m.groups());assert p%3==0
  parsed.append({'request_id':rid,'success':True,'input':256,'output':256,'finish_reason':finish,'latency_s':row['latency'],'TTFT_ms':row['first_chunk_latency']*1000,'TPOT_ms':(row['latency']-row['first_chunk_latency'])/255*1000,'accepted':a,'proposed':p,'proposal3_rounds':p//3,'content_sha256':hashlib.sha256(content.encode()).hexdigest(),'content':content,'usage':usage})
 trace=json.loads((raw/'trace_analysis/all_rank_validation.json').read_text());summary=json.loads((raw/'formal_client/summary.json').read_text());expected=statistics.mean(x['TPOT_ms'] for x in parsed);assert abs(expected-summary['mean_TPOT_ms_per_token'])<1e-9
 d0=next(x for x in trace['ranks'] if x['host']=='sj10' and x['pid']==json.loads((raw/'roles/sj10/status.json').read_text())['pids'][0]);last=d0['graph_replays'][-1]
 result[mode]={'summary':summary,'requests':parsed,'trace_validation':trace['status'],'D_rank0_last_graph':last,'D_rank0_last_step':d0['steps'][-1],'D_rank0_aten_total':d0['aten_calls'],'profiler_requests':json.loads((raw/'trace/requests.json').read_text()),'basic':json.loads((raw/'formal_client/basic_gate.json').read_text()),'tool_summary':json.loads(next((raw/'formal_client/formal').rglob('benchmark_summary.json')).read_text()),'native_aten_last':json.loads((b/('04_trace/静态激活_'+mode.upper())/'actual_native_last.receipt.json').read_text()) if (b/('04_trace/静态激活_'+mode.upper())/'actual_native_last.receipt.json').exists() else None}
 if mode=='on':result[mode]['long_smoke']=json.loads((raw/'long_smoke/result.json').read_text());result[mode]['post_long']=json.loads((raw/'post_long_checks/results.json').read_text())
a=result['off']['summary'];z=result['on']['summary'];result['difference']={'TPOT_reduction_pct':100*(1-z['mean_TPOT_ms_per_token']/a['mean_TPOT_ms_per_token']),'output_increase_pct':100*(z['output_tokens_per_latency_sum_second']/a['output_tokens_per_latency_sum_second']-1),'E2E_reduction_pct':100*(1-z['mean_E2E_s']/a['mean_E2E_s']),'TTFT_delta_ms':z['mean_TTFT_ms']-a['mean_TTFT_ms'],'same_output_across_modes':result['off']['requests'][0]['content_sha256']==result['on']['requests'][0]['content_sha256']}
for mode in ['off','on']:
 req=result[mode]['requests'];result[mode]['scope_notes']={'distinct_outputs_within_formal':len(set(x['content_sha256'] for x in req)),'decode_wall_ms_per_proposal_round':statistics.mean((x['latency_s']-x['TTFT_ms']/1000)*1000/x['proposal3_rounds'] for x in req),'round_time_scope':'Includes draft/verify/caches/host/streaming and changing generated tokens; not isolated GPU GEMM'}
(b/'03_逐请求与trace分析.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'difference':result['difference'],'off':{'notes':result['off']['scope_notes'],'nodes':result['off']['D_rank0_last_graph']['kernels'],'span_us':result['off']['D_rank0_last_graph']['span_us']},'on':{'notes':result['on']['scope_notes'],'nodes':result['on']['D_rank0_last_graph']['kernels'],'span_us':result['on']['D_rank0_last_graph']['span_us']}}))
