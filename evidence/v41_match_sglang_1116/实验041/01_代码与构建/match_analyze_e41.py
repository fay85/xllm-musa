from pathlib import Path
import json,sqlite3,base64,pickle,io,hashlib,re,statistics
b=Path('王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它/02_实验记录')
class DataOnly(pickle.Unpickler):
 def find_class(self,module,name):raise pickle.UnpicklingError('No global objects accepted')
result={}
for name,raw,proposal in [('control',b/'实验_039/原始证据_on',3),('candidate',b/'实验_041/原始证据_candidate',5),('reverse',b/'实验_041/原始证据_reverse',3)]:
 if not raw.exists():continue
 db=next((raw/'formal_client/formal').rglob('benchmark_data.db'));con=sqlite3.connect(db);con.row_factory=sqlite3.Row;rows=[dict(x) for x in con.execute('select * from result')];assert len(rows)==5
 server=(raw/'roles/sj10/rank0.log').read_text(errors='replace');parsed=[]
 for row in rows:
  assert row['success']==1 and row['prompt_tokens']==256 and row['completion_tokens']==256
  chunks=DataOnly(io.BytesIO(base64.b64decode(row['response_messages']))).load();assert isinstance(chunks,list);content='';usage=None;rid=None;finish=None
  for chunk in chunks:
   rid=chunk.get('id') or rid;usage=chunk.get('usage') or usage
   for ch in chunk.get('choices',[]):content+=ch.get('delta',{}).get('content') or '';finish=ch.get('finish_reason') or finish
  assert finish=='length'
  if usage:assert usage['prompt_tokens']==256 and usage['completion_tokens']==256
  terminal=[x for x in server.splitlines() if 'record_type: sequence_terminal' in x and 'request_id: '+rid+',' in x];assert len(terminal)==1
  assert 'prompt_tokens: 256, generated_tokens: 256' in terminal[0] and 'finish_reason: length' in terminal[0]
  m=re.search(r'speculative_accepted_tokens: (\d+), speculative_proposed_tokens: (\d+)',terminal[0]);assert m;a,p=map(int,m.groups());assert p%proposal==0
  parsed.append({'request_id':rid,'success':True,'input':256,'output':256,'finish_reason':finish,'latency_s':row['latency'],'TTFT_ms':row['first_chunk_latency']*1000,'TPOT_ms':(row['latency']-row['first_chunk_latency'])/255*1000,'accepted':a,'proposed':p,'proposal_rounds':p//proposal,'content_sha256':hashlib.sha256(content.encode()).hexdigest(),'content':content,'usage':usage})
 summary=json.loads((raw/'formal_client/summary.json').read_text());assert abs(statistics.mean(x['TPOT_ms'] for x in parsed)-summary['mean_TPOT_ms_per_token'])<1e-9
 out={'summary':summary,'requests':parsed,'proposal':proposal,'basic':json.loads((raw/'formal_client/basic_gate.json').read_text()),'tool_summary':json.loads(next((raw/'formal_client/formal').rglob('benchmark_summary.json')).read_text()),'scope_notes':{'distinct_outputs_within_formal':len({x['content_sha256'] for x in parsed}),'decode_wall_ms_per_proposal_round':statistics.mean((x['latency_s']-x['TTFT_ms']/1000)*1000/x['proposal_rounds'] for x in parsed),'round_time_scope':'draft+verify+cache+host+streaming, changing generated tokens; not isolated GPU time'}}
 if (raw/'trace_analysis/all_rank_validation.json').exists():
  trace=json.loads((raw/'trace_analysis/all_rank_validation.json').read_text());d0=next(x for x in trace['ranks'] if x['host']=='sj10' and x['pid']==json.loads((raw/'roles/sj10/status.json').read_text())['pids'][0]);out.update(trace_validation=trace['status'],D_rank0_last_graph=d0['graph_replays'][-1],D_rank0_last_step=d0['steps'][-1],profiler_requests=json.loads((raw/'trace/requests.json').read_text()))
 if (raw/'long_smoke/result.json').exists():out['long_smoke']=json.loads((raw/'long_smoke/result.json').read_text());out['post_long']=json.loads((raw/'post_long_checks/results.json').read_text())
 result[name]=out
result['differences']={}
for ctrl in ['control','reverse']:
 if ctrl not in result:continue
 a=result[ctrl]['summary'];z=result['candidate']['summary'];result['differences'][ctrl]={'TPOT_reduction_pct':100*(1-z['mean_TPOT_ms_per_token']/a['mean_TPOT_ms_per_token']),'output_increase_pct':100*(z['output_tokens_per_latency_sum_second']/a['output_tokens_per_latency_sum_second']-1),'E2E_reduction_pct':100*(1-z['mean_E2E_s']/a['mean_E2E_s']),'same_output':result[ctrl]['requests'][0]['content_sha256']==result['candidate']['requests'][0]['content_sha256']}
(b/'实验_041/03_逐请求与trace分析.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({n:{'summary':v['summary'],'rounds':v['requests'][0]['proposal_rounds'],'accept_proposed':[v['requests'][0]['accepted'],v['requests'][0]['proposed']],'notes':v['scope_notes'],'graph':{k:v.get('D_rank0_last_graph',{}).get(k) for k in ['kernels','span_us']}} for n,v in result.items() if n!='differences'}));print(json.dumps(result['differences']))
