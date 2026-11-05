"""Validate native usage and extract the two measured EvalScope runs."""
from pathlib import Path
import argparse,base64,csv,hashlib,io,json,math,pickle,re,sqlite3,statistics

class PlainDataUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  raise pickle.UnpicklingError('Only plain data from our benchmark is accepted')

parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
assert (args.run/'client_runs_finished.json').exists(),'Client runs have not finished'
# EvalScope 1.14.1 stores usage counters but excludes choices=[] usage packets
# from response_messages. Cross-check the native request ID against both roles.
server_root=args.run.parent
pd_evidence=json.loads((server_root/'basic_pd_path_evidence.json').read_text())
assert pd_evidence['status']=='PASS_BASIC_PD_PATH'
prefill_log=server_root/pd_evidence['prefill']/'rank0.log'
decode_log=server_root/pd_evidence['decode']/'rank0.log'
prefill_lines=prefill_log.read_text().splitlines();decode_lines=decode_log.read_text().splitlines()
server_provenance=[{'path':str(p.relative_to(server_root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [prefill_log,decode_log]]
def native_server_usage(response):
 ids={item['id'] for item in response if isinstance(item,dict) and item.get('id')}
 assert len(ids)==1,f'Expected a single native request ID: {ids}'
 request_id=next(iter(ids));key='request_id: '+request_id+','
 terminal=[line for line in decode_lines if 'record_type: sequence_terminal,' in line and key in line]
 success=[line for line in decode_lines if 'record_type: request_terminal,' in line and key in line and 'outcome: success, status_code: 0' in line]
 prefill=[line for line in prefill_lines if 'record_type: pd_prefill_stage,' in line and key in line]
 assert len(terminal)==len(success)==len(prefill)==1,f'Incomplete native PD records for {request_id}'
 fields=dict(re.findall(r'([a-z_]+): ([^,]+)',terminal[0]));pfields=dict(re.findall(r'([a-z_]+): ([^,]+)',prefill[0]))
 assert fields['finish_reason']=='length' and int(fields['num_prefix_cache_tokens'])==0
 assert int(pfields['prompt_tokens'])==int(fields['prompt_tokens']) and int(pfields['generated_tokens'])==1
 usage={'prompt_tokens':int(fields['prompt_tokens']),'completion_tokens':int(fields['generated_tokens']),'prompt_tokens_details':{'cached_tokens':0}}
 return usage,{'request_id':request_id,'prefill_line':prefill[0],'decode_line':terminal[0],'terminal_line':success[0]}
results=[];errors=[];all_rows=[];provenance=[]
for length in [256,2048]:
 directory=args.run/f'input{length}_output256'/'measured';databases=list(directory.rglob('benchmark_data.db'))
 assert len(databases)==1,f'Expected one measured database, found {databases}'
 database=databases[0];provenance.append({'path':str(database.relative_to(args.run)),'sha256':hashlib.sha256(database.read_bytes()).hexdigest()})
 with sqlite3.connect('file:'+str(database.resolve())+'?mode=ro',uri=True) as db:
  db.row_factory=sqlite3.Row;records=[dict(row) for row in db.execute('SELECT rowid, * FROM result ORDER BY rowid')]
 if len(records)!=5:errors.append(f'{length}/256 has {len(records)} requests, expected 5')
 rows=[]
 with (args.run/f'raw_measured_{length}_256.jsonl').open('x') as output:
  for record in records:
   response=PlainDataUnpickler(io.BytesIO(base64.b64decode(record['response_messages']))).load()
   # Native service logs are authoritative when the official client omits the
   # standalone SSE usage packet from its response_messages archive.
   usages=[item['usage'] for item in response if isinstance(item,dict) and isinstance(item.get('usage'),dict)]
   problems=[];native=None;server_record=None
   try:native,server_record=native_server_usage(response)
   except (AssertionError,KeyError,ValueError) as error:problems.append('native_server_evidence: '+str(error))
   if usages and native and any(usages[-1].get(k)!=native[k] for k in ['prompt_tokens','completion_tokens']):problems.append('sse_usage_server_mismatch')
   if not record['success']:problems.append('request_failed')
   if not record['is_stream']:problems.append('not_streaming')
   if not native or native.get('prompt_tokens')!=length or native.get('completion_tokens')!=256:problems.append('native_usage_missing_or_wrong_length')
   if record['prompt_tokens']!=length or record['completion_tokens']!=256:problems.append('tool_token_count_mismatch')
   latency=record['latency'];ttft=record['first_chunk_latency'];tpot=record['time_per_output_token']
   if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in [latency,ttft,tpot]) or not 0<ttft<latency or not tpot>0:problems.append('invalid_timing')
   if not problems and not math.isclose(tpot,(latency-ttft)/255,rel_tol=1e-6,abs_tol=1e-8):problems.append('tpot_formula_mismatch')
   if problems:errors.append({'shape':f'{length}/256','row':record['rowid'],'problems':problems})
   raw=dict(record);raw['response_messages']=response;raw['request']=json.loads(record['request']);raw['native_usage']=native;raw['native_usage_source']='matched_native_server_PD_logs';raw['native_server_record']=server_record;raw['validation_errors']=problems
   output.write(json.dumps(raw,ensure_ascii=False)+'\n')
   if not problems:
    row={'input_tokens':length,'output_tokens':256,'repeat':record['rowid'],'ttft_s':ttft,'tpot_ms':tpot*1000,'latency_s':latency,'end_to_end_output_tokens_s':256/latency,'decode_tokens_s':1/tpot}
    rows.append(row);all_rows.append(row)
 if len(rows)==5:
  summary={'input_tokens':length,'output_tokens':256,'repeats':5,'concurrency':1,'mode':'1P1D_TP8_EP8_eager','metrics':{}}
  for metric in ['ttft_s','tpot_ms','latency_s','end_to_end_output_tokens_s','decode_tokens_s']:
   values=[r[metric] for r in rows];summary['metrics'][metric]={'mean':statistics.mean(values),'median':statistics.median(values),'min':min(values),'max':max(values),'sample_stddev':statistics.stdev(values)}
  summary['output_tokens_over_sum_request_latency']=1280/sum(r['latency_s'] for r in rows)
  results.append(summary)
record={'status':'PASS' if not errors and len(results)==2 else 'FAIL','results':results,'errors':errors,'database_provenance':provenance,'server_log_provenance':server_provenance,'native_usage_verification':'Native request ID matched between official response chunks, P prefill record, D successful terminal and D sequence token counts. EvalScope omits standalone usage packets from response_messages.','definitions':{'ttft':'request start to first generated content chunk','tpot':'(request latency - TTFT)/(native completion tokens - 1), official EvalScope formula','end_to_end_output_tokens_s':'native completion tokens / full request latency','decode_tokens_s':'1 / TPOT; not end-to-end rate'},'scope':'Two elementary questions and ten performance requests do not establish general accuracy; inspect PD server evidence separately.'}
(args.run/'validated_summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
if all_rows:
 with (args.run/'measured_per_request.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
lines=['# 两机 1P1D eager 初步性能','',f'验收状态：**{record["status"]}**。各组并发 1、预热 1 次不计入统计、正式 5 次。','', '| 实际输入/输出 token | 平均 TTFT(s) | 平均 TPOT(ms) | 平均端到端(s) | 平均端到端输出 token/s |', '|---|---:|---:|---:|---:|']
for r in results:
 m=r['metrics'];lines.append(f'| {r["input_tokens"]}/256 | {m["ttft_s"]["mean"]:.4f} | {m["tpot_ms"]["mean"]:.4f} | {m["latency_s"]["mean"]:.4f} | {m["end_to_end_output_tokens_s"]["mean"]:.4f} |')
lines+=['','TPOT 采用官方 EvalScope 的 `(总时延−TTFT)/(输出 token−1)`，包含请求尾部开销；端到端输出 token/s 采用 `输出 token/总时延`，二者不能互换。','', '官方 EvalScope 会把 usage 计数写入数据库字段，但不把 choices=[] 的独立 usage 包保存到 response_messages。本报告按原生 request_id 与 P prefill、D 成功终止及 D sequence 计数逐条交叉核对，不仅依赖客户端 token 估算。原始 SQLite、工具保留的请求/响应 JSONL、匹配的服务端日志、逐次 CSV、标准差及最小最大值均保留。完整模型 P/D 路由、八卡拓扑与传输证据须另行核对；此客户端验收不替代服务端验收。']
if errors:lines+=['','存在错误，不能将上述部分结果称为完成性能验收：','',json.dumps(errors,ensure_ascii=False,indent=2)]
(args.run/'简要性能结果.md').write_text('\n'.join(lines)+'\n');print(json.dumps(record,ensure_ascii=False));raise SystemExit(0 if record['status']=='PASS' else 1)
