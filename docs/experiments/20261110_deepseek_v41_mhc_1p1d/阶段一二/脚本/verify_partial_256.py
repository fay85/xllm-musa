from pathlib import Path
import argparse,math,json,sqlite3,pickle,io,base64,re,statistics,datetime,hashlib
parser=argparse.ArgumentParser();parser.add_argument('--run',default='stage1_ref');parser.add_argument('--benchmark',default='stage1_ref_v2');a=parser.parse_args()
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops');run=root/'03_benchmarks'/a.benchmark;folder=run/'input256_output256/measured/deepseek-v4.1-flash-1017/parallel_1_number_5'
p=(root/'runs'/a.run/'sj9/rank0.log').read_text().splitlines();d=(root/'runs'/a.run/'sj10/rank0.log').read_text().splitlines()
class DataOnly(pickle.Unpickler):
 def find_class(self,*args):raise pickle.UnpicklingError('No object loading')
with sqlite3.connect('file:'+str(folder/'benchmark_data.db')+'?mode=ro',uri=True) as c:
 c.row_factory=sqlite3.Row;rows=[dict(r) for r in c.execute('select rowid,* from result order by rowid')]
assert len(rows)==5
checks=[]
for r in rows:
 assert r['success'] and r['prompt_tokens']==256 and r['completion_tokens']==256
 responses=DataOnly(io.BytesIO(base64.b64decode(r['response_messages']))).load();ids={x['id'] for x in responses if isinstance(x,dict) and x.get('id')};assert len(ids)==1;rid=next(iter(ids));key='request_id: '+rid+','
 pl=[x for x in p if 'record_type: pd_prefill_stage,' in x and key in x];dl=[x for x in d if 'record_type: sequence_terminal,' in x and key in x];ok=[x for x in d if 'record_type: request_terminal,' in x and key in x and 'outcome: success, status_code: 0' in x];assert len(pl)==len(dl)==len(ok)==1
 fields=dict(re.findall(r'([a-z_]+): ([^,]+)',dl[0]));assert fields['finish_reason']=='length' and int(fields['prompt_tokens'])==int(fields['generated_tokens'])==256 and int(fields['num_prefix_cache_tokens'])==0
 pf=dict(re.findall(r'([a-z_]+): ([^,]+)',pl[0]));assert int(pf['prompt_tokens'])==256 and int(pf['generated_tokens'])==1
 assert math.isclose(r['time_per_output_token'],(r['latency']-r['first_chunk_latency'])/255,rel_tol=1e-6)
 checks.append({'request_id':rid,'prefill':pl[0],'decode':dl[0],'success':ok[0]})
tool=json.loads((folder/'benchmark_summary.json').read_text());duration=tool['Test Duration (s)'];record={'status':'PASS_PARTIAL_256_GROUP_ONLY','at':datetime.datetime.now().astimezone().isoformat(),'configuration':a.run,'warmup_excluded':True,'repeats':5,'input_tokens':256,'output_tokens':256,'ttft_ms':statistics.mean(r['first_chunk_latency'] for r in rows)*1000,'tpot_ms':statistics.mean(r['time_per_output_token'] for r in rows)*1000,'latency_s':statistics.mean(r['latency'] for r in rows),'qps':5/duration,'output_tokens_s':1280/duration,'checks':checks,'database_sha256':hashlib.sha256((folder/'benchmark_data.db').read_bytes()).hexdigest()}
record['tpot_sample_stddev_ms']=statistics.stdev(r['time_per_output_token']*1000 for r in rows)
(run/'partial_256_verified.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='checks'},ensure_ascii=False,indent=2))
