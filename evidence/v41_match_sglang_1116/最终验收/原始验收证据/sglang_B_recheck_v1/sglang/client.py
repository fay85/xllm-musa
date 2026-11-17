#!/usr/bin/env python3
"""SG A/B: same official EvalScope perf client and request identity as xLLM.
Only CPU client orchestration; does not start/stop any model server/container.
"""
import argparse,datetime,hashlib,json,os,re,sqlite3,subprocess,time
from pathlib import Path
import requests
import runpy
_perf_env=runpy.run_path('/expt/00_control/performance_environment.py')
from tokenizers import Tokenizer

def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['A','B'],required=True);ap.add_argument('--router',required=True);ap.add_argument('--decode',required=True);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True);ap.add_argument('--evalscope',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--runtime-identity',type=Path,required=True);args=ap.parse_args()
 args.out.mkdir(parents=True,exist_ok=False)
 bodies=[json.loads(x) for x in args.inputs.read_text().splitlines()];tok=Tokenizer.from_file(str(args.tokenizer/'tokenizer.json'));requests_identity=[]
 assert sha(args.inputs)=='c3d7cb38060a2f81b228b83d8939e1a682480e9ab26a2c5a2df29ca66dadbc51','Unexpected fixed dataset'
 assert sha(args.tokenizer/'tokenizer.json')=='8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf','Unexpected tokenizer'
 for b in bodies:
  assert b['max_tokens']==256 and b['ignore_eos'] is True and b['temperature']==0 and b['top_p']==1 and b['reasoning_effort']=='none' and b['chat_template_kwargs']=={'thinking_mode':'chat'}
  rendered='<｜begin▁of▁sentence｜><｜User｜>'+b['messages'][0]['content']+'<｜Assistant｜></think>';expected=tok.encode(rendered,add_special_tokens=False).ids;assert len(expected)==256
  query={k:b[k] for k in ['model','messages','reasoning_effort','chat_template_kwargs']}
  r=requests.post(args.decode.rstrip('/')+'/v1/tokenize',json=query,timeout=90);r.raise_for_status();actual=r.json();entry={'request':query,'expected_token_ids':expected,'response':actual,'same_ids':actual.get('tokens')==expected};requests_identity.append(entry)
  (args.out/'tokenization_preflight.json').write_text(json.dumps(requests_identity,ensure_ascii=False,indent=2));assert entry['same_ids'],'SG chat template differs from fixed xLLM token IDs; stop before timed results'
 runtime=json.loads(args.runtime_identity.read_text());identity={'at':now(),'case':args.case,'dataset_sha':sha(args.inputs),'tokenizer_sha':sha(args.tokenizer/'tokenizer.json'),'runtime_identity':runtime,'length':256,'output':256,'concurrency':1,'warmup':1,'measured':5,'accuracy_status':'Only basic function check, no model accuracy acceptance'};(args.out/'identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2))
 readiness=[];ready=False
 for _ in range(120):
  try:
   r=requests.get(args.router.rstrip('/')+'/v1/models',timeout=3);readiness.append({'at':now(),'http_status':r.status_code})
   if r.ok:ready=True;break
  except requests.RequestException as e:readiness.append({'at':now(),'error':repr(e)})
  time.sleep(1)
 (args.out/'router_readiness.json').write_text(json.dumps(readiness,indent=2));assert ready,'Router did not become ready; no model gate/perf started'
 questions=[('只回答计算结果：1.3＋2.9 等于多少？','4.2'),('只回答计算结果：7×8 等于多少？','56')];gate=[]
 for prompt,expected in questions:
  b=dict(bodies[0],messages=[{'role':'user','content':prompt}],max_tokens=64,ignore_eos=False,stream=False);b.pop('stream_options',None)
  r=requests.post(args.router.rstrip('/')+'/v1/chat/completions',json=b,timeout=240);data=r.json();answer=data.get('choices',[{}])[0].get('message',{}).get('content') or '';passed=r.ok and bool(re.search(r'(?<![\d.])'+re.escape(expected)+r'(?![\d.])',answer));gate.append({'prompt':prompt,'expected':expected,'request':b,'http_status':r.status_code,'response':data,'passed':passed});(args.out/'basic_gate.json').write_text(json.dumps({'questions':gate,'passed':len(gate)==2 and all(x['passed'] for x in gate)},ensure_ascii=False,indent=2));assert passed,'Basic gate failure recorded; do not launch formal perf automatically'
 def snapshot(label):
  try:
   z=requests.get(args.decode.rstrip('/')+'/server_info',timeout=30);(args.out/('decode_server_info_'+label+'.json')).write_text(json.dumps({'at':now(),'http_status':z.status_code,'body':z.json()},ensure_ascii=False,indent=2))
  except Exception as e:(args.out/('decode_server_info_'+label+'_error.txt')).write_text(repr(e))
 snapshot('before_warmup')
 env=dict(os.environ,TOKENIZERS_PARALLELISM='false',LD_PRELOAD='',PYTHONPATH='',MUSA_VISIBLE_DEVICES='-1');runs=[]
 for phase,n in [('warmup',1),('measured',5)]:
  _perf_env['wait_before_phase'](args.out.parent,phase,output=args.out)
  out=args.out/phase;cmd=[str(args.evalscope),'perf','--api','openai','--model',bodies[0]['model'],'--url',args.router.rstrip('/')+'/v1/chat/completions','--dataset','line_by_line','--dataset-path',str(args.inputs),'--number',str(n),'--parallel','1','--num-workers','1','--db-commit-interval','1','--log-every-n-query','1','--warmup-num','0','--tokenizer-path',str(args.tokenizer),'--stream','--outputs-dir',str(out),'--no-timestamp','--no-test-connection','--connect-timeout','20','--read-timeout','1800','--total-timeout','3600'];item={'phase':phase,'count':n,'command':cmd,'started_at':now()};(args.out/'active.json').write_text(json.dumps(item,indent=2));t=time.monotonic()
  with (args.out/(phase+'.log')).open('xb') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
  item.update(exit_code=z.returncode,client_wall_seconds=time.monotonic()-t,finished_at=now());runs.append(item);(args.out/'status.json').write_text(json.dumps(runs,indent=2));assert z.returncode==0,'Official EvalScope returned failure'
  dbs=list(out.rglob('benchmark_data.db'));assert len(dbs)==1
  with sqlite3.connect('file:'+str(dbs[0])+'?mode=ro',uri=True) as db:
   db.row_factory=sqlite3.Row;rows=[dict(r) for r in db.execute('SELECT success,prompt_tokens,completion_tokens,latency,first_chunk_latency FROM result')]
  item['rows']=rows;(args.out/'status.json').write_text(json.dumps(runs,indent=2));assert len(rows)==n and all(r['success'] and r['prompt_tokens']==256 and r['completion_tokens']==256 and r['latency']>r['first_chunk_latency']>0 for r in rows),'Failed/short/mis-tokenized result: retain logs, cannot promote'
  snapshot('after_'+phase)
 _perf_env['audit_after_formal'](args.out.parent,output=args.out,phase_name='measured')
 rows=runs[-1]['rows'];lat=sum(r['latency'] for r in rows);ttft=sum(r['first_chunk_latency'] for r in rows)/5;tpot=sum((r['latency']-r['first_chunk_latency'])/(r['completion_tokens']-1) for r in rows)/5;summary={'status':'PASS_BASIC_FUNCTION_AND_EXACT_TOKEN_PERFORMANCE','case':args.case,'accuracy_status':'仅通过基础功能检查，未完成模型精度验收','formal_requests':5,'input_output':[256,256],'mean_TTFT_ms':ttft*1000,'mean_TPOT_ms_per_token':tpot*1000,'mean_E2E_seconds':lat/5,'output_tokens_per_latency_sum_second':1280/lat,'qps_per_latency_sum':5/lat,'official_wall_metrics_location':str(args.out/'measured'),'timing_definition':'EvalScope row latency and first_chunk_latency; TPOT=(latency-first_chunk_latency)/(completion_tokens-1). Official tool wall metrics retained separately, client process wall includes startup.','profiler_on':False,'finished_at':now()};(args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

