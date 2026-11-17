#!/usr/bin/env python3
"""Frozen official EvalScope protocol; short baseline plus one bounded long smoke."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,sqlite3,subprocess,time,statistics
import requests
from tokenizers import Tokenizer

def now(): return datetime.datetime.now().astimezone().isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();r=Path('/expt');run=r/'runs'/args.run;o=run/'formal_client';o.mkdir(exist_ok=False)
 url='http://10.121.32.9:31180';model='deepseek-v4.1-flash-1017-w4afp8-mt';binary=Path(json.loads((run/'sj10/role_config.json').read_text())['binary']);tokenizer=Path('/work/artifacts/model_view_w4afp8_mt_1017');inputs=Path('/work/artifacts/benchmark_inputs_v1');evalscope=Path('/work/artifacts/bench_venv_v2/bin/evalscope')
 (o/'job.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'started_at':now(),'run':args.run},indent=2))
 rc=1
 try:
  assert sha(inputs/'input256_output256.jsonl')=='c3d7cb38060a2f81b228b83d8939e1a682480e9ab26a2c5a2df29ca66dadbc51'
  assert sha(tokenizer/'tokenizer.json')=='8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf'
  tok=Tokenizer.from_file(str(tokenizer/'tokenizer.json'));identity={'source_commit':'17d19adc + E48 runtime diagnostic; E49 same ELF, D full draft Graph only','candidate_delta_manifest':'/expt/builds/e46_native_full_v4/source_manifest.json','binary_sha256':sha(binary),'tokenizer_sha256':sha(tokenizer/'tokenizer.json'),'profiler_on':False,'requests':{},'accuracy_status':'仅通过基础功能检查，未完成模型精度验收'}
  for length in [256,2048]:
   bodies=[json.loads(x) for x in (inputs/f'input{length}_output256.jsonl').read_text().splitlines()];rows=[]
   for b in bodies:
    assert b['max_tokens']==256 and b['ignore_eos'] and b['temperature']==0 and b['top_p']==1 and b['reasoning_effort']=='none' and b['chat_template_kwargs']=={'thinking_mode':'chat'}
    ids=tok.encode('<｜begin▁of▁sentence｜><｜User｜>'+b['messages'][0]['content']+'<｜Assistant｜></think>',add_special_tokens=False).ids;assert len(ids)==length;rows.append({'body':b,'expected_token_ids':ids})
   identity['requests'][str(length)]={'dataset_sha256':sha(inputs/f'input{length}_output256.jsonl'),'rows':rows}
  (o/'identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2))
  ready=False
  for _ in range(900):
   for host in ['sj9','sj10']:
    state=run/host/'status.json';done=run/host/'result.json'
    if done.exists():raise RuntimeError('Role terminal during readiness: '+host)
    if state.exists() and any(x is not None for x in json.loads(state.read_text()).get('exit_codes',[])):raise RuntimeError('Rank exited during readiness: '+host)
   log=run/'sj10/rank0.log'
   if log.exists() and 'ensure_link ready' in log.read_text(errors='replace'):
    try:
     if requests.get(url+'/readyz',timeout=2).ok and requests.get(url+'/v1/models',timeout=2).ok:ready=True;break
    except requests.RequestException:pass
   time.sleep(1)
  assert ready,'Readiness deadline reached; inspect role jobs before any retry'
  gate=[]
  for index,(question,expected) in enumerate([('只回答计算结果：1.3＋2.9 等于多少？','4.2'),('只回答计算结果：7×8 等于多少？','56')],1):
   b=dict(identity['requests']['256']['rows'][0]['body']);b.update(messages=[{'role':'user','content':question}],max_tokens=64,ignore_eos=False,stream=True,stream_options={'include_usage':True});records=[];content='';usage=None;t=time.monotonic();first_any=None;first_text=None;finish=None
   with requests.post(url+'/v1/chat/completions',json=b,stream=True,timeout=(10,120)) as response:
    response.raise_for_status()
    for line in response.iter_lines():
     if not line:continue
     elapsed=time.monotonic()-t;raw=line.decode();records.append({'elapsed_s':elapsed,'raw':raw})
     if not raw.startswith('data:') or raw[5:].strip()=='[DONE]':continue
     data=json.loads(raw[5:]);first_any=elapsed if first_any is None else first_any
     if data.get('usage'):usage=data['usage']
     for choice in data.get('choices',[]):
      delta=choice.get('delta',{}).get('content') or '';content+=delta
      if delta and first_text is None:first_text=elapsed
      if choice.get('finish_reason'):finish=choice['finish_reason']
   passed=bool(re.search(r'(?<![\d.])'+re.escape(expected)+r'(?![\d.])',content));row={'question':question,'expected':expected,'answer':content,'passed':passed,'usage':usage,'first_SSE_json_s':first_any,'first_content_s':first_text,'finish_reason':finish,'request':b};gate.append(row)
   (o/f'basic_{index}_sse.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records));(o/'basic_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2));assert passed,'Basic arithmetic check failed'
  env=dict(os.environ,LD_PRELOAD='',PYTHONPATH='',MUSA_VISIBLE_DEVICES='-1',TOKENIZERS_PARALLELISM='false');runs=[]
  for phase,length,count in [('screen_warmup',256,1),('screen',256,2),('formal_warmup',256,1),('formal',256,5)]:
   dest=o/phase;cmd=[str(evalscope),'perf','--api','openai','--model',model,'--url',url+'/v1/chat/completions','--dataset','line_by_line','--dataset-path',str(inputs/f'input{length}_output256.jsonl'),'--number',str(count),'--parallel','1','--num-workers','1','--db-commit-interval','1','--log-every-n-query','1','--warmup-num','0','--tokenizer-path',str(tokenizer),'--stream','--outputs-dir',str(dest),'--no-timestamp','--no-test-connection','--connect-timeout','20','--read-timeout',('180' if phase=='long_smoke' else '1800'),'--total-timeout',('300' if phase=='long_smoke' else '3600')]
   item={'phase':phase,'count':count,'input_tokens':length,'command':cmd,'started_at':now()};(o/'active.json').write_text(json.dumps(item,indent=2));t=time.monotonic()
   with (o/(phase+'.log')).open('xb') as log:p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   item.update(exit_code=p.returncode,finished_at=now(),client_wall_seconds=time.monotonic()-t);runs.append(item);(o/'status.json').write_text(json.dumps(runs,indent=2));assert p.returncode==0,'EvalScope process failed'
   dbs=list(dest.rglob('benchmark_data.db'));assert len(dbs)==1
   with sqlite3.connect('file:'+str(dbs[0])+'?mode=ro',uri=True) as db:
    db.row_factory=sqlite3.Row;rows=[dict(x) for x in db.execute('SELECT success,prompt_tokens,completion_tokens,latency,first_chunk_latency FROM result')]
   item['rows']=rows;(o/'status.json').write_text(json.dumps(runs,indent=2));assert len(rows)==count and all(x['success'] and x['prompt_tokens']==length and x['completion_tokens']==256 and x['latency']>x['first_chunk_latency']>0 for x in rows),'Invalid request retained; entire round cannot pass'
   if phase=='formal':
    latency=sum(x['latency'] for x in rows);tpots=[(x['latency']-x['first_chunk_latency'])/255*1000 for x in rows];summary={'status':'SHORT_BASELINE_VALID_LONG_SMOKE_PENDING','accuracy_status':identity['accuracy_status'],'mean_TTFT_ms':statistics.mean(x['first_chunk_latency'] for x in rows)*1000,'mean_TPOT_ms_per_token':statistics.mean(tpots),'TPOT_sample_stdev_ms':statistics.stdev(tpots),'mean_E2E_s':latency/count,'output_tokens_per_latency_sum_second':count*256/latency,'formal_requests':count,'input_output':[256,256],'profiler_on':False,'binary_sha256':identity['binary_sha256']};(o/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
  summary['status']='FORMAL_SHORT_PASSED_NOT_FINAL_THREE_ROUND';(o/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));rc=0
 except Exception as error:
  (o/'error.txt').write_text(repr(error));raise
 finally:
  (o/'finished.json').write_text(json.dumps({'at':now(),'exit_code':rc},indent=2))
if __name__=='__main__':main()

