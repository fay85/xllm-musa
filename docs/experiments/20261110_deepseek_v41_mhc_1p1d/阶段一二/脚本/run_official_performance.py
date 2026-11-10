"""Run unmodified EvalScope after a same-binary two-question gate."""
import argparse, datetime, hashlib, json, os, subprocess, sqlite3
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--gate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--binary',type=Path,default=Path('/work/artifacts/bin/xllm'));args=parser.parse_args()
root=Path('/work');binary=args.binary;gate=json.loads(args.gate.read_text());assert gate['status']=='PASS' and len(gate['questions'])==2 and all(q['passed'] for q in gate['questions']);assert gate['binary_sha256']==hashlib.sha256(binary.read_bytes()).hexdigest();assert gate['endpoint']==args.url
args.output.mkdir(parents=True,exist_ok=False);(args.output/'basic_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2))
venv=root/'artifacts/bench_venv_v2';inputs=root/'artifacts/benchmark_inputs_v1'
commands=[]
for length in [256,2048]:
 for phase,count in [('warmup',1),('measured',5)]:
  out=args.output/f'input{length}_output256'/phase
  cmd=[str(venv/'bin/evalscope'),'perf','--api','openai','--model','deepseek-v4.1-flash-1017','--url',args.url,'--dataset','line_by_line','--dataset-path',str(inputs/f'input{length}_output256.jsonl'),'--number',str(count),'--parallel','1','--num-workers','1','--db-commit-interval','1','--log-every-n-query','1','--warmup-num','0','--tokenizer-path',str(root/'artifacts/model_view_1017'),'--stream','--outputs-dir',str(out),'--no-timestamp','--no-test-connection','--connect-timeout','20','--read-timeout','1800','--total-timeout','3600']
  commands.append({'input_tokens':length,'requested_output_tokens':256,'phase':phase,'count':count,'command':cmd})
(args.output/'commands.json').write_text(json.dumps(commands,indent=2));results=[]
for item in commands:
 begin=datetime.datetime.now().astimezone().isoformat();(args.output/'active.json').write_text(json.dumps({'started_at':begin,**item},indent=2))
 with (args.output/f"{item['phase']}_input{item['input_tokens']}.log").open('xb') as log:
  result=subprocess.run(item['command'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,TOKENIZERS_PARALLELISM='false',LD_PRELOAD='',PYTHONPATH='',MUSA_VISIBLE_DEVICES='-1'))
 results.append({k:v for k,v in item.items() if k!='command'}|{'started_at':begin,'finished_at':datetime.datetime.now().astimezone().isoformat(),'exit_code':result.returncode});(args.output/'status.json').write_text(json.dumps(results,indent=2))
 if result.returncode:raise SystemExit(result.returncode)
 databases=list((args.output/f"input{item['input_tokens']}_output256"/item['phase']).rglob('benchmark_data.db'))
 assert len(databases)==1,databases
 with sqlite3.connect('file:'+str(databases[0])+'?mode=ro',uri=True) as db:
  rows=db.execute('SELECT success,prompt_tokens,completion_tokens,latency,first_chunk_latency FROM result').fetchall()
 assert len(rows)==item['count'] and all(r[0] and r[1]==item['input_tokens'] and r[2]==256 and r[3]>r[4]>0 for r in rows),f"{item['phase']} failed or incorrect token counts; no further requests submitted: {rows}"
(args.output/'client_runs_finished.json').write_text(json.dumps({'status':'CLIENT_RUNS_FINISHED_REQUIRES_RAW_USAGE_VALIDATION','note':'A zero CLI exit does not establish all requests succeeded. Verify row count, success flags, exact native input/output usage and phase separation before reporting performance.'},indent=2))
