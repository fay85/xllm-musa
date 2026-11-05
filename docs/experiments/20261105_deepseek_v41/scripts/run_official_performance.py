"""Run unmodified EvalScope after a same-binary two-question gate."""
import argparse, datetime, hashlib, json, os, subprocess
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--gate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
root=Path('/work');binary=root/'artifacts/bin/xllm';gate=json.loads(args.gate.read_text());assert gate['status']=='PASS' and len(gate['questions'])==2 and all(q['passed'] for q in gate['questions']);assert gate['binary_sha256']==hashlib.sha256(binary.read_bytes()).hexdigest();assert gate['endpoint']==args.url
args.output.mkdir(parents=True,exist_ok=False);(args.output/'basic_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2))
venv=root/'artifacts/bench_venv_v2';inputs=root/'artifacts/benchmark_inputs_v1'
commands=[]
for length in [256,2048]:
 for phase,count in [('warmup',1),('measured',5)]:
  out=args.output/f'input{length}_output256'/phase
  cmd=[str(venv/'bin/evalscope'),'perf','--api','openai','--model','deepseek-v4.1-flash-0909','--url',args.url,'--dataset','line_by_line','--dataset-path',str(inputs/f'input{length}_output256.jsonl'),'--number',str(count),'--parallel','1','--num-workers','1','--db-commit-interval','1','--log-every-n-query','1','--warmup-num','0','--tokenizer-path',str(root/'artifacts/model_view_0909'),'--stream','--outputs-dir',str(out),'--no-timestamp','--no-test-connection','--connect-timeout','20','--read-timeout','1800','--total-timeout','3600']
  commands.append({'input_tokens':length,'requested_output_tokens':256,'phase':phase,'count':count,'command':cmd})
(args.output/'commands.json').write_text(json.dumps(commands,indent=2));results=[]
for item in commands:
 begin=datetime.datetime.now().astimezone().isoformat();(args.output/'active.json').write_text(json.dumps({'started_at':begin,**item},indent=2))
 with (args.output/f"{item['phase']}_input{item['input_tokens']}.log").open('xb') as log:
  result=subprocess.run(item['command'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,TOKENIZERS_PARALLELISM='false',LD_PRELOAD='',PYTHONPATH='',MUSA_VISIBLE_DEVICES='-1'))
 results.append({k:v for k,v in item.items() if k!='command'}|{'started_at':begin,'finished_at':datetime.datetime.now().astimezone().isoformat(),'exit_code':result.returncode});(args.output/'status.json').write_text(json.dumps(results,indent=2))
 if result.returncode:raise SystemExit(result.returncode)
(args.output/'client_runs_finished.json').write_text(json.dumps({'status':'CLIENT_RUNS_FINISHED_REQUIRES_RAW_USAGE_VALIDATION','note':'A zero CLI exit does not establish all requests succeeded. Verify row count, success flags, exact native input/output usage and phase separation before reporting performance.'},indent=2))
