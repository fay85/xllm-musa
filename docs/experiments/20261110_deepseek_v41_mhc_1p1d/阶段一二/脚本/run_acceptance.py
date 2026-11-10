"""Wait for both owned ranks, check two answers, then run the frozen benchmark."""
import argparse,json,time,subprocess,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path('/work');out=root/'runs'/a.run/'acceptance';out.mkdir(exist_ok=False)
def state(phase,**kw):(out/'status.json').write_text(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'phase':phase,**kw},indent=2))
try:
 state('WAITING_FOR_BOTH_MODEL_SERVERS')
 for _ in range(180):
  ready=True
  for h in ('sj9','sj10'):
   r=root/'runs'/a.run/h
   if not (r/'status.json').exists():ready=False;continue
   s=json.loads((r/'status.json').read_text());assert not s['stopping'] and all(x is None for x in s['exit_codes']),f'{h} rank failed'
   ready=ready and 'Application startup complete.' in (r/'rank0.log').read_text()
  if ready:break
  time.sleep(10)
 else:raise RuntimeError('Model readiness timeout after 30 minutes; no requests submitted')
 url='http://10.121.32.9:39080/v1/chat/completions';gate=root/'02_correctness'/a.run
 binary=json.loads((root/f'artifacts/prepared/{a.run}/sj10.json').read_text())['binary']
 state('BASIC_QUESTIONS')
 cmd=['python3','/work/scripts/check_basic_questions.py','--url',url,'--output',str(gate),'--binary',binary]
 with (out/'basic_questions.log').open('wb') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 state('BENCHMARK',prefix_probe='SKIPPED_KNOWN_NATIVE_V2_COMPLETIONS_PROTOCOL_FAILURE',reference='/work/02_correctness/stage1_ref_prefix/result.json')
 cmd=['python3','/work/scripts/run_official_performance.py','--url',url,'--gate',str(gate/'basic_gate.json'),'--binary',binary,'--output',str(root/'03_benchmarks'/a.run)]
 with (out/'benchmark.log').open('wb') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 state('CLIENT_FINISHED_REQUIRES_RAW_VALIDATION')
except BaseException as e:
 state('FAILED',error=str(e));raise
