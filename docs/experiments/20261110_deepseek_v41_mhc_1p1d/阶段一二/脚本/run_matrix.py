"""Sequential owned-host experiments. No forced process/container termination."""
from pathlib import Path
import sys,json,shlex,time,subprocess,datetime,math,argparse,paramiko
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b.parents[2]/'sj9_12_登陆工具'));import sj_access
root='/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops'
def remote(code,timeout=30):
 with sj_access.ssh_client('sj10') as c:r=sj_access.execute(c,'sj10','python3 -c '+shlex.quote(code),root=True,timeout=timeout)
 if r['exit_status']:raise RuntimeError(r['stderr'] or r['stdout'])
 return r['stdout']
def state(phase,**kw):
 record={'at':datetime.datetime.now().astimezone().isoformat(),'phase':phase,**kw};(b/'证据/matrix_status.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
def read(path):
 for attempt in range(6):
  try:return json.loads(remote(f'from pathlib import Path;import json;p=Path({str(root+"/"+path)!r});print(json.dumps(p.read_text() if p.exists() else None))'))
  except (TimeoutError,EOFError,OSError,paramiko.SSHException) as error:
   if isinstance(error,paramiko.AuthenticationException) and 'timeout' not in str(error).lower():raise
   if attempt==5:raise
   print(json.dumps({'event':'READ_ONLY_SSH_RETRY','path':path,'attempt':attempt+1,'error':repr(error)}),flush=True);time.sleep(10)
def wait_finished(run,bench):
 state('WAIT_BENCHMARK',run=run,benchmark=bench)
 for _ in range(720):
  text=read(f'03_benchmarks/{bench}/client_runs_finished.json')
  if text is not None:return
  exitpath=f'runs/{run}/benchmark_v2.exit' if bench=='stage1_ref_v2' else f'runs/{run}/acceptance_supervisor.exit'
  status=read(exitpath)
  if status is not None:
   # The client can finish between these two independent read-only queries.
   # Require its completion marker; a zero CLI exit alone is insufficient.
   if read(f'03_benchmarks/{bench}/client_runs_finished.json') is not None:return
   raise RuntimeError(f'{run} exited before completed benchmark: {status}')
  time.sleep(20)
 raise RuntimeError(f'{run} benchmark timed out after four hours; leaving evidence and no forced stop')
def validate(run,bench):
 state('VALIDATE',run=run)
 shell=shlex.join(['python3','/work/scripts/summarize_performance.py','--run',f'/work/03_benchmarks/{bench}','--server-run',f'/work/runs/{run}'])
 cmd=['docker','exec','-e','MUSA_VISIBLE_DEVICES=-1','xiaopang1110_v41_decode_sj10','bash','-c',shell+f' > /work/03_benchmarks/{bench}/validation.log 2>&1']
 remote('import subprocess;subprocess.run('+repr(cmd)+',check=True)',timeout=40)
 data=json.loads(read(f'03_benchmarks/{bench}/validated_summary.json'));assert data['status']=='PASS';(b/'证据'/f'{run}_validated_summary.json').write_text(json.dumps(data,indent=2));return data

def stop(run):
 state('NORMAL_TERM',run=run)
 remote(f'from pathlib import Path;p=Path({root!r})/"runs"/{run!r};[(p/h/"stop.request").write_text("Benchmark and validation finished; normal TERM only\\n") for h in ("sj9","sj10")]')
 for _ in range(150):
  values=[read(f'runs/{run}/{h}/result.json') for h in ('sj9','sj10')]
  if all(x is not None for x in values):
   state('RANKS_EXITED',run=run,result=[json.loads(x) for x in values]);return
  time.sleep(2)
 raise RuntimeError(f'{run} ranks did not exit after five minutes. No SIGKILL, no reboot. Inspect before proceeding.')
def launch(run):
 state('LAUNCH',run=run)
 subprocess.run([sys.executable,str(b/'脚本/start_sj_roles.py'),'--run',run],check=True)
 shell=f'python3 /work/scripts/run_acceptance.py --run {run} > /work/runs/{run}/acceptance_supervisor.log 2>&1; echo $? > /work/runs/{run}/acceptance_supervisor.exit'
 cmd=['docker','exec','-d','-e','MUSA_VISIBLE_DEVICES=-1','xiaopang1110_v41_decode_sj10','bash','-c',shell]
 remote('import subprocess;subprocess.run('+repr(cmd)+',check=True)')
def probe_chat_after_performance(run):
 # Same frozen engine, only after all benchmark requests and validation ended.
 # A failed protocol control is not a model numerical result.
 for previous in (b/'证据').glob('stage2_*_chat_prefix_result.json'):
  prior=json.loads(previous.read_text())
  if prior['status']=='PROBE_FAILED':
   state('CHAT_PREFIX_SKIPPED_KNOWN_PROTOCOL_FAILURE',run=run,reference=previous.name);return
 state('CHAT_PREFIX_POST_BENCHMARK',run=run)
 out=f'/work/02_correctness/{run}_chat_prefix'
 shell=f'python3 /work/scripts/check_chat_prefix_parity.py --output {out} > /work/runs/{run}/chat_prefix.log 2>&1; echo $? > /work/runs/{run}/chat_prefix.exit'
 cmd=['docker','exec','-d','-e','MUSA_VISIBLE_DEVICES=-1','xiaopang1110_v41_decode_sj10','bash','-c',shell]
 remote('import subprocess;subprocess.run('+repr(cmd)+',check=True)')
 for _ in range(300):
  if read(f'runs/{run}/chat_prefix.exit') is not None:break
  time.sleep(2)
 else:raise RuntimeError('Chat prefix probe has not finished; do not stop model under an active request')
 value=read(f'02_correctness/{run}_chat_prefix/result.json')
 if value is None:raise RuntimeError('Chat probe exited without result; inspect saved log')
 record=json.loads(value);(b/'证据'/f'{run}_chat_prefix_result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));state('CHAT_PREFIX_RESULT',run=run,result=record)
parser=argparse.ArgumentParser();parser.add_argument('--resume-running-b',action='store_true');parser.add_argument('--b-run',default='stage1_swiglu_d');parser.add_argument('--resume-from-stage2',choices=('stage2_post','stage2_post_pre','stage2_all'));args=parser.parse_args();assert args.b_run.replace('_','').isalnum()
try:
 if args.resume_from_stage2:
  assert not args.resume_running_b
  assert json.loads((b/'证据/shared_swiglu_selection.json').read_text())['D_shared_swiglu'] in ('v4_fused','reference')
  state('RESUME_STAGE2_EXISTING_RUN',run=args.resume_from_stage2)
  stages=('stage2_post','stage2_post_pre','stage2_all');stages=stages[stages.index(args.resume_from_stage2):]
  for index,run in enumerate(stages):
   if index:launch(run)
   wait_finished(run,run);validate(run,run);probe_chat_after_performance(run);stop(run)
  state('ALL_BENCHMARKS_VALIDATED_AND_RANKS_STOPPED');raise SystemExit(0)
 if not args.resume_running_b:
  wait_finished('stage1_ref','stage1_ref_v2');baseline=validate('stage1_ref','stage1_ref_v2')
  state('PREFIX_PARITY_PROBE',run='stage1_ref')
  cmd=['docker','exec','-d','-e','MUSA_VISIBLE_DEVICES=-1','xiaopang1110_v41_decode_sj10','bash','-c','python3 /work/scripts/check_prefix_parity.py --output /work/02_correctness/stage1_ref_prefix > /work/runs/stage1_ref/prefix_parity.log 2>&1; echo $? > /work/runs/stage1_ref/prefix_parity.exit']
  remote('import subprocess;subprocess.run('+repr(cmd)+',check=True)')
  for _ in range(150):
   if read('runs/stage1_ref/prefix_parity.exit') is not None:break
   time.sleep(2)
  else:raise RuntimeError('Prefix probe still running; do not stop ranks underneath it')
  probe=json.loads(read('02_correctness/stage1_ref_prefix/result.json'))
  if probe['status']=='TOKEN_MISMATCH':raise RuntimeError('Reference same-prefix token mismatch requires review')
  stop('stage1_ref');launch(args.b_run);
 else:
  baseline=json.loads((b/'证据/stage1_ref_validated_summary.json').read_text());assert baseline['status']=='PASS'
 wait_finished(args.b_run,args.b_run);candidate=validate(args.b_run,args.b_run)
 diffs=[]
 for a,z in zip(baseline['results'],candidate['results']):
  m=a['metrics']['tpot_ms'];n=z['metrics']['tpot_ms'];threshold=max(.002*m['mean'],2*math.sqrt((m['sample_stddev']**2+n['sample_stddev']**2)/5));diffs.append({'tokens':a['input_tokens'],'old_ms':m['mean'],'new_ms':n['mean'],'threshold_ms':threshold,'clear_improvement':m['mean']-n['mean']>threshold})
 use_fused=all(x['clear_improvement'] for x in diffs);decision={'D_shared_swiglu':'v4_fused' if use_fused else 'reference','criteria':'Both groups improve TPOT beyond two combined standard errors and 0.2%; otherwise retain reference for mHC tests. This is a practical decision rule, not a universal statistical claim.','comparisons':diffs}
 (b/'证据/shared_swiglu_selection.json').write_text(json.dumps(decision,indent=2));state('SWIGLU_BASELINE_SELECTED',decision=decision)
 # Touch only not-yet-launched configuration files.
 remote(f'from pathlib import Path;import json;p=Path({root!r})/"artifacts/prepared"\nfor run in ("stage2_post","stage2_post_pre","stage2_all"):\n q=p/run/"sj10.json";d=json.loads(q.read_text())\n for rank in d["ranks"]:rank["environment"]["XLLM_DSV41_MUSA_SHARED_SWIGLU"]={decision["D_shared_swiglu"]!r}\n q.write_text(json.dumps(d,indent=2))')
 stop(args.b_run)
 for run in ('stage2_post','stage2_post_pre','stage2_all'):
  launch(run);wait_finished(run,run);validate(run,run);probe_chat_after_performance(run);stop(run)
 state('ALL_BENCHMARKS_VALIDATED_AND_RANKS_STOPPED')
except Exception as error:
 state('NEEDS_REVIEW',error=repr(error));raise
