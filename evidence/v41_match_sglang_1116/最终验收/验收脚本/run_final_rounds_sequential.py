from pathlib import Path
import datetime,json,subprocess,time
O=Path('/home/xiaopang/桌面/Albert分享/20261021从xllm开始/王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它');S=O/'06_最终报告/验收脚本';C=O/'00_总控与恢复'
def remote(host,container,file,receipt,timeout=150):
 args=['python3',str(C/'remote_python_tolerant.py'),'--host',host]
 if container:args+=['--container',container]
 args+=['--file',str(file),'--timeout',str(timeout),'--save',str(receipt)]
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout+70)
 assert p.returncode==0,(str(file),p.stdout[-1500:],p.stderr[-500:])
 v=json.loads(receipt.read_text());assert v['exit_status']==0,v['stderr']
 return json.loads(v['stdout'].strip().splitlines()[-1])
def record(state,run,extra=None):
 now=datetime.datetime.now().astimezone().isoformat();v={'at':now,'stage':state,'run':run,'detail':extra};print(json.dumps(v,ensure_ascii=False),flush=True)
 with (S/'执行流水.jsonl').open('a') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n')
 p=C/'任务状态.json';x=json.loads(p.read_text());x.update(updated_at=now,status=state,current_experiment=run,next_action='Guarded final rounds; same frozen source/config. Continue from exact receipts, do not repeat completed round.');x['active_remote_jobs']['gpu']=None if state.endswith('ARCHIVED') else run;x['runtime_jobs']={'run':run,'state':state};p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
selection=json.loads((S/'最终选择.json').read_text());assert selection['decision_complete'] and selection['closure_review_complete']
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--continue-running',choices=['final_short2_v1']);args=ap.parse_args()
runs=['final_short1_v1','final_short2_v1','final_short3_v1','final_long_v1']
if args.continue_running:
 assert (S/'final_short1_v1/归档.json').exists() and (S.parent/'原始验收证据/final_short1_v1/formal_client/summary.json').exists()
 runs=runs[1:]
for run in runs:
 d=S/run;assert not (d/'归档.json').exists(),'Already completed; do not rerun'
 if run==args.continue_running:
  for h in ['sj9','sj10']:
   z=json.loads((d/('续跑'+h+'身份核验.json')).read_text());assert z['exit_status']==0
   z=json.loads(z['stdout']);assert len(z['ranks'])==8 and z['run']==run
   assert time.time()-datetime.datetime.fromisoformat(z['at']).timestamp()<240
  z=json.loads((d/'客户端启动_v2.json').read_text());assert z['exit_status']==0 and 'pid' in json.loads(z['stdout'])
 else:
  for h in ['sj9','sj10']:
   v=remote(h,None,C/'resource_preflight.py',d/(h+'_启动前资源.json'),45);assert v['gpu_free'],v
  for h in ['sj9','sj10']:remote(h,None,d/'start_role.py',d/(h+'_启动.json'),60)
  for attempt in range(12):
   v=remote('sj10','xiaopang1116_v41_match_decode_sj10',d/'start_client.py',d/('客户端启动_'+str(attempt)+'.json'),45)
   if 'pid' in v:break
   assert v.get('ready') is False;time.sleep(10)
  else:raise RuntimeError('Role metadata not visible within bounded startup wait; model roles left intact for inspection')
 record('FINAL_RUNNING',run)
 poll=max([int(p.stem.split("_")[-1]) for p in d.glob("状态_*.json")],default=0)
 while True:
  poll+=1;v=remote('sj10','xiaopang1116_v41_match_decode_sj10',d/'poll.py',d/('状态_'+str(poll)+'.json'),30)
  for h in ['sj9','sj10']:assert 'result.json' not in v.get(h,{}),'Role ended unexpectedly; stop orchestration and observe external-use policy'
  if 'finished.json' in v:
   assert json.loads(v['finished.json'])['exit_code']==0,v
   break
  time.sleep(15)
 record('FINAL_MEASURED',run,json.loads(v['summary.json']))
 for h,container in [('sj10','xiaopang1117_v41_sg438_decode_sj10'),('sj9','xiaopang1117_v41_sg438_prefill_sj9')]:remote(h,container,d/('stop_'+h+'.py'),d/(h+'_退出请求.json'))
 v=remote('sj10','xiaopang1116_v41_match_decode_sj10',d/'wait_exit.py',d/'退出结果.json',60)
 assert v['both_finished'] and all(v[h]['exit_codes']==[0]*8 for h in ['sj9','sj10']),v
 for h in ['sj9','sj10']:remote(h,None,d/'release.py',d/(h+'_资源释放.json'),45)
 remote('sj9',None,d/'archive.py',d/'归档.json',60)
 subprocess.run(['python3',str(S/'download_final_evidence.py'),'--receipt',str(d/'归档.json'),'--name',run],check=True)
 record('FINAL_ROUND_ARCHIVED',run)
 if run.startswith('final_short'):
  m=json.loads((S.parent/'原始验收证据'/run/'formal_client/summary.json').read_text());assert m['mean_TPOT_ms_per_token']<=21.57 and m['mean_TTFT_ms']<=990.04 and m['output_tokens_per_latency_sum_second']>=39.45,'Threshold missed; keep all evidence and stop before next round'
print('ALL_FOUR_ROUNDS_ARCHIVED_PENDING_GLOBAL_AUDIT',flush=True)
