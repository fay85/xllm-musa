from pathlib import Path
import subprocess,json,sys,datetime
O=Path('/home/xiaopang/桌面/Albert分享/20261021从xllm开始/王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它');S=O/'02_实验记录/实验_057/06_恢复脚本';C=O/'00_总控与恢复';helper=C/'remote_python_tolerant.py'
def remote(host,container,file,receipt,timeout=150):
 args=['python3',str(helper),'--host',host]
 if container:args+=['--container',container]
 args+=['--file',str(file),'--timeout',str(timeout),'--save',str(S/receipt)]
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout+70)
 print(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'receipt':receipt,'exit':p.returncode,'stdout':p.stdout[-1800:],'stderr':p.stderr[-500:]},ensure_ascii=False),flush=True)
 assert p.returncode==0
 v=json.loads((S/receipt).read_text());assert v['exit_status']==0
 return v
assert json.loads((S/'E57关闭日志D正常退出请求.json').read_text())['exit_status']==0
remote('sj9','xiaopang1117_v41_sg438_prefill_sj9',S/'e57_candidate_stop_sj9.py','E57关闭日志P正常退出请求.json')
v=remote('sj10','xiaopang1116_v41_match_decode_sj10',S/'e57_candidate_wait_exit.py','E57关闭日志退出结果.json',60)
x=json.loads(v['stdout']);assert x['both_finished'] and all(x[h]['exit_codes']==[0]*8 for h in ['sj9','sj10'])
for h in ['sj9','sj10']:remote(h,None,S/'e57_candidate_release.py','E57关闭日志'+h+'资源释放.json',45)
remote('sj10','xiaopang1116_v41_match_decode_sj10',S/'e57_candidate_trace_archive.py','E57关闭日志归档.json')
subprocess.run(['python3',str(S/'e57_candidate_download.py')],check=True)
for h in ['sj9','sj10']:
 v=remote(h,None,C/'resource_preflight.py','E57反向'+h+'启动前资源.json',45);assert json.loads(v['stdout'])['gpu_free']
for h in ['sj9','sj10']:remote(h,None,S/'e57_reverse_start_role.py','E57反向'+h+'启动.json',60)
for h,container in [('sj9','xiaopang1117_v41_sg438_prefill_sj9'),('sj10','xiaopang1117_v41_sg438_decode_sj10')]:remote(h,container,S/('e57_reverse_environment_'+h+'.py'),'E57反向'+h+'实际环境核验.json',60)
remote('sj10','xiaopang1116_v41_match_decode_sj10',S/'e57_reverse_start_client.py','E57反向客户端启动.json',45)
print('PIPELINE_FINISHED_REVERSE_RUNNING',flush=True)
