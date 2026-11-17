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
v=remote('sj10','xiaopang1116_v41_match_decode_sj10',S/'e57_reverse_poll.py','E57反向最终状态.json',30)
assert json.loads(json.loads(v['stdout'])['formal_client/finished.json'])['exit_code']==0
for h,container in [('sj10','xiaopang1117_v41_sg438_decode_sj10'),('sj9','xiaopang1117_v41_sg438_prefill_sj9')]:remote(h,container,S/('e57_reverse_stop_'+h+'.py'),'E57反向'+h+'正常退出请求.json')
v=remote('sj10','xiaopang1116_v41_match_decode_sj10',S/'e57_reverse_wait_exit.py','E57反向退出结果.json',60)
x=json.loads(v['stdout']);assert x['both_finished'] and all(x[h]['exit_codes']==[0]*8 for h in ['sj9','sj10'])
for h in ['sj9','sj10']:remote(h,None,S/'e57_reverse_release.py','E57反向'+h+'资源释放.json',45)
remote('sj10','xiaopang1116_v41_match_decode_sj10',S/'e57_reverse_archive.py','E57反向归档.json')
subprocess.run(['python3',str(S/'e57_reverse_download.py')],check=True)
subprocess.run(['python3',str(S/'e57_analyze_local.py')],check=True)
print('E57_REVERSE_ARCHIVED_PENDING_DECISION',flush=True)
