# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Resume frozen B after both roles already started; never restart roles here."""
from pathlib import Path
import datetime
import json
import subprocess
import time

ROOT = Path("/home/xiaopang/桌面/Albert分享/20261021从xllm开始/王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它/06_最终报告/验收脚本/SGLang同窗口")
CONTROL = Path("/home/xiaopang/桌面/Albert分享/20261021从xllm开始/王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它/00_总控与恢复")
def remote(host: str, script: str, receipt: str, container: str | None = None, timeout: int = 40) -> dict:
    args = ['python3', str(CONTROL / 'remote_python_tolerant.py'), '--host', host]
    if container:
        args += ['--container', container]
    args += ['--file', str(ROOT / script), '--timeout', str(timeout), '--save', str(ROOT / receipt)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout+70)
    assert result.returncode == 0, (script, result.stdout[-2000:], result.stderr[-1000:])
    response = json.loads((ROOT / receipt).read_text())
    assert response['exit_status'] == 0, response
    return json.loads(response['stdout'].strip().splitlines()[-1])

for attempt in range(2, 92):
    state = remote('sj9', 'ready.py', f'就绪_{attempt}.json')
    print(json.dumps({'stage':'SG_READY_CHECK','detail':state}), flush=True)
    if state['ready']:
        break
    remote('sj9', 'poll.py', f'启动日志_{attempt}.json')
    time.sleep(20)
else:
    raise RuntimeError('No readiness after 30 minutes. Preserve services for inspection; no auto restart.')
remote('sj9', 'start_router.py', '路由启动.json')
remote('sj9', 'publish_client_after_ready.py', '客户端发布.json')
remote('sj10', 'start_client.py', '客户端启动.json', 'xiaopang1116_v41_match_decode_sj10')
for attempt in range(1, 181):
    state = remote('sj9', 'poll.py', f'正式状态_{attempt}.json')
    print(json.dumps({'stage':'SG_FORMAL_WAIT','at':state['at'],'finished':state.get('finished.json'),'summary':state.get('performance/summary.json')}, ensure_ascii=False), flush=True)
    if 'finished.json' in state:
        assert json.loads(state['finished.json'])['exit_code'] == 0, state
        break
    time.sleep(15)
else:
    raise RuntimeError('Formal client did not finish in bounded monitoring window')
print('SG_FORMAL_FINISHED_REQUIRES_IDENTITY_CHECKED_NORMAL_EXIT_AND_ARCHIVE', flush=True)

