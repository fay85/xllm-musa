from pathlib import Path
import subprocess,os
w=Path.cwd();g=w/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做';r=g/'03_代码/xllm-musa';e=g/'20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它/02_实验记录/实验_055';raw=e/'原始证据_on_v1/trace_analysis';out=e/'04_trace/E55_on_draft_gate';out.mkdir(parents=True,exist_ok=True);t=out/'actual_native_last.pftrace';env=dict(os.environ,PYTHONPATH='/tmp/xiaopang_v41_perfetto_pkg')
subprocess.run(['python3',str(r/'tools/musa_deepseek_v41/render_native_cpp_trace.py'),'--trace',str(t),'--receipt',str(t.with_suffix('.receipt.json')),'--output',str(out/'Perfetto图'),'--browser-profile','/tmp/xiaopang_match_e55_on_browser','--ui-url','http://127.0.0.1:18766'],env=env,check=True)

