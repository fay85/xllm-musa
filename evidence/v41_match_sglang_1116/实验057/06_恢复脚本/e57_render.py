from pathlib import Path
import subprocess,os,json
w=Path.cwd();g=w/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做';r=g/'03_代码/xllm-musa';e=g/'20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它/02_实验记录/实验_057';raw=e/'原始证据_candidate/trace_analysis';out=e/'04_trace/E57_progress_disabled';out.mkdir(parents=True,exist_ok=True);t=out/'actual_native_last.pftrace';env=dict(os.environ,PYTHONPATH='/tmp/xiaopang_v41_perfetto_pkg')
subprocess.run(['python3',str(r/'tools/musa_deepseek_v41/convert_native_cpp_trace.py'),'--trace',str(raw/'sj10/rank0.pt.trace.json'),'--native-stacks',str(raw/'sj10/rank0.native_stacks.jsonl'),'--resolved-symbols',str(raw/'resolved_symbols.json'),'--binary-sha256',json.loads((raw.parent/'roles/sj10/role_config.json').read_text())['binary_sha256'],'--step','last','--step-prefix','V41/step/','--include-aten','--include-runtime','--output',str(t),'--verify'],env=env,check=True)
subprocess.run(['python3',str(r/'tools/musa_deepseek_v41/render_native_cpp_trace.py'),'--trace',str(t),'--receipt',str(t.with_suffix('.receipt.json')),'--output',str(out/'Perfetto图'),'--browser-profile','/tmp/xiaopang_match_e57_browser','--ui-url','http://127.0.0.1:18766'],env=env,check=True)

