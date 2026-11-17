from pathlib import Path
import json,subprocess,os
w=Path.cwd();g=w/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做';repo=g/'03_代码/xllm-musa';e=Path(__file__).parent.parent
env=dict(os.environ,PYTHONPATH='/tmp/xiaopang_v41_perfetto_pkg')
for mode in ['off_v1','on_v1']:
 raw=e/('原始证据_'+mode)/'trace_analysis';events=json.loads((raw/'sj10/rank0.pt.trace.json').read_text())['traceEvents']
 candidates=[v for v in events if v.get('ph')=='X' and v.get('name')=='V41/DSpark/full_draft_graph_replay' and 'Record function id' in v.get('args',{})]
 event=sorted(candidates,key=lambda v:v['ts'])[-1];record=event['args']['Record function id']
 assert sum(v.get('ph')=='X' and v.get('args',{}).get('Record function id')==record for v in events)==1
 out=e/'04_trace'/('E55_'+mode+'_actual_draft_scope');out.mkdir(exist_ok=False)
 (out/'selected_actual_scope.json').write_text(json.dumps(event,indent=2))
 trace=out/'actual_draft.pftrace'
 subprocess.run(['python3',str(repo/'tools/musa_deepseek_v41/convert_native_cpp_trace.py'),'--trace',str(raw/'sj10/rank0.pt.trace.json'),'--native-stacks',str(raw/'sj10/rank0.native_stacks.jsonl'),'--resolved-symbols',str(raw/'resolved_symbols.json'),'--binary-sha256','c8a4f4cf1819baad39f30b8ad9962de73bb69653b5a6bd71d75c2d600c7bef2d','--step','last','--step-prefix','V41/step/','--record-id',str(record),'--include-aten','--include-runtime','--output',str(trace),'--verify'],env=env,check=True)
 subprocess.run(['python3',str(repo/'tools/musa_deepseek_v41/render_native_cpp_trace.py'),'--trace',str(trace),'--receipt',str(trace.with_suffix('.receipt.json')),'--output',str(out/'Perfetto图'),'--browser-profile','/tmp/xiaopang_e55_'+mode+'_draft_browser','--ui-url','http://127.0.0.1:18766'],env=env,check=True)
 print(mode,record,flush=True)

