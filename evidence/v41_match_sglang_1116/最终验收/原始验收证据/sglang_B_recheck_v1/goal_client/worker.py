from pathlib import Path
import subprocess,json,datetime
r=Path('/expt/runs/sglang_B_recheck_v1')
cmd=['python3', '/expt/scripts/sglang_reference_final_client.py', '--case', 'B', '--router', 'http://10.121.32.7:28192', '--decode', 'http://10.121.32.9:28588', '--inputs', '/work/artifacts/benchmark_inputs_v1/input256_output256.jsonl', '--tokenizer', '/work/artifacts/model_view_w4afp8_mt_1017', '--evalscope', '/work/artifacts/bench_venv_v2/bin/evalscope', '--out', '/expt/runs/sglang_B_recheck_v1/performance', '--runtime-identity', '/expt/scripts/sglang_reference_final_identity.json']
with (r/"client.log").open("xb") as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(r/"finished.json").write_text(json.dumps({"exit_code":p.returncode,"at":datetime.datetime.now().astimezone().isoformat()}))
