import runpy as _compile_guard_runpy
from pathlib import Path as _compile_guard_path
_compile_guard_root = _compile_guard_path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control')
if not _compile_guard_root.exists():
    _compile_guard_root = _compile_guard_path('/expt/00_control')
_compile_guard_runpy.run_path(str(_compile_guard_root / 'assert_compilation_clear.py'))

"""Run inside the same SJ10 CPU client container as xLLM final rounds."""
from pathlib import Path
import datetime,json,os,subprocess
root=Path('/expt');run=root/'runs/sglang_B_recheck_v1';run.mkdir(exist_ok=False)
ref=Path('/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/B_recheck_20261117')
# The client container does not mount the SG root; runner and identity are
# copied into /expt/scripts by the host publication step.
runner=root/'scripts/sglang_reference_final_client.py'
identity=root/'scripts/sglang_reference_final_identity.json'
assert runner.is_file() and identity.is_file()
cmd=['python3',str(runner),'--case','B','--router','http://10.121.32.7:28192','--decode','http://10.121.32.9:28588','--inputs','/work/artifacts/benchmark_inputs_v1/input256_output256.jsonl','--tokenizer','/work/artifacts/model_view_w4afp8_mt_1017','--evalscope','/work/artifacts/bench_venv_v2/bin/evalscope','--out',str(run/'performance'),'--runtime-identity',str(identity)]
worker=run/'worker.py'
worker.write_text('from pathlib import Path\nimport subprocess,json,datetime\nr=Path('+repr(str(run))+')\ncmd='+repr(cmd)+'\nwith (r/"client.log").open("xb") as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)\n(r/"finished.json").write_text(json.dumps({"exit_code":p.returncode,"at":datetime.datetime.now().astimezone().isoformat()}))\n')
with (run/'supervisor.log').open('xb') as f:
 p=subprocess.Popen(['python3',str(worker)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,LD_PRELOAD='',PYTHONPATH='',MUSA_VISIBLE_DEVICES='-1',TOKENIZERS_PARALLELISM='false'))
v={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat(),'client_host':'sj10','command':cmd}
(run/'job.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))

