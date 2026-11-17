import runpy as _compile_guard_runpy
from pathlib import Path as _compile_guard_path
_compile_guard_root = _compile_guard_path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control')
if not _compile_guard_root.exists():
    _compile_guard_root = _compile_guard_path('/expt/00_control')
_compile_guard_runpy.run_path(str(_compile_guard_root / 'assert_compilation_clear.py'))

from pathlib import Path
import json,subprocess,os,datetime
root=Path('/expt');run=root/'runs/p55_draft_gate_reverse_off_v1';assert not (run/'formal_client').exists();p=root/'scripts/match_e55_client_v1.py';assert p.exists()
with (run/'formal_client_supervisor.log').open('xb') as log:q=subprocess.Popen(['python3',str(p),'--run',run.name],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,LD_PRELOAD='',MUSA_VISIBLE_DEVICES='-1',PYTHONPATH='',TOKENIZERS_PARALLELISM='false'),start_new_session=True)
v={'pid':q.pid,'start_ticks':Path('/proc',str(q.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()};(run/'formal_client_launch.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
