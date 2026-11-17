from pathlib import Path
import json
p=Path('/expt/builds/e44_mixed_mccl_module_v2/commands.json');v=json.loads(p.read_text());print(json.dumps({'cpp':v[0],'link':v[1]},indent=2))
b=Path('/workspace/xllm-mt-main/build/cmake.linux-x86_64-cpython-310');print('parallel archives',[str(x) for x in (b/'xllm/core/framework/parallel_state').rglob('*.a')])
