from pathlib import Path
import json,shlex
b=Path('/workspace/xllm-mt-main/build/cmake.linux-x86_64-cpython-310');p=b/'compile_commands.json';v=json.loads(p.read_text());item=next(x for x in v if x['file'].endswith('/musa_aot_kernel.cpp'));cmd=shlex.split(item['command']);print(json.dumps({'compile':cmd,'runtime_files':[x.name for x in Path('/expt/artifacts/native_sg438_control_v1/lib/runtime').iterdir()],'torch_musa_libs':[x.name for x in Path('/opt/venv/lib/python3.10/site-packages/torch_musa/lib').iterdir()]}))
