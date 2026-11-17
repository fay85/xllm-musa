from pathlib import Path
import json,subprocess,hashlib,tarfile
s=Path('/workspace/xllm-mt-main');b=s/'build/cmake.linux-x86_64-cpython-310';o=Path('/expt/diagnostics/e52_rope_build_inputs');o.mkdir(exist_ok=False)
cmds=json.loads((b/'compile_commands.json').read_text());hits=[x for x in cmds if 'deepseek_v4_dsa_rope.cu' in x['file']];assert len(hits)==1
(o/'rope_compile_command.json').write_text(json.dumps(hits[0],indent=2))
for name in ['e50_native_gate_gemv_v3','e40_prequant_native_v1']:
 p=Path('/expt/builds')/name/'commands.json';(o/(name+'_commands.json')).write_bytes(p.read_bytes())
p=Path('/workspace/sglang/python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/norm_rope_kernels.py');assert p.exists();(o/p.name).write_bytes(p.read_bytes())
z=Path('/expt/exports/e52_rope_build_inputs.tar.gz')
with tarfile.open(z,'w:gz') as t:t.add(o,arcname='e52_rope_build_inputs')
print(json.dumps({'archive':str(z),'bytes':z.stat().st_size,'sha256':hashlib.sha256(z.read_bytes()).hexdigest(),'compile':hits[0]}))
