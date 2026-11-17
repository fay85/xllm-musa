from pathlib import Path
import json
b=Path('/workspace/xllm-mt-main/build/cmake.linux-x86_64-cpython-310');s=(b/'build.sg438.ninja').read_text();flags=sorted(set(l for l in s.splitlines() if l.startswith('  FLAGS =') and '-include ' in l));print(json.dumps(flags[:4]))
