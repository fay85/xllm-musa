from pathlib import Path
import json
p=Path('/workspace/sglang/python/sglang/kernels/ops/quantization/fp8_kernel.py');lines=p.read_text().splitlines();out=[]
for i,l in enumerate(lines):
 if l.startswith('def sglang_per_token_group_quant_fp8'):
  out=lines[max(0,i-2):i+170];break
p2=Path('/expt/repos/xllm-musa/tests/core/kernels/musa/musa_python_kernel_test_module.cpp');a=p2.read_text().splitlines();print(json.dumps({'fp8_quant':out,'bindings':a[120:160]},ensure_ascii=False))
