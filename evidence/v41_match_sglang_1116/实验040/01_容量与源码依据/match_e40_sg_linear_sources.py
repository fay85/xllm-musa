from pathlib import Path
import json,re
root=Path('/workspace/sglang/python/sglang');out=[]
for p in (root/'srt/layers/quantization').rglob('*.py'):
 if p.name not in ['fp8.py','fp8_utils.py','musa_fp8.py','musa_w4afp8.py','w4afp8.py'] and 'musa' not in p.name:continue
 for i,line in enumerate(p.read_text(errors='replace').splitlines(),1):
  if re.search(r'block.*32|32.*block|Musa.*Fp8|mate.*gemm|fp8_blockwise|block_quant|GroupShape|scaled_mm',line,re.I):out.append({'file':str(p),'line':i,'text':line})
print(json.dumps(out))
