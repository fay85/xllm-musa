from pathlib import Path
import json,hashlib
r=Path('/workspace/sglang/python/sglang');out={}
for file,a,b in [('srt/layers/quantization/fp8_utils.py',1350,1555),('srt/layers/quantization/w4afp8.py',40,115)]:
 p=r/file;lines=p.read_text().splitlines();out[file]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'lines':'\n'.join(f'{i+1}: {x}' for i,x in enumerate(lines) if a<=i+1<=b)}
print(json.dumps(out))
