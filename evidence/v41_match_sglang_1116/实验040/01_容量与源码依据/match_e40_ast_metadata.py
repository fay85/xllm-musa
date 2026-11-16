import inspect,json
from pathlib import Path
from triton.compiler import ASTSource
v={'signature':str(inspect.signature(ASTSource)),'source':inspect.getsource(ASTSource)};root=Path('/expt/diagnostics/e40_prequant_real_weights_v1/triton_cache');items=[]
for p in root.rglob('*.json'):
 try:j=json.loads(p.read_text())
 except:continue
 if isinstance(j,dict) and '_prequantized_decode_gemm_kernel' in str(j.get('name','')):items.append({'path':str(p),'data':j})
v['sg_gemm_metadata']=items[:8];print(json.dumps(v))
