from pathlib import Path
import json,inspect
from triton.backends.compiler import AttrsDescriptor
z={'attrs_signature':str(inspect.signature(AttrsDescriptor)),'attrs_source':inspect.getsource(AttrsDescriptor),'irs':[]}
for root,glob in [('/expt/diagnostics/e40_prequant_real_weights_v1/triton_cache','**/_prequantized_decode_gemm_kernel.ttir'),('/expt/builds/e40_prequant_aot_v1/triton_cache','**/prequantized_block32_gemm.ttir')]:
 for p in list(Path(root).glob(glob))[:3]:
  text=p.read_text();z['irs'].append({'path':str(p),'head':text[:6000]})
print(json.dumps(z))
