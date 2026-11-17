from pathlib import Path
import json
r=Path("/expt/builds/e53_rope_qdq_production_module_v1")
print(json.dumps({n:(r/n).read_text(errors="replace")[-2400:] for n in ["finished.json","worker.log","cu.log","cpp.log","link.log","import.log"] if (r/n).exists()}))
