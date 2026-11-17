from pathlib import Path
import json
p=Path('/work/artifacts/model_view_w4afp8_mt_1017/config.json');cfg=json.loads(p.read_text());print(json.dumps(cfg,indent=2))
