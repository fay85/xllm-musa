# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Copy small auditable evidence, preserving originals and listing exclusions."""
from pathlib import Path
import hashlib
import json
import shutil

SOURCE = Path("/home/xiaopang/桌面/Albert分享/20261021从xllm开始/王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它/06_最终报告")
DEST = Path("/home/xiaopang/桌面/Albert分享/20261021从xllm开始/王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/03_代码/xllm-musa/evidence/v41_match_sglang_1116/最终验收")
DEST.mkdir(exist_ok=True)
included=[]
omitted=[]
skip_top={'候选收口复核_尚待E55与E56.md','当前阶段说明_待最终验收.md','阶段历史进展_非最终验收.png','阶段历史数据_非最终验收.csv','阶段进展图_非最终验收.py'}
for p in SOURCE.rglob('*'):
 if not p.is_file():continue
 relative=p.relative_to(SOURCE)
 if relative.name in skip_top or '__pycache__' in relative.parts:continue
 if any(z in relative.name for z in ['容器可写层盘点','最终容器盘点_清理前']):continue
 if p.name=='container_before.json' or p.name.endswith(('.tar.gz','.pt.trace.json','.pftrace','.native_stacks.jsonl')) or p.stat().st_size>8_000_000:
  omitted.append({'path':str(relative),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'local':str(p)})
  continue
 dest=DEST/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 data=p.read_bytes()
 if p.suffix=='.md':
  text=data.decode()
  for i in [55,56,57]:text=text.replace('../02_实验记录/实验_'+str(i).zfill(3),'../实验'+str(i).zfill(3))
  data=(text.rstrip()+'\n').encode()
 dest.write_bytes(data)
 included.append({'path':str(relative),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
manifest={'scope':'Raw rows, logs, configs, checks, screenshots and final reports. Markdown experiment links adjusted for Git layout; source_sha retains original. Large archives/traces/ELF/weights and full Docker inspect excluded.','included':included,'omitted':omitted,'nfs_root':'/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116'}
(DEST/'归档范围与校验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'files':len(included),'bytes':sum(x['bytes'] for x in included),'omitted':len(omitted)},ensure_ascii=False))

