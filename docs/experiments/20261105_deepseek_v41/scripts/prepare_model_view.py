"""Create our metadata copy and weight symlinks, leaving official files unchanged."""
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离')
source=Path('/data/nfs_shared/models/DeepSeek-V4.1-Flash')
view=root/'artifacts/model_view_0909'
view.mkdir(exist_ok=False)
manifest=[]
for p in sorted(source.rglob('*')):
 relative=p.relative_to(source)
 if any(part.startswith('.') or part == '__pycache__' for part in relative.parts) or not p.is_file():continue
 dest=view/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.safetensors':
  dest.symlink_to(p);manifest.append({'file':str(relative),'kind':'readonly_source_symlink','bytes':p.stat().st_size})
 else:
  if p.stat().st_size>100*1024**2:raise RuntimeError('Unexpected large non-weight asset: '+str(relative))
  shutil.copy2(p,dest);manifest.append({'file':str(relative),'kind':'metadata_copy','sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
(root/'evidence/model_view_manifest.json').write_text(json.dumps({'source':str(source),'view':str(view),'files':manifest},indent=2))
code='from xllm.python.models.deepseek_v41_engram import build_engram_hash_state; print(build_engram_hash_state("/work/artifacts/model_view_0909"))'
cmd=['docker','exec','-d','-w','/work/repos/xllm-musa','xiaopang1105_cpu_test_dev6','bash','-c',"python3 -c "+__import__('shlex').quote(code)+' > /work/evidence/engram_hash_build.log 2>&1; rc=$?; echo "$rc" > /work/evidence/engram_hash_build.exit']
subprocess.run(cmd,check=True)
print(json.dumps({'view':str(view),'files':len(manifest),'hash_build':'launched'}))
