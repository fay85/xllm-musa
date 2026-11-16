from pathlib import Path
import json,hashlib,tarfile,shutil,subprocess
r=Path('/expt');out=r/'diagnostics/e40_dp8_memory_budget/sg_dense_sources';out.mkdir(exist_ok=True);sg=Path('/workspace/sglang');files=['python/sglang/srt/layers/quantization/fp8_utils.py','python/sglang/srt/layers/quantization/w4afp8.py','python/sglang/kernels/ops/quantization/musa_decode_quant_gemm.py','python/sglang/kernels/ops/quantization/musa_mate_block128.py'];rows=[]
for name in files:
 p=sg/name;assert p.is_file();d=out/name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d);rows.append({'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
commit=subprocess.run(['git','-C',str(sg),'rev-parse','HEAD'],capture_output=True,text=True);v={'source':str(sg),'head':commit.stdout.strip(),'commit_command_exit':commit.returncode,'files':rows};(out/'manifest.json').write_text(json.dumps(v,indent=2));archive=r/'exports/e40_sg_dense_sources.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='sg_dense_sources')
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'manifest':v}))
