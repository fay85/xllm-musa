from pathlib import Path
import json,hashlib
from safetensors import safe_open
from safetensors.torch import save_file
root=Path('/data/nfs_shared/models/DeepSeek-V4.1-Flash-1017');out=Path('/work/artifacts/hc_1017_subset');out.mkdir(exist_ok=True)
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];values={};records=[]
for layer in (0,1,14,39):
 for part in ('attn','ffn'):
  for suffix in ('fn','scale','base'):
   key=f'layers.{layer}.hc_{part}_{suffix}';f=index[key]
   with safe_open(root/f,framework='pt',device='cpu') as h:values[key]=h.get_tensor(key).clone()
   records.append({'tensor':key,'shard':f,'shape':list(values[key].shape),'dtype':str(values[key].dtype)})
save_file(values,out/'hc.safetensors');(out/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'hc.safetensors' for k in values}},indent=2))
(out/'manifest.json').write_text(json.dumps({'source':str(root),'model_commit':'1bae0e1088993b9b82de08f59de5be5a34e42170','tensors':records,'sha256':hashlib.sha256((out/'hc.safetensors').read_bytes()).hexdigest(),'bytes':(out/'hc.safetensors').stat().st_size},indent=2))
print((out/'manifest.json').read_text())
