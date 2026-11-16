from pathlib import Path
import json,struct,math,re,datetime
r=Path('/expt');model=Path('/work/artifacts/model_view_w4afp8_mt_1017');out=r/'diagnostics/e40_dp8_memory_budget';out.mkdir(exist_ok=True)
index=json.loads((model/'model.safetensors.index.json').read_text());files=sorted(set(index['weight_map'].values()));rows={}
for name in files:
 with (model/name).open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];assert n<100_000_000;header=json.loads(f.read(n))
 for key,v in header.items():
  if key=='__metadata__':continue
  rows[key]={'shape':v['shape'],'dtype':v['dtype'],'disk_bytes':v['data_offsets'][1]-v['data_offsets'][0],'file':name}
assert set(index['weight_map'])<=set(rows)
config=json.loads((model/'config.json').read_text());(out/'checkpoint_header_inventory.json').write_text(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'config':config,'tensors':rows},indent=2))
selected={k:v for k,v in rows.items() if (k.startswith('layers.0.') and 'experts.' not in k) or k.startswith(('embed','head','norm','mtp.0.','mtp.main','mtp.markov')) and 'experts.' not in k or k.startswith('layers.1.engram.')}
print(json.dumps({'out':str(out),'tensor_count':len(rows),'files':len(files),'total_disk_bytes':sum(v['disk_bytes'] for v in rows.values()),'config':config,'selected':selected},ensure_ascii=False))
