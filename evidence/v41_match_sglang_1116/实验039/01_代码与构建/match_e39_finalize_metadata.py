from pathlib import Path
import json
r=Path('/expt')
for mode in ['off','on']:
 for h in ['sj9','sj10']:
  p=r/f'runs/p39_static_activation_{mode}_v1/{h}/prepared_config.json';c=json.loads(p.read_text());original=json.loads((r/f'runs/p37_sg438_native_mccl_v2/{h}/prepared_config.json').read_text());c['source_dir']=original['source_dir'];c['artifact_source_note']='Frozen E37 ELF at commit0130ef9a; current CPU builder source has unused E38 edits; actual source SHA identified by E37 manifest';c['shared_expert_parallelism']='TP8' if h=='sj10' else 'replicated';p.write_text(json.dumps(c,indent=2))
print('metadata corrected before launch')
