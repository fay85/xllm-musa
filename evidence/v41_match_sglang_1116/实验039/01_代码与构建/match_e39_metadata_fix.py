from pathlib import Path
import json
r=Path('/expt')
for mode in ['off','on']:
 for host in ['sj9','sj10']:
  p=r/f'runs/p39_static_activation_{mode}_v1/{host}/prepared_config.json';c=json.loads(p.read_text());env=c['ranks'][0]['environment'];c['framework_executor']='eager; native model handles target full Graph and static draft';c['shared_expert_parallelism']='TP8' if env.get('XLLM_DSV41_SHARED_EXPERT_TP')=='1' else 'see actual runtime env';c['source_dir']='/expt/artifacts/native_sg438_mccl_v1 (frozen ELF built before E38 working-tree changes)';p.write_text(json.dumps(c,indent=2));print(mode,host,{k:v for k,v in env.items() if any(x in k for x in ['SHARED','STATIC_ACTIVATION','GRAPH','W4A8_SO'])})
