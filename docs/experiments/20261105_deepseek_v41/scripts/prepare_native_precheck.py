"""Prepare one-host full-model loading/answer preflight, never a 1P1D result."""
from pathlib import Path
import argparse,json,re
parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args();assert re.fullmatch(r'[A-Za-z0-9_-]{1,48}',args.run_id)
root=Path('/work');source=root/'artifacts/prepared/candidate_v1/dev7.json';cfg=json.loads(source.read_text());run_id=args.run_id
cfg['role']='PRECHECK';cfg['mode']='single_host_precheck_not_1P1D';cfg['requires_control_plane']=False
for item in cfg['ranks']:
 rank=item['rank'];flags=dict(x[2:].split('=',1) for x in item['command'][1:])
 flags.update(port=str(24000+rank),master_node_addr='10.20.34.28:24100',enable_disagg_pd='false',enable_native_service_adapter='false',instance_role='DEFAULT',dump_config_json_file=f'/work/runs/{run_id}/dev7/resolved_rank{rank}.json',disagg_pd_port='24200',transfer_listen_port=str(24300+rank))
 for key in ['etcd_addr','etcd_namespace','kv_cache_transfer_mode']:flags.pop(key,None)
 item['command']=[cfg['binary']]+['--'+k+'='+v for k,v in flags.items()]
 item['environment'].update(MC_MIN_RPC_PORT=str(24400+10*rank),MC_MAX_RPC_PORT=str(24409+10*rank),ENABLE_DECODE_RESPONSE_TO_SERVICE='false')
cfg['status']='prepared_only_requires_idle_check';dest=root/'artifacts/prepared'/run_id;dest.mkdir(exist_ok=False);(dest/'dev7.json').write_text(json.dumps(cfg,indent=2));print(str(dest/'dev7.json'))
