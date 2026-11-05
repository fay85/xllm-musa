"""Generate reviewed two-host configurations; never launches or reserves GPUs."""
from pathlib import Path
import argparse, hashlib, json, re
parser=argparse.ArgumentParser();parser.add_argument('--prefill',required=True);parser.add_argument('--decode',required=True);parser.add_argument('--run-id',required=True);args=parser.parse_args()
assert re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.run_id), 'Use a simple run ID'
ips={'dev4':'10.20.34.24','dev5':'10.20.34.25','dev6':'10.20.34.26','dev7':'10.20.34.28'}
assert args.prefill in ips and args.decode in ips and args.prefill!=args.decode
root=Path('/work');repo=root/'repos/xllm-musa';binary=root/'artifacts/bin/xllm';model=root/'artifacts/model_view_0909'
assert binary.is_file(),'Native build not ready'
out=root/'artifacts/prepared'/args.run_id;out.mkdir(parents=True,exist_ok=False)
sha=hashlib.sha256(binary.read_bytes()).hexdigest();control=ips[args.decode]
common_libs=['/work/artifacts/bin','/work/repos/xllm-musa/build/cmake.linux-x86_64-cpython-310/mooncake-common','/usr/local/lib/python3.10/dist-packages/torch/lib','/usr/local/lib/python3.10/dist-packages/torch_musa/lib','/usr/local/musa/lib','/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib','/opt/intel/oneapi/mkl/lib/intel64','/usr/local/openmpi/lib','/usr/local/lib','/usr/lib/x86_64-linux-gnu']
for host,role,portbase in [(args.prefill,'PREFILL',21000),(args.decode,'DECODE',22000)]:
 rows=[]
 for rank in range(8):
  flags=dict(model=str(model),model_id='deepseek-v4.1-flash-0909',backend='llm',model_impl='python',python_model_path=str(repo),host=ips[host],port=portbase+rank,master_node_addr=f'{ips[host]}:{portbase+100}',nnodes=8,node_rank=rank,dp_size=1,ep_size=8,cp_size=1,kv_split_size=1,enable_graph=False,enable_schedule_overlap=False,enable_prefix_cache=False,enable_in_batch_prefix_cache=False,enable_chunked_prefill=True,max_tokens_per_chunk_for_prefill=256,max_tokens_per_batch=256,max_seqs_per_batch=2,communication_backend='mccl',max_cache_size=1024**3,max_memory_utilization=0.90,random_seed=1105,logtostderr=True,enable_dump_config_json=True,dump_config_json_file=f'/work/runs/{args.run_id}/{host}/resolved_rank{rank}.json',enable_disagg_pd=True,instance_role=role,kv_cache_transfer_mode='PULL',enable_native_service_adapter=True,etcd_addr=f'{control}:29079',etcd_namespace=f'xiaopang1105-dsv41-{args.run_id}',disagg_pd_port=portbase+200,transfer_listen_port=portbase+300+rank,num_speculative_tokens=0,block_size=128,use_cpp_chat_template=True)
  env=dict(LD_PRELOAD='/workspace/artifacts/libmusa_backend_init.so',PYTHONPATH=str(repo),LD_LIBRARY_PATH=':'.join(common_libs),XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS='120',XLLM_MUSA_ENABLE_TP_CUSTOM_AR='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MAX_JOBS='2',MC_MIN_RPC_PORT=str(portbase+400+rank*10),MC_MAX_RPC_PORT=str(portbase+409+rank*10),ENABLE_DECODE_RESPONSE_TO_SERVICE='true',MCCL_TIMEOUT='120',MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',MTHREADS_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
  rows.append({'rank':rank,'command':[str(binary)]+[f'--{k}={str(v).lower() if isinstance(v,bool) else v}' for k,v in flags.items()],'environment':env})
 cfg={'host':host,'host_ip':ips[host],'role':role,'TP':8,'EP':8,'MoE_TP':1,'mode':'1P1D','eager':True,'binary':str(binary),'binary_sha256':sha,'model':str(model),'ranks':rows,'namespace':f'xiaopang1105-dsv41-{args.run_id}','control_host':control,'control_ports':{'etcd_peer':29078,'etcd':29079,'router_http':29080,'router_rpc':29081,'coordinator_http':29082,'coordinator_rpc':29083},'status':'prepared_only_requires_idle_check_and_loader_preflight'}
 (out/f'{host}.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps({'prepared':str(out),'prefill':args.prefill,'decode':args.decode,'status':'not_launched'}))
