from pathlib import Path
import json,hashlib,shutil
r=Path('/expt');oldrun='p2_full_decode_graph_v1';run='p3_attention_replica_graph_v1';out=r/'runs'/run;out.mkdir(exist_ok=False)
checks=r/'builds/attention_replica_v1/small_checks_v7';assert json.loads((checks/'gpu_finished.json').read_text())['exit_code']==0
assert json.loads((checks/'projection_checks.json').read_text())['status']=='PASS_ATTENTION_REPLICATION_PROJECTIONS'
binary=r/'artifacts/attention_replica_v1/bin/xllm';sha=hashlib.sha256(binary.read_bytes()).hexdigest();assert sha=='2ff1368fbb7f32eabe8d5bb9896376e04a926f2ca110cdbdc4cdf19a29f082bc'
so=r/'builds/attention_replica_v1/sparse_h64/artifact/libdeepseek_v41_sparse_h64_bf16.so';assert hashlib.sha256(so.read_bytes()).hexdigest()=='a1f754eeaeb312183cf0e754c923c9aa0b34e3dd95f5b9bbe5b0eee4b11576ce'
dst=r/'artifacts/attention_replica_v1/lib';dst.mkdir(exist_ok=False);shutil.copy2(so,dst/so.name);so=dst/so.name;so.chmod(0o555)
for host in ['sj9','sj10']:
 c=json.loads((r/'runs'/oldrun/host/'role_config.json').read_text());oldbinary=c['binary'];c.update(binary=str(binary),binary_sha256=sha,run_id=run,namespace='xiaopang1116-match-'+run.replace('_','-'),status='READY_NOT_STARTED',source_manifest='/expt/builds/attention_replica_v1/source_manifest.json',treatment='D replicated Attention TP1/H64; Engram/embed/head TP8 shared TP8 routed EP8; ordinary fullGraph, no DP scheduling/LL. D memory fraction .95; fixed cache still 1GiB.')
 for rank in c['ranks']:
  rank['command']=[a.replace(oldbinary,str(binary)).replace('/'+oldrun+'/','/'+run+'/').replace('xiaopang1116-match-'+oldrun.replace('_','-'),c['namespace']) for a in rank['command']]
  e=rank['environment'];e['XLLM_CPP_STACK_DIR']=f'/expt/profiling/{run}/{host}';e['XLLM_DSV41_REPLICATED_ATTENTION']='1' if host=='sj10' else '0';e['XLLM_V41_LIGHT_PREFIX']=''
  if host=='sj10':
   e['XLLM_DSV41_MATE_SPARSE_SO']=str(so)
   assert '--max_memory_utilization=0.9' in rank['command'];rank['command']=[a.replace('--max_memory_utilization=0.9','--max_memory_utilization=0.95') for a in rank['command']]
  assert e['XLLM_DSV41_DSPARK_NATIVE']=='0'
 d=out/host;d.mkdir();(d/'prepared_config.json').write_text(json.dumps(c,indent=2))
s=Path('/expt/scripts/match_fullgraph_client.py').read_text().replace('artifacts/full_decode_graph_v1/bin/xllm','artifacts/attention_replica_v1/bin/xllm').replace('builds/full_decode_graph_v1/source_manifest.json','builds/attention_replica_v1/source_manifest.json').replace('FULL_DECODE_GRAPH_SHORT_VALID_NO_LONG_REGRESSION_YET','ATTENTION_REPLICA_SHORT_VALID_NO_LONG_REGRESSION_YET');compile(s,'client','exec');Path('/expt/scripts/match_p3_client.py').write_text(s)
print(json.dumps({'run':run,'sha':sha,'h64_library':str(so),'prepared':True}))
