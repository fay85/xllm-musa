"""Audit stopped P/D rank logs and preserve the earlier basic-gate snapshots."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,re
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);a=parser.parse_args();root=a.run
assert (root/'performance_v2/client_runs_finished.json').exists()
basic=json.loads((root/'basic_pd_path_evidence.json').read_text());assert basic['status']=='PASS_BASIC_PD_PATH'
snap=root/'basic_questions/server_log_snapshots';snap.mkdir(exist_ok=True)
rows=[]
for item in basic['rows']:
 host=item['host'];rank=item['rank'];p=root/host/f'rank{rank}.log';payload=p.read_bytes();text=payload.decode();lines=text.splitlines()
 result=json.loads((root/host/'result.json').read_text());assert result['exit_codes']==[0]*8
 archive=json.loads((root/host/'final_archive.json').read_text());assert archive['container_state']['ExitCode']==0 and not archive['container_state']['Running']
 assert 'transport retry counter exceeded' not in text and 'Pull KV byte regions failed' not in text and 'Traceback (most recent call last)' not in text
 pulls=[line for line in lines if '[Mooncake][PDTransfer] direction=pull,' in line and re.search(r'\bregions=[1-9][0-9]*,',line) and 'success=true' in line]
 if host==basic['decode']:assert len(pulls)==14,(host,rank,len(pulls))
 digest=hashlib.sha256();offset=0;found=False
 for line in payload.splitlines(keepends=True):
  digest.update(line);offset+=len(line)
  if digest.hexdigest()==item['log_sha256']:
   (snap/f'{host}_rank{rank}.log').write_bytes(payload[:offset]);found=True;break
 assert found,('Historical basic-gate log snapshot not reconstructed',host,rank)
 rows.append({'host':host,'rank':rank,'final_log':str(p.relative_to(root)),'final_log_sha256':hashlib.sha256(payload).hexdigest(),'successful_nonempty_pulls':len(pulls),'basic_snapshot_bytes':offset,'basic_snapshot_sha256':item['log_sha256'],'rank_exit_code':0})
ptext=(root/basic['prefill']/'rank0.log').read_text();dtext=(root/basic['decode']/'rank0.log').read_text()
ps=[x for x in ptext.splitlines() if 'record_type: pd_prefill_stage,' in x]
ds=[x for x in dtext.splitlines() if 'record_type: sequence_terminal,' in x]
term=[x for x in dtext.splitlines() if 'record_type: request_terminal,' in x and 'outcome: success, status_code: 0' in x]
def fields(line):return dict(re.findall(r'([a-z_]+): ([^,]+)',line))
ps=[fields(x) for x in ps];ds=[fields(x) for x in ds];term=[fields(x) for x in term]
assert len(ps)==len(ds)==len(term)==14,(len(ps),len(ds),len(term))
ids={x['request_id'] for x in ps};assert len(ids)==14 and ids=={x['request_id'] for x in ds}=={x['request_id'] for x in term}
counts=collections.Counter((int(x['prompt_tokens']),int(x['generated_tokens']),x['finish_reason']) for x in ds)
assert counts==collections.Counter({(21,4,'stop'):1,(17,2,'stop'):1,(256,256,'length'):6,(2048,256,'length'):6}),counts
report={'status':'PASS_FULL_RUN_PD_AND_NORMAL_EXIT','at':datetime.datetime.now().astimezone().isoformat(),'request_count':14,'basic_questions':2,'warmup_requests':2,'measured_requests':10,'nonempty_pulls_per_decode_rank':14,'normal_rank_exits':16,'normal_container_exits':2,'request_shape_counts':[{'input_tokens':k[0],'output_tokens':k[1],'finish_reason':k[2],'count':v} for k,v in sorted(counts.items())],'rank_log_evidence':rows,'scope':'Successful true two-host request path and clean process exit; not a per-layer official numerical-equivalence audit.'}
(root/'final_pd_path_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
