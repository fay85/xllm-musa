"""Record concrete native TP/EP ranks and nonempty PD pulls from every D rank."""
from pathlib import Path
import ast,json,re,argparse,hashlib,datetime
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();rows=[];snapshot=a.run/'pd_evidence_logs';snapshot.mkdir(exist_ok=False)
for host in ('sj9','sj10'):
 cfg=json.loads((a.run/host/'role_config.json').read_text());assert len(cfg['ranks'])==8 and cfg['TP']==8 and cfg['EP']==8 and cfg['MoE_TP']==1
 for rank in cfg['ranks']:
  f=a.run/host/f"rank{rank['rank']}.log";raw=f.read_bytes();s=raw.decode(errors='replace');target=snapshot/f"{host}_rank{rank['rank']}.log";target.write_bytes(raw);assert '--enable_graph=true' not in str(rank['command'])
  selected_shared=re.findall(r'V4.1 MUSA shared SwiGLU: mode=([^,]+),',s)
  expected_shared=rank['environment'].get('XLLM_DSV41_MUSA_SHARED_SWIGLU','reference')
  assert selected_shared and set(selected_shared)=={expected_shared},f'{host} rank {rank["rank"]}: shared SwiGLU dispatch not confirmed'
  mhc_selections=[ast.literal_eval(line.split('V4.1 MUSA mHC: ',1)[1]) for line in s.splitlines() if 'V4.1 MUSA mHC: ' in line]
  if 'stage2_mhc' in cfg.get('source_dir',''):
   expected_mhc={k:rank['environment'].get('XLLM_DSV41_MUSA_HC_'+k.upper(),'reference') for k in ('pre','post','metadata')}
   assert mhc_selections and all(x==expected_mhc for x in mhc_selections),f'{host} rank {rank["rank"]}: mHC dispatch not confirmed'
  pulls=[]
  for line in s.splitlines():
   if '[Mooncake][PDTransfer]' in line and 'direction=pull' in line and 'success=true' in line:
    m=re.search(r', regions=(\d+),',line)
    if m and int(m.group(1))>0:pulls.append(line)
  if host=='sj10':assert len(pulls)>=2,f'No nonempty successful PD transfers for D rank {rank["rank"]}'
  rows.append({'host':host,'rank':rank['rank'],'selected_shared_swiglu':expected_shared,'shared_selection_records':len(selected_shared),'mhc_selection':mhc_selections[0] if mhc_selections else None,'mhc_selection_records':len(mhc_selections),'nonempty_successful_pulls':len(pulls),'first_two_pulls':pulls[:2],'log_snapshot':str(target.relative_to(a.run)),'source_live_path':str(f),'log_sha256':hashlib.sha256(raw).hexdigest(),'configured_nic_filter':rank['environment'].get('MC_TE_FILTERS'),'model_path':cfg['model'],'binary_sha256':cfg['binary_sha256'],'python_source_flag':[x for x in rank['command'] if x.startswith('--python_model_path=')]})
record={'status':'PASS_EIGHT_D_RANKS_RECEIVED_NONEMPTY_PD_STATE','at':datetime.datetime.now().astimezone().isoformat(),'attention_TP':8,'routed_EP':8,'expert_TP':1,'shared_expert':'replicated','mode':'1P1D eager','scope':'rank configurations and successful nonempty state transfer; not layerwise numerical equivalence','rows':rows}
(a.run/'pd_evidence.json').write_text(json.dumps(record,indent=2));print(json.dumps({'status':record['status'],'D_rank_pulls':[{k:r[k] for k in ['rank','nonempty_successful_pulls']} for r in rows if r['host']=='sj10']}))
