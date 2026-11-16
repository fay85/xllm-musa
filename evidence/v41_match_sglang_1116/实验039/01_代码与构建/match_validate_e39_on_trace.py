from pathlib import Path
import re,json,collections,subprocess,hashlib,os,datetime,shutil,tarfile
r=Path('/expt');run=r/'runs/p39_static_activation_on_v1';o=run/'trace_analysis';o.mkdir(exist_ok=True);rows=[];offsets=set();selected=[]
for host in ['sj9','sj10']:
 done=json.loads((run/host/'result.json').read_text());assert done['exit_codes']==[0]*8,done
 d=r/'profiling/p39_static_activation_on_v1'/host
 for f in sorted(d.glob('*.pt.trace.json')):
  data=json.loads(f.read_text());ev=data['traceEvents'];kernels=[x for x in ev if x.get('ph')=='X' and 'kernel' in x.get('cat','').lower()];steps=[x for x in ev if x.get('ph')=='X' and x.get('cat') in ['cpu_op','user_annotation'] and x.get('name','').startswith('V41/step/')];aten=[x for x in ev if x.get('ph')=='X' and x.get('name','').startswith('aten::')];assert kernels and steps and aten,(host,f.name)
  pids={str(x['pid']) for x in steps};assert len(pids)==1;p=next(iter(pids));sf=d/('native_cpp_stacks_'+p+'.jsonl');assert sf.exists(),(p,[x.name for x in d.glob('*.jsonl')]);rf={x['args']['Record function id']:x for x in ev if 'Record function id' in x.get('args',{})};calls=[];stackmap={};matched=0
  with sf.open() as stream:
   for line in stream:
    item=json.loads(line)
    if item['kind']=='stack':stackmap[item['stack_id']]=item['frames']
    elif item['kind']=='record' and item['record_id'] in rf:
     e=rf[item['record_id']];assert e['name']==item['name'] and e['tid']==item['os_thread'];matched+=1
  row={'host':host,'file':str(f),'sidecar':str(sf),'pid':int(p),'events':len(ev),'gpu_kernels':len(kernels),'aten_calls':len(aten),'steps':[{'name':x['name'],'ts':x['ts'],'dur':x['dur']} for x in steps],'matched_native_record_ids':matched,'top_aten_counts':collections.Counter(x['name'] for x in aten).most_common(15)};rows.append(row)
  graphs=[e for e in ev if e.get('ph')=='X' and e.get('name')=='musaGraphLaunch']
  per_graph=[]
  for graph in graphs:
   corr=graph['args']['correlation'];nodes=[e for e in kernels if e['args'].get('correlation')==corr];assert nodes
   by_name={}
   for n in nodes:
    v=by_name.setdefault(n['name'],{'count':0,'sum_us':0,'max_us':0});v['count']+=1;v['sum_us']+=n['dur'];v['max_us']=max(v['max_us'],n['dur'])
   intervals=sorted((n['ts'],n['ts']+n['dur']) for n in nodes);start,end=intervals[0];union=0
   for a,b in intervals[1:]:
    if a>end:union+=end-start;start,end=a,b
    else:end=max(end,b)
   union+=end-start
   per_graph.append({'correlation':corr,'launch_ts':graph['ts'],'launch_duration_us':graph['dur'],'kernels':len(nodes),'span_us':intervals[-1][1]-intervals[0][0],'kernel_interval_union_us':union,'sum_kernel_us':sum(n['dur'] for n in nodes),'top_kernels_by_sum_us':sorted(({'name':k,**v} for k,v in by_name.items()),key=lambda v:v['sum_us'],reverse=True),'stream_counts':dict(collections.Counter(n['args']['stream'] for n in nodes))})
  row['graph_replays']=per_graph
  row['refresh_lines']=[line for line in (run/host/('rank'+str(json.loads((run/host/'status.json').read_text())['pids'].index(int(p)))+'.log')).read_text(errors='replace').splitlines() if 'native MUSA graph' in line]
  if host=='sj10':
   assert len(per_graph)==len(steps)>0 and all(g['kernels']>1000 for g in per_graph); assert all('/q=4/' in step['name'] for step in steps)
   refreshed=[int(m.group(1)) for x in row['refresh_lines'] if (m:=re.search(r'Refreshed (\d+) native MUSA graphs',x))]
   restored=[int(m.group(1)) for x in row['refresh_lines'] if (m:=re.search(r'Restored (\d+) native MUSA graph',x))]
   assert len(refreshed)==len(restored)==1 and refreshed==restored and refreshed[0]>0
   row['refreshed_graph_count']=refreshed[0]
  cfg=json.loads((run/host/'status.json').read_text())
  if int(p)==cfg['pids'][0]:
   dest=o/host;dest.mkdir(exist_ok=True);shutil.copy2(f,dest/'rank0.pt.trace.json');shutil.copy2(sf,dest/'rank0.native_stacks.jsonl');selected.append({'host':host,'trace':str(dest/'rank0.pt.trace.json'),'sidecar':str(dest/'rank0.native_stacks.jsonl')})
   for frames in stackmap.values():
    offsets.update(x['module_offset'] for x in frames if x['module'].endswith('/bin/xllm'))
assert len(rows)==16 and len(selected)==2
binary=r/'artifacts/native_sg438_mccl_v1/bin/xllm';sha=hashlib.sha256(binary.read_bytes()).hexdigest();assert sha=='6468707445ac26b74f6e8766dd20b13b5567e367a3d172d482aff777b0d09293';ordered=sorted(offsets);lines=subprocess.check_output(['addr2line','-C','-f','-e',str(binary)]+[hex(x) for x in ordered],text=True).splitlines();assert len(lines)==len(ordered)*2;symbols={str(x):lines[i*2:i*2+2] for i,x in enumerate(ordered)};assert any('DeepseekV41' in x[0] for x in symbols.values());(o/'resolved_symbols.json').write_text(json.dumps(symbols,indent=2));(o/'all_rank_validation.json').write_text(json.dumps({'status':'ALL16_GPU_NATIVE_VALID_TARGET_GRAPH_KERNELS_CORRELATED','binary_sha256':sha,'selected':selected,'ranks':rows,'profile_workload':[256,8],'scope':'Counts and paths; heavy profiler changes timing, do not infer unprofiled Amdahl fractions'},indent=2))

archive=r/'exports/p39_static_activation_on_v1_evidence.tar.gz';archive.parent.mkdir(exist_ok=True)
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
 for sub in ['formal_client','trace','trace_analysis','long_smoke','post_long_checks']:tar.add(run/sub,arcname=sub)
 for host in ['sj9','sj10']:
  for name in ['role_config.json','runtime_identity.json','status.json','result.json','rank0.log','supervisor_job.json']:tar.add(run/host/name,arcname='roles/'+host+'/'+name)
 for name in ['native_artifact.json','source_manifest.json','full_v2_build.log','full_v2_job.json','full_v2_finished.json','ninja_v2_delta.json']:
  p=r/'builds/native_sg438_control/e37_native_mccl'/name
  if p.exists():tar.add(p,arcname='build/'+name)
v={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'at':datetime.datetime.now().astimezone().isoformat()};archive.with_suffix('.receipt.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
