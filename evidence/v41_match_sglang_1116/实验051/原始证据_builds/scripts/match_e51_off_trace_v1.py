from pathlib import Path
import json,datetime,requests,time,concurrent.futures,hashlib,os
r=Path('/expt');run=r/'runs/p51_bit_gather_off_v1';o=run/'trace';assert json.loads((run/'formal_client/finished.json').read_text())['exit_code']==0;o.mkdir(exist_ok=False)
(o/'job.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()},indent=2))
body=json.loads(Path('/work/artifacts/benchmark_inputs_v1/input256_output256.jsonl').read_text().splitlines()[0]);body.update(max_tokens=8,stream=True,stream_options={'include_usage':True});(o/'request.json').write_text(json.dumps(body,ensure_ascii=False,indent=2));urls=['http://10.121.32.7:22000','http://10.121.32.9:32000'];results=[];rc=1
pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
def control(url,action):
 t=time.monotonic();res=requests.post(url+'/'+action,timeout=(10,300));return {'url':url,'action':action,'http_status':res.status_code,'text':res.text,'elapsed_s':time.monotonic()-t}
def request(label,stop_prefill=False):
 t=time.monotonic();usage=None;content='';first=None;first_any=None;fut=None;finish=None
 with (o/(label+'_sse.jsonl')).open('x') as f:
  with requests.post('http://10.121.32.9:31180/v1/chat/completions',json=body,stream=True,timeout=(10,180)) as res:
   res.raise_for_status()
   for line in res.iter_lines():
    if not line:continue
    stamp=time.monotonic()-t;s=line.decode();f.write(json.dumps({'elapsed_s':stamp,'raw':s},ensure_ascii=False)+'\n');f.flush()
    if not s.startswith('data:') or s[5:].strip()=='[DONE]':continue
    data=json.loads(s[5:]);first_any=stamp if first_any is None else first_any
    usage=data.get('usage') or usage
    for c in data.get('choices',[]):
     delta=c.get('delta',{}).get('content') or '';content+=delta;finish=c.get('finish_reason') or finish
     if delta and first is None:
      first=stamp
      if stop_prefill:fut=pool.submit(control,urls[0],'stop_profile')
 elapsed=time.monotonic()-t
 result={'label':label,'E2E_s':elapsed,'first_json_s':first_any,'first_content_s':first,'content':content,'usage':usage,'finish_reason':finish};results.append(result);(o/'requests.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));assert usage and usage['prompt_tokens']==256 and usage['completion_tokens']==8
 return fut
started=False;stops=[]
try:
 assert not list((r/'profiling/p51_bit_gather_off_v1').rglob('*.pt.trace.json'))
 request('off_before_1');request('off_before_2')
 starts=list(pool.map(lambda u:control(u,'start_profile'),urls));started=True;(o/'start_profile.json').write_text(json.dumps(starts,indent=2));assert all(x['http_status']==200 for x in starts)
 future=request('profiled',True)
 dstop=pool.submit(control,urls[1],'stop_profile')
 stops=[future.result() if future else control(urls[0],'stop_profile'),dstop.result()];started=False;(o/'stop_profile.json').write_text(json.dumps(stops,indent=2));assert all(x['http_status']==200 for x in stops)
 request('off_after');manifest=[]
 for host in ['sj9','sj10']:
  p=r/'profiling/p51_bit_gather_off_v1'/host;traces=list(p.glob('*.pt.trace.json'));stacks=list(p.glob('native_cpp_stacks_*.jsonl'));assert len(traces)==8 and len(stacks)==8,(host,len(traces),len(stacks))
  for f in traces+stacks:
   sha=hashlib.sha256()
   with f.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):sha.update(chunk)
   manifest.append({'host':host,'path':str(f),'bytes':f.stat().st_size,'sha256':sha.hexdigest()})
 (o/'manifest.json').write_text(json.dumps(manifest,indent=2));rc=0
except Exception as e:
 (o/'error.txt').write_text(repr(e));raise
finally:
 if started:
  stops=list(pool.map(lambda u:control(u,'stop_profile'),urls));(o/'stop_profile_cleanup.json').write_text(json.dumps(stops,indent=2))
 pool.shutdown(wait=True);(o/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat(),'status':'CAPTURED_REQUIRES_TRACE_EVENT_VALIDATION' if rc==0 else 'FAILED'},indent=2))
