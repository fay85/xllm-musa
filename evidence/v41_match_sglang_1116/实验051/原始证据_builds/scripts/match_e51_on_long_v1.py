from pathlib import Path
import json,requests,time,os,datetime
r=Path('/expt/runs/p51_bit_gather_on_v1');o=r/'long_smoke';assert json.loads((r/'formal_client/finished.json').read_text())['exit_code']==0;assert json.loads((r/'formal_client/finished.json').read_text())['exit_code']==0;o.mkdir(exist_ok=False)
(o/'job.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()}));body=json.loads(Path('/work/artifacts/benchmark_inputs_v1/input2048_output256.jsonl').read_text().splitlines()[0]);body.update(stream=True,stream_options={'include_usage':True});(o/'request.json').write_text(json.dumps(body));rc=1
try:
 t=time.monotonic();first=None;usage=None;content='';finish=None
 with (o/'response_sse.jsonl').open('x') as f:
  with requests.post('http://10.121.32.9:31180/v1/chat/completions',json=body,stream=True,timeout=(10,180)) as res:
   res.raise_for_status()
   for line in res.iter_lines():
    if not line:continue
    elapsed=time.monotonic()-t;raw=line.decode();f.write(json.dumps({'elapsed_s':elapsed,'raw':raw})+'\n');f.flush()
    if not raw.startswith('data:') or raw[5:].strip()=='[DONE]':continue
    data=json.loads(raw[5:]);usage=data.get('usage') or usage
    for ch in data.get('choices',[]):
     delta=ch.get('delta',{}).get('content') or '';content+=delta;finish=ch.get('finish_reason') or finish
     if delta and first is None:first=elapsed
 elapsed=time.monotonic()-t;result={'latency_s':elapsed,'first_content_s':first,'usage':usage,'content':content,'finish_reason':finish,'scope':'One bounded long-shape cache/Graph bucket smoke, not formal performance round'};(o/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));assert usage and usage['prompt_tokens']==2048 and usage['completion_tokens']==256;rc=0
except Exception as e:(o/'error.txt').write_text(repr(e));raise
finally:(o/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
