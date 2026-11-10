from pathlib import Path
import json,sqlite3,urllib.request,urllib.error,time,datetime
root=Path('/work');out=root/'02_correctness/stage1_swiglu_d_client_diagnostic';out.mkdir(exist_ok=False)
p=next((root/'03_benchmarks/stage1_swiglu_d/input256_output256/warmup').rglob('benchmark_data.db'))
with sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True) as c:body=json.loads(c.execute('select request from result limit 1').fetchone()[0])
(out/'request.json').write_text(json.dumps(body,ensure_ascii=False,indent=2));start=time.monotonic();result={'started_at':datetime.datetime.now().astimezone().isoformat()}
req=urllib.request.Request('http://10.121.32.9:39080/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
 with opener.open(req,timeout=600) as r,(out/'response.jsonl').open('w') as f:
  result['http_status']=r.status
  for line in r:
   f.write(json.dumps({'elapsed':time.monotonic()-start,'line':line.decode(errors='replace')},ensure_ascii=False)+'\n');f.flush()
except urllib.error.HTTPError as e:
 result['http_status']=e.code;result['body']=e.read().decode(errors='replace')
except Exception as e:result['exception']=repr(e)
result['elapsed']=time.monotonic()-start;(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
