from pathlib import Path
import requests,json,time
r=Path('/expt/runs/p51_bit_gather_on_v1');assert json.loads((r/'long_smoke/finished.json').read_text())['exit_code']==0;o=r/'post_long_checks';o.mkdir(exist_ok=False);base=json.loads(Path('/work/artifacts/benchmark_inputs_v1/input256_output256.jsonl').read_text().splitlines()[0]);results=[]
for name,prompt,answer,tokens in [('arithmetic1','只回答计算结果：1.3＋2.9 等于多少？','4.2',16),('arithmetic2','只回答计算结果：7×8 等于多少？','56',16),('short_bucket_return',None,None,8)]:
 b=dict(base);b.update(stream=True,max_tokens=tokens,ignore_eos=prompt is None,stream_options={'include_usage':True})
 if prompt:b['messages']=[{'role':'user','content':prompt}]
 content='';usage=None;t=time.monotonic()
 with (o/(name+'_sse.jsonl')).open('x') as f:
  with requests.post('http://10.121.32.9:31180/v1/chat/completions',json=b,stream=True,timeout=(10,90)) as resp:
   resp.raise_for_status()
   for line in resp.iter_lines():
    if not line:continue
    raw=line.decode();f.write(json.dumps({'elapsed_s':time.monotonic()-t,'raw':raw})+'\n')
    if raw.startswith('data:') and raw[5:].strip()!='[DONE]':
     d=json.loads(raw[5:]);usage=d.get('usage') or usage
     for c in d.get('choices',[]):content+=c.get('delta',{}).get('content') or ''
 passed=content.strip()==answer if answer else usage and usage['prompt_tokens']==256 and usage['completion_tokens']==8
 results.append({'name':name,'content':content,'usage':usage,'passed':bool(passed),'elapsed_s':time.monotonic()-t});(o/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));assert passed
print(json.dumps(results,ensure_ascii=False))
