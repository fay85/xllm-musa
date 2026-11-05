"""Single-host smoke for exact prompt lengths and chunk boundaries, not PD performance."""
from pathlib import Path
import json,time,urllib.request
root=Path('/work');out=root/'runs/load_precheck_v2/prompt_length_smoke';out.mkdir(exist_ok=False);rows=[]
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for length in [256,2048]:
 body=json.loads((root/f'artifacts/benchmark_inputs_v1/input{length}_output256.jsonl').read_text().splitlines()[0]);body['max_tokens']=8
 (out/f'input{length}_request.json').write_text(json.dumps(body,indent=2));start=time.monotonic();usage=None;content=[];finish=None
 request=urllib.request.Request('http://10.20.34.28:24000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with opener.open(request,timeout=1200) as response,(out/f'input{length}_sse.jsonl').open('x') as log:
  for line in response:
   text=line.decode().rstrip('\r\n');log.write(json.dumps({'elapsed_s':time.monotonic()-start,'line':text})+'\n');log.flush()
   if not text.startswith('data:') or text[5:].strip()=='[DONE]':continue
   value=json.loads(text[5:].strip())
   if value.get('usage'):usage=value['usage']
   for choice in value.get('choices',[]):
    content.append(choice.get('delta',{}).get('content') or '')
    if choice.get('finish_reason'):finish=choice['finish_reason']
 row={'requested_prompt_tokens':length,'requested_output_tokens':8,'usage':usage,'content':''.join(content),'finish_reason':finish,'elapsed_s':time.monotonic()-start,'scope':'single-host prompt-length/chunking smoke, not 1P1D benchmark'};rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
 assert usage and usage['prompt_tokens']==length and usage['completion_tokens']==8 and ''.join(content),'Invalid actual token counts or empty generation'
(out/'PASS').write_text('Both exact input lengths completed eight decode tokens; no quality claim for truncated prose.\n')
