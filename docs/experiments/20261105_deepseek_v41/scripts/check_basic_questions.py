"""Two explicit basic-answer checks before performance; preserve full SSE."""
import argparse, datetime, decimal, hashlib, json, time, urllib.request
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--binary',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
results=[]
for index,question,expected in [(1,'只输出计算结果，不要解释：1.3 + 2.9 = ?',decimal.Decimal('4.2')),(2,'只输出计算结果，不要解释：7 × 8 = ?',decimal.Decimal('56'))]:
 body={'model':'deepseek-v4.1-flash-0909','messages':[{'role':'user','content':question}],'temperature':0,'top_p':1,'reasoning_effort':'none','chat_template_kwargs':{'thinking_mode':'chat'},'max_tokens':96,'stream':True,'stream_options':{'include_usage':True}}
 (args.output/f'question{index}_request.json').write_text(json.dumps(body,ensure_ascii=False,indent=2));start=time.perf_counter();parts=[];reasoning=[];usage=None;finish=None
 request=urllib.request.Request(args.url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with (args.output/f'question{index}_sse.jsonl').open('x') as log,opener.open(request,timeout=1200) as response:
  for line in response:
   text=line.decode('utf-8').rstrip('\r\n');log.write(json.dumps({'elapsed_s':time.perf_counter()-start,'line':text},ensure_ascii=False)+'\n');log.flush()
   if not text.startswith('data:'):continue
   value=text[5:].strip()
   if value=='[DONE]':continue
   packet=json.loads(value)
   if packet.get('usage'):usage=packet['usage']
   for choice in packet.get('choices',[]):
    delta=choice.get('delta',{});parts.append(delta.get('content') or '');reasoning.append(delta.get('reasoning_content') or '')
    if choice.get('finish_reason'):finish=choice['finish_reason']
 content=''.join(parts).strip()
 try:passed=decimal.Decimal(content)==expected and finish!='length'
 except decimal.InvalidOperation:passed=False
 results.append({'id':index,'question':question,'expected':str(expected),'content':content,'reasoning_content':''.join(reasoning),'finish_reason':finish,'usage':usage,'elapsed_s':time.perf_counter()-start,'passed':passed})
 (args.output/'answers.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
 if not passed:raise SystemExit('Basic answer failed or non-numeric format: inspect raw response before performance')
record={'status':'PASS','scope':'two elementary questions only; not broad model accuracy','questions':results,'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),'checked_at':datetime.datetime.now().astimezone().isoformat(),'endpoint':args.url}
(args.output/'basic_gate.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));print(json.dumps(record,ensure_ascii=False))
