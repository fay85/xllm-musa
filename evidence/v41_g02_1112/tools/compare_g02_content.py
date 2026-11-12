# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys,sqlite3,json,base64,pickle,io,hashlib
W=Path('/home/xiaopang/桌面/Albert分享/20261021从xllm开始');P=W/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261112新的开始'
class Safe(pickle.Unpickler):
 def find_class(self, module: str, name: str) -> object:raise ValueError('global objects disabled')
 def persistent_load(self, pid: object) -> object:raise ValueError('persistent objects disabled')
def data(mode: str, n: int) -> list[dict]:
 e=P/('ToDoList_3' if mode=='DEVICE' else 'ToDoList_2');f=next((e/'04_性能'/mode/f'raw/performance/input{n}_output256/measured').rglob('benchmark_data.db'))
 with sqlite3.connect(f) as db:
  db.row_factory=sqlite3.Row;rs=[dict(r) for r in db.execute('select request,response_messages from result order by start_time')]
 out=[]
 for r in rs:
  msgs=Safe(io.BytesIO(base64.b64decode(r['response_messages']))).load();text='';reason=''
  for x in msgs:
   for c in x.get('choices',[]):
    d=c.get('delta') or c.get('message') or {};text+=d.get('content') or '';reason+=d.get('reasoning_content') or ''
  out.append({'request':json.loads(r['request']),'content':text,'reasoning':reason,'content_sha256':hashlib.sha256(text.encode()).hexdigest()})
 return out
report=[]
for n in [256,2048]:
 before=data('OFF',n)
 for mode in ['HOST','DEVICE']:
  after=data(mode,n)
  for i,(b,a) in enumerate(zip(before,after)):
   assert b['request']==a['request'],(mode,n,i,'input_changed')
   mismatch=next((k for k,(v,w) in enumerate(zip(b['content'],a['content'])) if v!=w),None)
   if mismatch is None and len(b['content'])!=len(a['content']):mismatch=min(len(b['content']),len(a['content']))
   report.append({'mode':mode,'input_tokens':n,'request_index':i,'exact_content_equal':b['content']==a['content'],'reasoning_equal':b['reasoning']==a['reasoning'],'before_sha256':b['content_sha256'],'after_sha256':a['content_sha256'],'first_character_difference':mismatch,'before_near_difference':b['content'][max(0,(mismatch or 0)-30):(mismatch or 0)+60] if mismatch is not None else None,'after_near_difference':a['content'][max(0,(mismatch or 0)-30):(mismatch or 0)+60] if mismatch is not None else None,'scope':'Decoded content equality, not raw logits or hidden special-token equality'})
(E:=P/'ToDoList_3/03_正确性/G02相同请求输出核对.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'comparisons':len(report),'exact_content_matches':sum(x['exact_content_equal'] for x in report),'differences':[x for x in report if not x['exact_content_equal']]},ensure_ascii=False,indent=2))
