"""Recompute frozen B metrics from original EvalScope rows; never use chunk count as tokens."""
from pathlib import Path
import argparse,base64,hashlib,io,json,pickle,sqlite3,statistics
class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise ValueError('Only primitive response records are supported')
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
r=a.root;identity=json.loads((r/'identity.json').read_text());summary=json.loads((r/'summary.json').read_text())
assert summary['profiler_on'] is False and identity['dataset_sha']=='c3d7cb38060a2f81b228b83d8939e1a682480e9ab26a2c5a2df29ca66dadbc51'
assert identity['tokenizer_sha']=='8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf'
assert json.loads((r/'basic_gate.json').read_text())['passed']
assert json.loads((r/'compilation_window.json').read_text())['all_clear'], 'Compilation overlap or missing monitoring'
assert all(x['same_ids'] and len(x['expected_token_ids'])==256 for x in json.loads((r/'tokenization_preflight.json').read_text()))
checks={}
for phase,n in [('warmup',1),('measured',5)]:
    files=list((r/phase).rglob('benchmark_data.db'));assert len(files)==1
    with sqlite3.connect('file:'+str(files[0])+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row;rows=[dict(x) for x in db.execute('SELECT * FROM result')]
    assert len(rows)==n
    out=[]
    for row in rows:
        assert row['success']==1 and row['prompt_tokens']==row['completion_tokens']==256
        chunks=PrimitiveUnpickler(io.BytesIO(base64.b64decode(row['response_messages']))).load()
        ids={x['id'] for x in chunks if x.get('id')};assert len(ids)==1
        reasons=[c['finish_reason'] for x in chunks for c in x.get('choices',[]) if c.get('finish_reason')]
        assert reasons==['length'],reasons
        text=''.join(c.get('delta',{}).get('content') or '' for x in chunks for c in x.get('choices',[]))
        out.append({'id':next(iter(ids)),'TTFT_ms':row['first_chunk_latency']*1000,'TPOT_ms':(row['latency']-row['first_chunk_latency'])*1000/255,'E2E_s':row['latency'],'SSE_packets':len(chunks),'content_sha256':hashlib.sha256(text.encode()).hexdigest()})
    checks[phase]=out
rows=checks['measured'];m={'mean_TTFT_ms':statistics.mean(x['TTFT_ms'] for x in rows),'mean_TPOT_ms_per_token':statistics.mean(x['TPOT_ms'] for x in rows),'mean_E2E_seconds':statistics.mean(x['E2E_s'] for x in rows),'output_tokens_per_latency_sum_second':1280/sum(x['E2E_s'] for x in rows)}
assert all(abs(v-summary[k])<1e-8 for k,v in m.items())
assert len({x['id'] for records in checks.values() for x in records})==6
result={'status':'FORMAL_B_ROWS_RECOMPUTED','metrics':m,'requests':checks,'token_accounting':'EvalScope server usage updates token counters independently of stored content chunks; input IDs also checked with server /v1/tokenize, each response ends with length. SSE count is not token count.','accuracy_status':'仅通过基础功能检查，未完成模型精度验收'}
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(m))

