"""Compare the same prefix at P and after PD continuation using raw token IDs.

Only run while the benchmark is idle. This is a logits-distribution probe,
not a hidden-state dump or a broad model accuracy test.
"""
from pathlib import Path
import argparse,json,time,urllib.request,importlib.util,math
from tokenizers import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--url',default='http://10.121.32.9:39080/v1/completions');a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
model=Path('/work/artifacts/model_view_1017');spec=importlib.util.spec_from_file_location('official_encoding',model/'encoding/encoding.py');encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder);tok=Tokenizer.from_file(str(model/'tokenizer.json'))
text=encoder.encode_messages([{'role':'user','content':'只输出计算结果，不要解释：1.3 + 2.9 = ?'}],thinking_mode='chat');text=text[0] if isinstance(text,tuple) else text
ids=tok.encode(text,add_special_tokens=False).ids
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def request(label,ids,count):
 body={'model':'deepseek-v4.1-flash-1017','prompt':tok.decode(ids,skip_special_tokens=False),'token_ids':ids,'add_special_tokens':False,'max_tokens':count,'ignore_eos':True,'temperature':0,'top_p':1,'stream':False,'echo':False,'skip_special_tokens':False,'include_stop_str_in_output':True,'logprobs':5}
 (a.output/f'{label}_request.json').write_text(json.dumps(body));start=time.monotonic()
 req=urllib.request.Request(a.url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 try:
  with opener.open(req,timeout=300) as r:response=json.loads(r.read())
 except urllib.error.HTTPError as e:
  (a.output/f'{label}_http_error.txt').write_bytes(e.read());raise
 (a.output/f'{label}_response.json').write_text(json.dumps(response,ensure_ascii=False,indent=2));print(label,time.monotonic()-start,flush=True);return response
result={'status':'RUNNING','scope':'same-prefix output token/logprob comparison; no hidden-state comparison','prompt_token_ids':ids,'checks':[]};(a.output/'result.json').write_text(json.dumps(result))
try:
 full=request('pd_four_steps',ids,4);choice=full['choices'][0];lp=choice['logprobs'];generated=lp['token_ids'];scores=lp['token_logprobs'];assert len(generated)==len(scores)==4;assert full['usage']['prompt_tokens']==len(ids)
 for pos in range(1,4):
  single=request(f'prefill_prefix_{pos}',ids+generated[:pos],1);q=single['choices'][0]['logprobs'];assert single['usage']['prompt_tokens']==len(ids)+pos
  row={'position':pos,'pd_token':generated[pos],'prefill_token':q['token_ids'][0],'pd_logprob':scores[pos],'prefill_logprob':q['token_logprobs'][0]};row['same_token']=row['pd_token']==row['prefill_token'];row['logprob_abs_difference']=abs(row['pd_logprob']-row['prefill_logprob']);result['checks'].append(row)
 result['status']='PASS_TOKEN_AGREEMENT' if all(x['same_token'] for x in result['checks']) else 'TOKEN_MISMATCH'
except Exception as e:result['status']='PROBE_FAILED';result['error']=repr(e)
(a.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False));raise SystemExit(0 if result['status']=='PASS_TOKEN_AGREEMENT' else 1)
