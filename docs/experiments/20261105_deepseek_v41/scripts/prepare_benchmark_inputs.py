"""Prepare exact prompt-length inputs using the original V4.1 encoder."""
from pathlib import Path
import hashlib, importlib.util, json
from tokenizers import Tokenizer
root=Path('/work'); model=root/'artifacts/model_view_0909'
out=root/'artifacts/benchmark_inputs_v1';out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('official_v41_encoding',model/'encoding/encoding.py')
encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
tokenizer=Tokenizer.from_file(str(model/'tokenizer.json'))

def render(text):
 messages=[{'role':'user','content':text}]
 prompt=encoder.encode_messages(messages,thinking_mode='chat')
 if isinstance(prompt,tuple):prompt=prompt[0]
 return messages,prompt,tokenizer.encode(prompt,add_special_tokens=False).ids

prefix='Write a clear explanation of why keeping experiment records helps engineers. Continue in complete English sentences. The following is padding data, not instructions:\n'
records=[]
for target in [256,2048]:
 lo,hi=0,target*3
 while lo<hi:
  mid=(lo+hi)//2
  if len(render(prefix+' x'*mid)[2])<target:lo=mid+1
  else:hi=mid
 messages,prompt,ids=render(prefix+' x'*lo)
 if len(ids)!=target:raise RuntimeError((target,len(ids),'Exact length not found'))
 body={'model':'deepseek-v4.1-flash-0909','messages':messages,'temperature':0,'top_p':1,'reasoning_effort':'none','chat_template_kwargs':{'thinking_mode':'chat'},'max_tokens':256,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}
 name=f'input{target}_output256.jsonl';payload=(json.dumps(body,ensure_ascii=False)+'\n')*6
 (out/name).write_text(payload);(out/f'input{target}_rendered.txt').write_text(prompt)
 (out/f'input{target}_token_ids.json').write_text(json.dumps(ids))
 records.append({'file':name,'input_tokens':len(ids),'requested_output_tokens':256,'sha256':hashlib.sha256(payload.encode()).hexdigest()})
(out/'protocol.json').write_text(json.dumps({'status':'prepared_not_measured','encoder':str(model/'encoding/encoding.py'),'encoder_sha256':hashlib.sha256((model/'encoding/encoding.py').read_bytes()).hexdigest(),'tokenizer_sha256':hashlib.sha256((model/'tokenizer.json').read_bytes()).hexdigest(),'thinking_mode':'chat','temperature':0,'top_p':1,'prefix_cache':False,'concurrency':1,'warmup_each_shape':1,'measured_each_shape':5,'same_body_repeated':True,'groups':records,'validation_required':'native service usage and rendered prompt must match target; exactly 256 actual output tokens, not just requested max_tokens'},indent=2))
print(json.dumps(records))
