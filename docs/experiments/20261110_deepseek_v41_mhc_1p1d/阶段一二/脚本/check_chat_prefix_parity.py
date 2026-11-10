"""Post-benchmark Chat probe. Never used as a performance sample.

First establish that raw IDs affect the input using a deliberately different
message and an elementary answer. Then compare next-token logprobs after the
same requested token prefix, generated through PD versus a fresh prefill.
"""
from pathlib import Path
import argparse
import datetime
import importlib.util
import json
import time
import urllib.error
import urllib.request

from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--url', default='http://10.121.32.9:39080/v1/chat/completions')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
model = Path('/work/artifacts/model_view_1017')
spec = importlib.util.spec_from_file_location('official_encoding', model / 'encoding/encoding.py')
encoding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoding)
tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def ids_for(question: str) -> list[int]:
    text = encoding.encode_messages([{'role': 'user', 'content': question}], thinking_mode='chat')
    if isinstance(text, tuple):
        text = text[0]
    return tokenizer.encode(text, add_special_tokens=False).ids


def request(label: str, ids: list[int], count: int, message: str) -> dict:
    body = {'model': 'deepseek-v4.1-flash-1017', 'messages': [{'role': 'user', 'content': message}],
            'token_ids': ids, 'add_special_tokens': False, 'temperature': 0, 'top_p': 1,
            'reasoning_effort': 'none', 'chat_template_kwargs': {'thinking_mode': 'chat'},
            'max_tokens': count, 'ignore_eos': True, 'logprobs': True, 'top_logprobs': 5,
            'include_stop_str_in_output': True, 'stream': True, 'stream_options': {'include_usage': True}}
    (args.output / f'{label}_request.json').write_text(json.dumps(body, ensure_ascii=False, indent=2))
    req = urllib.request.Request(args.url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    packets, probabilities, text = [], [], []
    usage, request_id = None, None
    try:
        with opener.open(req, timeout=120) as response, (args.output / f'{label}_sse.jsonl').open('x') as log:
            for raw in response:
                line = raw.decode().rstrip('\r\n')
                log.write(json.dumps({'elapsed_s': time.monotonic() - started, 'line': line}, ensure_ascii=False) + '\n')
                log.flush()
                if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                    continue
                packet = json.loads(line[5:].strip())
                packets.append(packet)
                request_id = packet.get('id', request_id)
                if packet.get('error'):
                    raise RuntimeError(packet['error'])
                usage = packet.get('usage') or usage
                for choice in packet.get('choices', []):
                    probabilities.extend((choice.get('logprobs') or {}).get('content', []))
                    text.append((choice.get('delta') or {}).get('content') or '')
    except urllib.error.HTTPError as error:
        (args.output / f'{label}_http_error.txt').write_bytes(error.read())
        raise
    record = {'id': request_id, 'usage': usage, 'logprobs': probabilities, 'text': ''.join(text),
              'elapsed_s': time.monotonic() - started, 'packets': packets}
    (args.output / f'{label}_response.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    assert usage and usage['prompt_tokens'] == len(ids), 'Raw input IDs were not honored, or input counting differs'
    assert usage['completion_tokens'] == count and len(probabilities) == count, 'Incomplete token logprobs'
    assert all(isinstance(x.get('token_id'), int) and isinstance(x.get('logprob'), (int, float)) for x in probabilities)
    return record


result = {'at': datetime.datetime.now().astimezone().isoformat(), 'status': 'RUNNING', 'checks': [],
          'scope': 'Requested raw IDs, returned token counts and semantic control; not a wire capture, hidden-state dump or full-vocabulary logit comparison.'}
try:
    question = '只输出计算结果，不要解释：1.3 + 2.9 = ?'
    # A gateway that ignores token_ids will answer this message instead.
    sentinel = request('raw_ids_semantic_control', ids_for(question), 4, '只输出计算结果，不要解释：7 × 8 = ?')
    result['semantic_control_text'] = sentinel['text']
    assert sentinel['text'].lstrip().startswith('4.2'), 'Raw IDs semantic control failed; do not compare numerical results'
    full = request('pd_four_steps', ids_for(question), 4, question)
    tokens = [x['token_id'] for x in full['logprobs']]
    for position in range(1, 4):
        fresh = request(f'prefill_prefix_{position}', ids_for(question) + tokens[:position], 1, question)
        old, new = full['logprobs'][position], fresh['logprobs'][0]
        result['checks'].append({'position': position, 'pd_token': old['token_id'], 'prefill_token': new['token_id'],
                                 'same_token': old['token_id'] == new['token_id'], 'pd_logprob': old['logprob'],
                                 'prefill_logprob': new['logprob'], 'logprob_abs_difference': abs(old['logprob'] - new['logprob']),
                                 'pd_request_id': full['id'], 'prefill_request_id': fresh['id']})
    result['status'] = 'PASS_TOKEN_AGREEMENT' if all(x['same_token'] for x in result['checks']) else 'TOKEN_MISMATCH'
except (AssertionError, RuntimeError, ValueError, KeyError, OSError) as error:
    result['status'] = 'PROBE_FAILED'
    result['error'] = repr(error)
(args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if result['status'] == 'PASS_TOKEN_AGREEMENT' else 1)
