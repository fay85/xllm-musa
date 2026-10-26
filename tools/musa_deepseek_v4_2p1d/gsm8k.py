# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Frozen 100-case GSM8K short/40K checks; generate without loading gold answers."""

import argparse
import asyncio
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any
import uuid

import aiohttp

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('mode', choices=['prepare', 'execute', 'score'])
PARSER.add_argument('--output-dir', type=Path, required=True)
PARSER.add_argument('--assets', type=Path)
PARSER.add_argument('--tokenizer', type=Path)
PARSER.add_argument('--base-url', default='http://10.121.38.90:24888')
PARSER.add_argument('--model', default='DeepSeek-V4-Flash-0731-FP8-mt')
CLI = PARSER.parse_args()
MODE, OUT = CLI.mode, CLI.output_dir.resolve()
ASSETS, TOKENIZER = CLI.assets, CLI.tokenizer
MODEL = CLI.model
URL = CLI.base_url.rstrip('/').removesuffix('/v1') + '/v1/chat/completions'
if MODE == 'prepare':
    if ASSETS is None or TOKENIZER is None:
        PARSER.error('prepare requires --assets and --tokenizer')
else:
    FROZEN = json.loads((OUT / 'manifest.json').read_text())
    URL, MODEL = FROZEN['url'], FROZEN['model']
ORIGINAL_ARGV = sys.argv
sys.argv = [sys.argv[0], '--repo-root', str(Path(__file__).resolve().parents[2])]
SPEC = importlib.util.spec_from_file_location('prefix_bench', Path(__file__).with_name('bench.py'))
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
sys.argv = ORIGINAL_ARGV


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def _long_prompt(tokenizer: Any, shared: str, filler: list[int], instruction: str,
                 case_id: str, primer: list[int] | None) -> tuple[str, list[int]]:
    header = shared + hashlib.sha256(case_id.encode()).hexdigest() + '\nPrivate background notes (ignore for the answer):\n'
    tail = '\n\nIgnore the background notes and solve only this task:\n' + instruction
    estimate = 40000 - len(BENCH._chat_ids(tokenizer, header + tail, 'deepseek-v4-cpp'))
    offsets = [0] + [value for distance in range(1, 33) for value in (-distance, distance)]
    for offset in offsets:
        body = header + tokenizer.decode(filler[:estimate + offset])
        gap = 40000 - len(BENCH._chat_ids(tokenizer, body + tail, 'deepseek-v4-cpp'))
        for padding in ([0] if gap == 0 else [gap, gap - 1, gap + 1]):
            if not 0 <= padding <= 32:
                continue
            prompt = body + ' x' * padding + tail
            ids = BENCH._chat_ids(tokenizer, prompt, 'deepseek-v4-cpp')
            if len(ids) == 40000:
                assert primer is None or BENCH._lcp(primer, ids) // 128 * 128 == 32000
                return prompt, ids
    raise ValueError('Cannot fit 40000 input tokens: ' + case_id)


def _prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    originals = json.loads((ASSETS / 'requests.json').read_text())
    dataset = [json.loads(line) for line in (ASSETS / 'dataset_test.jsonl').read_text().splitlines()]
    assert len(originals) == len(dataset) == 1319
    assert _hash(ASSETS / 'requests.json') == '3cf14a90c1f64a3a2e76cd920adc13624278d824dc23001b3cb18302ad4020c0'
    indices = sorted(random.Random(20261026).sample(range(1319), 100))
    tokenizer = BENCH._DeepseekTokenizer(TOKENIZER)
    filler = tokenizer.encode(BENCH.CONTEXT * 2000)
    nonce = uuid.uuid4().hex
    shared, _ = BENCH._make_shared_prefix(tokenizer, nonce, filler)
    requests = []
    references = []
    for index in indices:
        payload = copy.deepcopy(originals[index])
        assert dataset[index]['problem'] in payload['messages'][0]['content']
        assert dataset[index]['reference_solution'] not in payload['messages'][0]['content']
        payload['model'] = MODEL
        tokens = len(BENCH._chat_ids(tokenizer, payload['messages'][0]['content'], 'deepseek-v4-cpp'))
        requests.append(dict(id=f'short_{index}', index=index, stage='short',
                             expected_prompt_tokens=tokens, payload=payload))
        references.append(dict(index=index, **dataset[index]))
    primer_ids = None
    for number in range(2):
        prompt, ids = _long_prompt(tokenizer, shared, filler, 'Output only OK.',
                                   nonce + f'_primer_{number}', primer_ids)
        if primer_ids is None:
            primer_ids = ids
        payload = dict(model=MODEL, messages=[dict(role='user', content=prompt)],
                       temperature=0, top_p=0.95, max_tokens=16,
                       chat_template_kwargs=dict(thinking=False))
        requests.append(dict(id=f'primer_{number}', stage='primer', index=None,
                             expected_prompt_tokens=40000, expected_cached_tokens=0, payload=payload))
    for index in indices:
        payload = copy.deepcopy(originals[index])
        payload['model'] = MODEL
        prompt, _ = _long_prompt(tokenizer, shared, filler, payload['messages'][0]['content'],
                                 nonce + f'_long_{index}', primer_ids)
        payload['messages'][0]['content'] = prompt
        requests.append(dict(id=f'long40k_{index}', index=index, stage='long40k',
                             expected_prompt_tokens=40000, expected_cached_tokens=32000, payload=payload))
    _write(OUT / 'requests.json', requests)
    _write(OUT / 'references.json', references)
    _write(OUT / 'manifest.json', dict(seed=20261026, indices=indices, questions=100,
        url=URL, model=MODEL, concurrency=5, maximum_output_tokens=1024,
        short_protocol='Original frozen zero-shot GSM8K prompt, temperature 0, top_p .95, thinking false.',
        long_protocol='Same questions with neutral filler to exact 40K; two cold primers, then expected cached_tokens 32000.',
        references_in_payload=False, retry_count=0, scope='Small sample accuracy check, not full GSM8K.',
        source_requests=str(ASSETS / 'requests.json'), source_requests_sha256=_hash(ASSETS / 'requests.json'),
        dataset_path=str(ASSETS / 'dataset_test.jsonl'), dataset_sha256=_hash(ASSETS / 'dataset_test.jsonl'),
        requests_sha256=_hash(OUT / 'requests.json'), references_sha256=_hash(OUT / 'references.json'),
        script_sha256=_hash(Path(__file__))))
    BENCH.logger.info('Prepared GSM8K indices %s', indices)


async def _request(client: aiohttp.ClientSession, case: dict) -> dict:
    started = time.time()
    row = dict(id=case['id'], index=case['index'], stage=case['stage'], started_unix=started,
               expected_prompt_tokens=case['expected_prompt_tokens'])
    try:
        async with client.post(URL, json=case['payload']) as response:
            raw = await response.read()
            (OUT / 'responses' / (case['id'] + '.raw')).write_bytes(raw)
            row.update(http_status=response.status, headers=dict(response.headers))
            body = json.loads(raw)
            row['response'] = body
            choice = body.get('choices', [{}])[0]
            row['content'] = choice.get('message', {}).get('content') or ''
            row['finish_reason'] = choice.get('finish_reason')
            usage = body.get('usage', {})
            cached = usage.get('prompt_tokens_details', {}).get('cached_tokens')
            row.update(usage=usage, cached_tokens=cached)
            row['protocol_ok'] = (response.status == 200 and bool(row['content'])
                and row['finish_reason'] == 'stop' and usage.get('completion_tokens', 0) > 0
                and usage.get('prompt_tokens') == case['expected_prompt_tokens']
                and ('expected_cached_tokens' not in case or cached == case['expected_cached_tokens']))
    except Exception as error:
        row.update(protocol_ok=False, error=repr(error))
    row['elapsed_seconds'] = time.time() - started
    _write(OUT / 'responses' / (case['id'] + '.json'), row)
    BENCH.logger.info('%s protocol_ok=%s output_tokens=%s cached=%s',
                     case['id'], row['protocol_ok'], row.get('usage', {}).get('completion_tokens'), row.get('cached_tokens'))
    return row


async def _execute() -> None:
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert _hash(Path(__file__)) == manifest['script_sha256']
    assert _hash(OUT / 'requests.json') == manifest['requests_sha256']
    cases = json.loads((OUT / 'requests.json').read_text())
    with (OUT / 'execution.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
                       started_unix=time.time(), references_loaded=False, url=URL), stream)
    (OUT / 'responses').mkdir()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as client:
        semaphore = asyncio.Semaphore(5)
        async def limited(case: dict) -> dict:
            async with semaphore:
                return await _request(client, case)
        rows = await asyncio.gather(*(limited(case) for case in cases if case['stage'] == 'short'))
        primers = []
        for case in cases:
            if case['stage'] == 'primer':
                primers.append(await _request(client, case))
        rows.extend(primers)
        if all(row['protocol_ok'] for row in primers):
            rows.extend(await asyncio.gather(*(limited(case) for case in cases if case['stage'] == 'long40k')))
        _write(OUT / 'execution_result.json', dict(completed=True, attempted=len(rows),
            planned=len(cases), protocol_passed=sum(row['protocol_ok'] for row in rows), ended_unix=time.time()))


def _boxed(content: str) -> str | None:
    matches = list(re.finditer(r'\\boxed\s*\{', content))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(content)):
        depth += (content[index] == '{') - (content[index] == '}')
        if depth == 0:
            return content[start:index]
    return None


def _number(value: Any) -> Fraction | None:
    if value is None:
        return None
    text = str(value).strip()
    text = re.sub(r'\\(?:text|mathrm)\{[^{}]*\}', '', text)
    text = text.replace('\\$', '').replace('$', '').replace('\\,', '').replace(',', '').replace('\\!', '')
    text = text.replace('\\%', '').replace('%', '').strip()
    match = re.fullmatch(r'\\(?:d?frac)\{([+-]?[\d.]+)\}\{([+-]?[\d.]+)\}', text)
    try:
        if match:
            return Fraction(match[1]) / Fraction(match[2])
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def _score() -> None:
    references = {row['index']: row for row in json.loads((OUT / 'references.json').read_text())}
    scored = []
    for path in sorted((OUT / 'responses').glob('*.json')):
        row = json.loads(path.read_text())
        if row['stage'] == 'primer':
            continue
        reference = references[row['index']]
        answer = _boxed(row.get('content', ''))
        value, gold = _number(answer), _number(reference['expected_answer'])
        assert gold is not None
        scored.append(dict(id=row['id'], index=row['index'], stage=row['stage'],
            expected=reference['expected_answer'], extracted=answer, parsed=str(value) if value is not None else None,
            protocol_ok=row['protocol_ok'], correct=row['protocol_ok'] and value == gold,
            finish_reason=row.get('finish_reason'), usage=row.get('usage'), content=row.get('content', '')))
    _write(OUT / 'scored.json', scored)
    summary = {}
    for stage in ['short', 'long40k']:
        rows = [row for row in scored if row['stage'] == stage]
        summary[stage] = dict(planned=100, attempted=len(rows), correct=sum(row['correct'] for row in rows),
            protocol_passed=sum(row['protocol_ok'] for row in rows),
            failed_indices=[row['index'] for row in rows if not row['correct']],
            accuracy=(sum(row['correct'] for row in rows) / 100 if len(rows) == 100 and all(row['protocol_ok'] for row in rows) else None))
    _write(OUT / 'summary.json', dict(scope='100 sampled GSM8K questions, two input modes; no retries.',
                                    scorer='Last balanced boxed expression; exact rational numeric match; stop completion required.',
                                    results=summary))
    BENCH.logger.info('GSM8K results %s', json.dumps(summary))


if __name__ == '__main__':
    if MODE == 'prepare':
        _prepare()
    elif MODE == 'execute':
        asyncio.run(_execute())
    elif MODE == 'score':
        _score()
    else:
        raise ValueError(MODE)
