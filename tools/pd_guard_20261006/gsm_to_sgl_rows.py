#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Adapt a completed frozen run for independent official CPU scoring.

Only this offline adapter reads reference answers. They never enter generation.
Transport failures remain in the denominator; no case filtering or retries.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    complete = json.loads((args.run_dir / 'complete.json').read_text())
    if (complete.get('dry_run') or complete.get('partial_diagnostic') or
            complete.get('http_requests_sent') != 1319 or
            not complete.get('all_planned_requests_recorded')):
        raise ValueError('A complete 1319-question generation run is required')
    dataset_path = args.assets / 'gsm8k_frozen/dataset_test.jsonl'
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != '67e9046d913470477dfb321814d4e3240240a397cfc19c6c610af02dd2cb6456':
        raise ValueError('Reference dataset hash mismatch')
    dataset = [json.loads(line) for line in dataset_path.read_text().splitlines() if line]
    transport = [json.loads(line) for line in (args.run_dir / 'transport.jsonl').read_text().splitlines()]
    if len(dataset) != 1319 or [row['index'] for row in transport] != list(range(1319)):
        raise ValueError('Expected full ordered coverage, no duplicates or retries')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        for index, reference in enumerate(dataset):
            body = (args.run_dir / f'response_{index}.http.body').read_bytes()
            if hashlib.sha256(body).hexdigest() != transport[index]['raw_body_sha256']:
                raise ValueError(f'Response evidence hash mismatch: {index}')
            response = json.loads((args.run_dir / f'response_{index}.json').read_text())
            choice = (response.get('choices') or [{}])[0]
            message = choice.get('message') or {}
            row = {'id': f'gsm8k-{index}', 'problem': reference['problem'],
                   'expected_answer': reference['expected_answer'],
                   'request_payload': json.loads((args.run_dir / f'payload_{index}.json').read_text()),
                   'effective_request_payload': json.loads((args.run_dir / f'http_payload_{index}.json').read_text()),
                   'generation': message.get('content') or '', 'response': response,
                   'usage': response.get('usage'), 'finish_reason': choice.get('finish_reason'),
                   'reasoning_content': message.get('reasoning_content')}
            error = transport[index].get('error') or response.get('error')
            if error:
                row.update(error=error, finish_reason='error')
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'rows': 1319, 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
