#!/usr/bin/env python3
"""Build paired variable-length VL workloads with observable 77-83% total KV reuse."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import random

ROOT = Path('/workspace/decode_tp2_dev_artifacts/performance_comparison_20261027/workloads_stable_kv')
SPEC = importlib.util.spec_from_file_location(
    'existing_builder', '/workspace/xllm-mt-main/tools/musa_qwen3_8_vl_bench/build_20k_prompts.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    tokenizer = builder.load_tokenizer()
    images = [builder.image_tag('/workspace/xllm_vl_bench_ref/images', name)
              for name in builder.IMAGE_NAMES]
    header = ('Below is a collection of news articles, some illustrated with photos, followed by an additional '
              'document. Read everything carefully and answer the question at the end.\n\n')
    common_end = 8288
    common = builder.add_text(tokenizer, header, common_end - 814 - builder.count(tokenizer, header))
    common += images[0] + images[1]
    assert builder.count(tokenizer, common) == common_end
    prefixes = {}
    for cached in [17408, 18432]:
        for image_hits in [2, 3]:
            prefix = common + (images[2] if image_hits == 3 else '')
            prefix = builder.add_text(tokenizer, prefix, cached + 96 - builder.count(tokenizer, prefix))
            prefixes[cached, image_hits] = prefix

    def make(text_tokens: int, image_hits: int, doc_id: int, cached: int | None = None) -> tuple[str, dict]:
        expanded = text_tokens + 5 * 407
        cached = cached or 17408
        prefix = prefixes[cached, image_hits]
        text_at_prefix = cached + 96 - image_hits * 407
        body = prefix + f'\nAdditional document {doc_id} says the following. '
        marker_tokens = builder.count(tokenizer, body) - builder.count(tokenizer, prefix)
        body = builder.add_text(tokenizer, body, text_tokens - text_at_prefix - marker_tokens)
        body += ''.join(images[image_hits:])
        actual = builder.count(tokenizer, body)
        assert actual == expanded, (actual, expanded)
        assert 0.77 <= cached / expanded <= 0.83
        return body, {'text_tokens': text_tokens, 'expanded_tokens': expanded,
                      'expected_cached_tokens': cached, 'image_prefix_hits': image_hits,
                      'expected_kv_hit_rate': cached / expanded, 'doc_id': doc_id}

    seeds = []
    for cached in [17408]:
        for hits in [2, 3]:
            text = 20000 if cached == 17408 else 20500
            for rank in range(6):
                seeds.append(make(text, hits, 900000 + cached + hits * 3 + rank, cached)[0])
    manifest = []
    for concurrency in [6, 12, 16, 18]:
        rng = random.Random(20261027)
        rounds, metadata = [], []
        for phase in range(5):
            prompts, records = [], []
            for i in range(60):
                doc_id = 2000000 + concurrency * 10000 + phase * 100 + i
                prompt, record = make(rng.randint(19500, 20500), 2 + i % 2, doc_id)
                prompts.append(prompt)
                records.append(record)
            rounds.append(prompts)
            metadata.append(records)
        payload = {'schema_version': 2, 'seed': 20261027, 'concurrency': concurrency,
                   'text_jitter_tokens': 500,
                   'prefill_chunk_tokens': 1024, 'images': builder.IMAGE_NAMES,
                   'image_hit_definition': 'image tokens within restored KV prefix; encoder feature cache is separate',
                   'kv_hit_definition': 'cached_tokens / expanded prompt_tokens from Prefill engine logs',
                   'seed_prompts': seeds, 'warm_rounds': rounds[:4], 'measured_prompts': rounds[4],
                   'warm_metadata': metadata[:4], 'measured_metadata': metadata[4], 'osl': 100}
        path = ROOT / f'workload_c{concurrency}.json'
        path.write_text(json.dumps(payload, ensure_ascii=False))
        manifest.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'measured_text_range': [min(x['text_tokens'] for x in metadata[4]),
                                                 max(x['text_tokens'] for x in metadata[4])],
                         'measured_kv_range': [min(x['expected_kv_hit_rate'] for x in metadata[4]),
                                               max(x['expected_kv_hit_rate'] for x in metadata[4])]})
        print(json.dumps(manifest[-1]), flush=True)
    (ROOT / 'workload_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
