# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Check original QDQ cache state across chunks, pages and host restore.

The host restore is an explicit offline transport simulation, not Mooncake
or full-model P/D acceptance. Engram is disabled here to isolate CSA2 state.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401

from scripts.logger import logger


def _configure() -> None:
    for name in ('SHARED_SWIGLU', 'ROPE', 'HC_POST', 'HC_PRE', 'HC_METADATA'):
        os.environ['XLLM_DSV41_MUSA_' + name] = 'v4_fused'
    os.environ['XLLM_DSV41_MUSA_GATE'] = 'v41_fused'
    os.environ['XLLM_DSV41_MUSA_ATTENTION_QUANT'] = 'original_qdq'
    os.environ['XLLM_DSV41_MUSA_FP4_LINEAR'] = 'partition32'
    os.environ['XLLM_DSV41_MUSA_FP4_MOE'] = 'v41_decode'
    os.environ['XLLM_TILELANG_LIB'] = '/usr/local/lib/python3.10/dist-packages/tilelang/lib/libtilelang.so'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bmm-so', type=Path, required=True)
    parser.add_argument('--diagnose-prefix', type=Path)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one idle GPU')
    _configure()
    os.environ['XLLM_DSV41_MATE_BMM_SO'] = str(args.bmm_so)
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime'] = native
    from xllm.python import initialize_runtime
    initialize_runtime()
    from tests.python.test_deepseek_v41_model import (
        _DEEPSEEK_V41_FORWARD_CONFIG, _V41ForwardHarness, _random_init_v41,
        _allocate_v41_layer_caches,
    )
    from xllm.python.models.deepseek_v41 import DeepseekV41ForCausalLM

    device = torch.device('musa:0')
    fixture = args.output.with_suffix('.fixture')
    fixture.mkdir(parents=True, exist_ok=False)
    config = {
        **_DEEPSEEK_V41_FORWARD_CONFIG, 'hidden_size': 64,
        'moe_intermediate_size': 32, 'device': str(device),
        'dtype': 'bfloat16', 'max_position_embeddings': 512,
        'engram_layer_ids': [], 'engram_num_embeddings': [],
        'quantization_config': {'quant_method': 'fp8', 'weight_block_size': [32, 32],
                                'scale_fmt': 'ue8m0', 'expert_dtype': 'fp4'},
        'enable_graph': False, 'num_speculative_tokens': 0,
        'model_path': str(fixture),
    }
    (fixture / 'config.json').write_text(json.dumps(config, indent=2))
    torch.manual_seed(111115)
    with torch.inference_mode():
        model = DeepseekV41ForCausalLM(config)
        _random_init_v41(model)
        for name, tensor in model.named_buffers():
            if 'experts_' in name and name.endswith('_scale'):
                tensor.fill_(120)
        for name, parameter in model.named_parameters():
            if 'experts_' in name and parameter.dtype == torch.uint8:
                parameter.copy_(torch.randint(0, 256, parameter.shape, dtype=torch.uint8).to(device))
        decoder = DeepseekV41ForCausalLM(config)
        decoder.load_state_dict(model.state_dict())

    result = {'status': 'RUNNING', 'checks': [], 'scope': __doc__,
              'tolerances': {'logits_atol': .02, 'bf16_cache_atol': .02,
                             'live_fp32_state_atol': 1e-4},
              'weights': 'synthetic six-layer model; original activation/weight formats',
              'engram': 'disabled only for attention transport isolation'}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def _save() -> None:
        args.output.write_text(json.dumps(result, indent=2))

    class _ExpandedHarness(_V41ForwardHarness):
        def __init__(self, lm: torch.nn.Module) -> None:
            # Allocate all state before the public one-time executor bind.
            self.lm, self.cfg = lm, lm.cfg
            self.n_seqs, self.dtype, self.device = 2, torch.bfloat16, device
            self.swa_bt = torch.arange(1, 5, dtype=torch.int32, device=device).reshape(2, 2)
            self.layer_caches = _allocate_v41_layer_caches(
                lm.cfg, 2, torch.bfloat16, device, reserve_token_sink=True,
                original_attention_qdq=True)

    def _harness(lm: torch.nn.Module, length: int, reverse: bool = False) -> object:
        harness = _ExpandedHarness(lm)
        pages1 = math.ceil(length / 128)
        pages2 = math.ceil(math.ceil(length / 2) / 128)
        for ratio, pages in [(1, pages1), (2, pages2)]:
            table = torch.arange(1, 2 * pages + 1, dtype=torch.int32, device=device).reshape(2, pages)
            if reverse:
                table = table.flip((0, 1)).contiguous()
            if ratio == 1:
                harness.tok1_bt = table
            else:
                harness.tok2_bt = table
        if reverse:
            harness.swa_bt = harness.swa_bt.flip((0, 1)).contiguous()
        for i, cache in enumerate(harness.layer_caches):
            if i not in lm.cfg.kv_source_layer_ids:
                continue
            pages = pages1 if lm.cfg.compress_ratios[i] == 1 else pages2
            key = torch.zeros(1 + 2 * pages, 128, lm.cfg.head_dim,
                              dtype=torch.bfloat16, device=device)
            index = torch.zeros(1 + 2 * pages, 128, lm.cfg.index_head_dim,
                                dtype=torch.bfloat16, device=device)
            state, score = cache.compress_kv_state, cache.compress_score_state
            if lm.cfg.compress_ratios[i] > 1:
                # Match production's 32-row state planes, even for the small window.
                shape = (cache.swa.shape[0], 32, lm.cfg.head_dim)
                state = torch.zeros(shape, device=device)
                score = torch.zeros(shape, device=device)
            harness.layer_caches[i] = replace(cache, key=key, index=index,
                                               indexer_scale=None,
                                               compress_kv_state=state,
                                               compress_score_state=score)
        from xllm.python.model_executor.executor import ModelExecutor
        harness.executor = ModelExecutor(lm, dict(config), max_seqs_per_batch=2)
        harness.executor.bind_kv_caches(harness.layer_caches)
        harness.backend = harness.executor.attention_backend
        for layer, cache in enumerate(harness.layer_caches):
            if harness.backend._kv_caches[layer] is not cache:
                raise AssertionError('Executor is not bound to the tested cache objects')
            for name in ('key', 'index', 'swa', 'compress_kv_state', 'compress_score_state'):
                value = getattr(cache, name)
                if value is None:
                    continue
                table = (harness.tok1_bt if lm.cfg.compress_ratios[layer] == 1 else harness.tok2_bt
                         ) if name in ('key', 'index') else harness.swa_bt
                if int(table.max()) >= value.shape[0]:
                    raise AssertionError(f'Test page table exceeds physical cache: {layer}/{name}')
        return harness

    def _table(harness: object, layer: int, name: str) -> torch.Tensor:
        if name in ('key', 'index'):
            return harness.tok1_bt if model.cfg.compress_ratios[layer] == 1 else harness.tok2_bt
        return harness.swa_bt

    def _canonical(harness: object, layer: int, name: str) -> torch.Tensor:
        value = getattr(harness.layer_caches[layer], name)
        table = _table(harness, layer, name)
        return value.index_select(0, table.flatten().long()).cpu()

    def _cache_check(left: object, right: object, length: int) -> list[dict]:
        records = []
        for layer, cache in enumerate(left.layer_caches):
            for name in ('key', 'index', 'swa'):
                if getattr(cache, name) is None:
                    continue
                actual = _canonical(right, layer, name)
                expected = _canonical(left, layer, name)
                error = float((actual.float() - expected.float()).abs().max())
                records.append({'layer': layer, 'plane': name, 'max_abs': error,
                                'different': int((actual != expected).sum())})
                torch.testing.assert_close(actual, expected, rtol=0, atol=.02)
                if float(expected.abs().max()) <= 1e-8:
                    raise AssertionError(f'Vacuous untouched cache plane: {layer}/{name}')
                tensor = getattr(right.layer_caches[layer], name)
                if bool(tensor[0].ne(0).any()):
                    raise AssertionError(f'Padding block0 was modified: {layer}/{name}')
            if length % 2 and cache.compress_kv_state is not None:
                for name in ('compress_kv_state', 'compress_score_state'):
                    for seq in (0, 1):
                        position = length - 1
                        def _live(h: object) -> torch.Tensor:
                            plane = getattr(h.layer_caches[layer], name)
                            column = (position // h.layer_caches[layer].swa.shape[1]) % h.swa_bt.shape[1]
                            block = int(h.swa_bt[seq, column])
                            return plane[block, position % 32].cpu()
                        actual, expected = _live(right), _live(left)
                        error = float((actual - expected).abs().max())
                        records.append({'layer': layer, 'plane': name, 'seq': seq,
                                        'live_position': position, 'max_abs': error})
                        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)
        return records

    def _restore(source: object, target: object) -> int:
        moved = 0
        for i, cache in enumerate(source.layer_caches):
            for name in ('key', 'index', 'swa', 'compress_kv_state', 'compress_score_state'):
                value = getattr(cache, name)
                if value is None:
                    continue
                src = _table(source, i, name).flatten().long()
                dst = _table(target, i, name).flatten().long()
                # Offline transfer simulation: CPU stores bytes, performs no model math.
                payload = value.index_select(0, src).cpu()
                recovered = payload.to(device)
                getattr(target.layer_caches[i], name).index_copy_(0, dst, recovered)
                torch.testing.assert_close(getattr(target.layer_caches[i], name).index_select(0, dst).cpu(),
                                           payload, rtol=0, atol=0)
                moved += payload.numel() * payload.element_size()
        return moved

    if args.diagnose_prefix is not None:
        failure = torch.load(args.diagnose_prefix, weights_only=True)
        ids = failure['ids_cpu'].to(device)
        captures = {}
        current_capture = {}
        current_harness = None
        handles = []
        import xllm.python.attention.csa2_attention as attention_module
        original_select = attention_module._select_candidate_blocks
        candidate_calls = []
        candidate_captures = {}
        def _record_candidate(score: torch.Tensor, lengths: torch.Tensor,
                              blocks: int, size: int, *, stable_ties: bool = False) -> torch.Tensor:
            selected = original_select(score, lengths, blocks, size, stable_ties=stable_ties)
            candidate_calls.append({'score': score.cpu(), 'selected': selected.cpu(),
                                    'lengths': lengths.cpu()})
            return selected
        attention_module._select_candidate_blocks = _record_candidate
        def _capture(tag: str, layer_id: int) -> Callable:
            def _hook(module: torch.nn.Module, inputs: tuple, output: object) -> None:
                current_capture[tag + '.input'] = inputs[0].cpu()
                value = output[0] if isinstance(output, tuple) else output
                current_capture[tag + '.output'] = value.cpu()
                if tag.endswith('.attn'):
                    topk = module._csa2_used_topk
                    if topk is not None:
                        offset = model.cfg.window_size + inputs[0].shape[0] // 2
                        current_capture[tag + '.topk'] = torch.where(topk >= 0, topk - offset, -1).cpu()
                    for plane in ('index', 'key'):
                        if getattr(current_harness.layer_caches[layer_id], plane) is not None:
                            current_capture[tag + '.cache_' + plane] = _canonical(current_harness, layer_id, plane)
            return _hook
        for i, layer in enumerate(model.model.layers):
            for name, module in [('layer', layer), ('attn', layer.self_attn), ('ffn', layer.mlp)]:
                handles.append(module.register_forward_hook(_capture(f'{i}.{name}', i)))
        with torch.inference_mode():
            for length in (ids.shape[1], 129):
                current_capture = {}
                candidate_calls = []
                current_harness = _harness(model, ids.shape[1])
                h = current_harness.step(ids[:, :length].flatten(), torch.arange(length, device=device).repeat(2),
                                         [length] * 2, [length] * 2, is_prefill=True)
                captures[length] = current_capture
                candidate_captures[length] = candidate_calls
        attention_module._select_candidate_blocks = original_select
        for handle in handles:
            handle.remove()
        report = {}
        full, part = captures[ids.shape[1]], captures[129]
        for tag in full:
            a, b = full[tag], part[tag]
            if '.cache_' in tag:
                ratio = model.cfg.compress_ratios[int(tag.split('.')[0])]
                count = 129 // ratio
                a = a.reshape(2, -1, a.shape[-1])[:, :count]
                b = b.reshape(2, -1, b.shape[-1])[:, :count]
            elif tag.endswith('.topk'):
                a = a[:, :129]
            else:
                a = a.reshape(2, ids.shape[1], *a.shape[1:])[:, :129]
                b = b.reshape(2, 129, *b.shape[1:])
            diff = (a.float() - b.float()).abs()
            coordinates = (diff.reshape(2, diff.shape[1], -1).amax(-1) > 0).nonzero()
            item = {'max_abs': float(diff.max()), 'different_elements': int((a != b).sum()),
                    'first_different_positions': coordinates[:12].tolist()}
            if tag.endswith('.topk') and coordinates.numel():
                seq, position = coordinates[0].tolist()
                item['first_full'] = a[seq, position].tolist()
                item['first_prefix'] = b[seq, position].tolist()
            report[tag] = item
        fa, pa = candidate_captures[ids.shape[1]][0], candidate_captures[129][0]
        report['candidate_first_sequence'] = {
            'common_scores_identical': torch.equal(fa['score'][:129, :129], pa['score']),
            'different_common_mask_elements': int((fa['selected'][:129, :129] != pa['selected']).sum()),
            'row25_full_selected': fa['selected'][25].nonzero().flatten().tolist(),
            'row25_prefix_selected': pa['selected'][25].nonzero().flatten().tolist(),
            'row25_full_visible_scores': fa['score'][25, :26].tolist(),
            'row25_prefix_visible_scores': pa['score'][25, :26].tolist(),
        }
        torch.save({'layers': captures, 'candidates': candidate_captures}, args.output.with_suffix('.layers.pt'))
        args.output.write_text(json.dumps(report, indent=2))
        logger.info('Prefix diagnosis complete')
        return

    try:
        with torch.inference_mode():
            scenarios = [
                ('ragged-compressor-window', [[3, 1], [2, 0], [7, 8], [1, 2], [12, 14]], False),
                ('token-page-and-compressor-boundary', [[129, 129], [128, 128], [3, 3]], True),
            ]
            for label, chunks, transfer in scenarios:
                length = sum(c[0] for c in chunks)
                assert length == sum(c[1] for c in chunks)
                logger.info('Starting attention state scenario: %s', label)
                ids = torch.randint(0, model.cfg.vocab_size, (2, length), device=device)
                ids_cpu = ids.cpu()
                expected_cpu, actual_cpu = None, None
                result['current'] = {'case': label, 'phase': 'full-prefill', 'length': length}
                _save()
                full = _harness(model, length)
                ref_h = full.step(ids.flatten(), torch.arange(length, device=device).repeat(2),
                                  [length, length], [length, length], is_prefill=True)
                expected = full.logits(ref_h).reshape(2, length, -1)
                if not bool(torch.isfinite(expected).all()) or float(expected.abs().max()) <= 1e-4:
                    raise AssertionError('Reference logits are non-finite or degenerate')
                expected_cpu = expected.cpu()
                current = _harness(model, length)
                seen = [0, 0]
                errors = []
                moved = 0
                for step, counts in enumerate(chunks):
                    positions = torch.cat([torch.arange(seen[s], seen[s] + counts[s], device=device) for s in (0, 1)])
                    tokens = torch.cat([ids[s, seen[s]:seen[s] + counts[s]] for s in (0, 1)])
                    seen = [seen[s] + counts[s] for s in (0, 1)]
                    h = current.step(tokens, positions, seen, counts,
                                     is_prefill=max(counts) > 1, is_chunked_prefill=step > 0)
                    actual = current.logits(h)
                    actual_cpu = actual.cpu()
                    reference = torch.cat([expected[s, seen[s] - counts[s]:seen[s]] for s in (0, 1)])
                    maximum = float((actual - reference).abs().max())
                    errors.append(maximum)
                    result['current'] = {'case': label, 'step': step, 'seen': seen, 'max_abs_logits': maximum}
                    _save()
                    torch.testing.assert_close(actual, reference, rtol=0, atol=.02)
                    if transfer and step == 0:
                        destination = _harness(decoder, length, reverse=True)
                        moved = _restore(current, destination)
                        current = destination
                cache_errors = _cache_check(full, current, length)
                result['checks'].append({'name': label, 'status': 'PASS', 'chunks': chunks,
                                         'max_abs_logits_by_chunk': errors, 'cache_planes': cache_errors,
                                         'host_restore_bytes': moved, 'physical_page_ids_reversed': transfer})
                _save()
                logger.info('Attention state scenario passed: %s', label)
        result.update(status='PASS', count=len(result['checks']))
        _save()
    except Exception:
        result['status'] = 'FAIL'
        _save()
        scope = locals()
        # Never query an already-failed device context while recording failure.
        failure = {name: scope[name] for name in
                   ('ids_cpu', 'seen', 'expected_cpu', 'actual_cpu') if name in scope}
        torch.save(failure, args.output.with_suffix('.failure.pt'))
        raise


if __name__ == '__main__':
    main()
