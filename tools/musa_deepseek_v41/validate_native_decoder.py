# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Native full decoder loop against the Python six-layer model.

Replaces each complete decoder, including shifted mHC, normalization, CSA2,
and original-FP4/EP1 MoE/shared expert. Python model-level embedding/hash,
layer iteration and head remain controls. Engram is disabled to isolate this
boundary; this is not the full CausalLM factory or actual network PD proof.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import MethodType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch_musa  # noqa: F401

from scripts.logger import logger
from tools.musa_deepseek_v41.validate_attention_cache_state import _configure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bmm-so', type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError('Expose exactly one verified idle GPU')
    _configure()
    os.environ['XLLM_DSV41_MUSA_DENSE_QUANT'] = 'original_qdq'
    os.environ['XLLM_DSV41_MATE_BMM_SO'] = str(args.bmm_so)
    import musa_python_kernel_test_module as native
    sys.modules['xllm_runtime'] = native
    from xllm.python import initialize_runtime
    initialize_runtime()
    from tests.python.test_deepseek_v41_model import (
        _DEEPSEEK_V41_FORWARD_CONFIG, _V41ForwardHarness, _random_init_v41,
        _allocate_v41_layer_caches,
    )
    from xllm.python.model_executor.executor import ModelExecutor
    from xllm.python.models.deepseek_v41 import DeepseekV41ForCausalLM

    device = torch.device('musa:0')
    fixture = args.output.with_suffix('.fixture')
    fixture.mkdir(parents=True, exist_ok=False)
    config = {**_DEEPSEEK_V41_FORWARD_CONFIG, 'hidden_size': 64,
        'moe_intermediate_size': 32, 'device': str(device), 'dtype': 'bfloat16',
        'max_position_embeddings': 512, 'window_size': 32, 'sliding_window': 32,
        'engram_layer_ids': [], 'engram_num_embeddings': [],
        'quantization_config': {'quant_method': 'fp8', 'weight_block_size': [32, 32],
                                'scale_fmt': 'ue8m0', 'expert_dtype': 'fp4'},
        'enable_graph': False, 'num_speculative_tokens': 0, 'model_path': str(fixture)}
    (fixture/'config.json').write_text(json.dumps(config, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(111157)
    with torch.inference_mode():
        reference = DeepseekV41ForCausalLM(config)
        _random_init_v41(reference)
        for name, tensor in reference.named_buffers():
            if 'experts_' in name and name.endswith('_scale'):
                tensor.fill_(120)
        for name, parameter in reference.named_parameters():
            if 'experts_' in name and parameter.dtype == torch.uint8:
                parameter.copy_(torch.randint(0, 256, parameter.shape, dtype=torch.uint8).to(device))
        target = DeepseekV41ForCausalLM(config)
        target.load_state_dict(reference.state_dict())
        for model in (reference, target):
            for layer_id, layer in enumerate(model.model.layers):
                for key in ('q_a_proj', 'q_b_proj', 'kv_proj', 'o_b_proj'):
                    model._adapt_loaded_fp8_dense(f'model.layers.{layer_id}.self_attn.{key}')
                if layer.self_attn.indexer is not None:
                    model._adapt_loaded_fp8_dense(f'model.layers.{layer_id}.self_attn.indexer.wq_b')
                for key in ('shared_w1', 'shared_w3', 'shared_w2'):
                    model._adapt_loaded_fp8_dense(f'model.layers.{layer_id}.mlp.{key}')

    class Harness(_V41ForwardHarness):
        def __init__(self, lm: torch.nn.Module, reverse: bool = False) -> None:
            self.lm, self.cfg = lm, lm.cfg
            self.n_seqs, self.dtype, self.device = 1, torch.bfloat16, device
            self.swa_bt = torch.tensor([[1, 2]], device=device, dtype=torch.int32)
            self.tok1_bt = torch.tensor([[1, 2, 3]], device=device, dtype=torch.int32)
            self.tok2_bt = torch.tensor([[1, 2]], device=device, dtype=torch.int32)
            if reverse:
                self.swa_bt = self.swa_bt.flip(-1).contiguous()
                self.tok1_bt = self.tok1_bt.flip(-1).contiguous()
                self.tok2_bt = self.tok2_bt.flip(-1).contiguous()
            self.layer_caches = _allocate_v41_layer_caches(
                lm.cfg, 1, torch.bfloat16, device, reserve_token_sink=True,
                original_attention_qdq=True)
            for layer_id, cache in enumerate(self.layer_caches):
                if layer_id not in lm.cfg.kv_source_layer_ids:
                    continue
                count = 4 if lm.cfg.compress_ratios[layer_id] == 1 else 3
                key = torch.zeros(count, 128, lm.cfg.head_dim, device=device, dtype=torch.bfloat16)
                index = torch.zeros(count, 128, lm.cfg.index_head_dim, device=device, dtype=torch.bfloat16)
                state, score = cache.compress_kv_state, cache.compress_score_state
                if lm.cfg.compress_ratios[layer_id] > 1:
                    state = torch.zeros(cache.swa.shape[0], 32, lm.cfg.head_dim, device=device)
                    score = torch.zeros_like(state)
                self.layer_caches[layer_id] = replace(cache, key=key, index=index,
                    compress_kv_state=state, compress_score_state=score)
            self.executor = ModelExecutor(lm, dict(config), max_seqs_per_batch=1)
            self.executor.bind_kv_caches(self.layer_caches)
            self.backend = self.executor.attention_backend
            assert all(left is right for left, right in zip(self.layer_caches, self.backend._kv_caches))
            self.native_runtime = None
            self.context_length = 0

        def native_cache(self, layer_id: int) -> object:
            source = self.layer_caches[layer_id]
            out = native.NativeV41AttentionCache()
            out.swa = source.swa
            if source.key is not None:
                out.key, out.index = source.key, source.index
            out.compressor_kv_state, out.compressor_score_state = source.compress_kv_state, source.compress_score_state
            return out

        def native_step(self, ids: torch.Tensor, positions: torch.Tensor,
                        end: int, count: int) -> torch.Tensor:
            self.native_runtime = native.NativeV41AttentionRuntime()
            self.context_length = end
            return self.step(ids, positions, [end], [count], is_prefill=count>1,
                             is_chunked_prefill=end>count and count>1)

    modules = []
    for layer_id, layer in enumerate(target.model.layers):
        a = layer.self_attn
        cfg = native.NativeV41AttentionConfig()
        cfg.mode = getattr(native.NativeV41AttentionMode, a.mode.mode.upper())
        for key, value in {'layer': layer_id, 'source_layer': a.mode.main_kv_source_layer,
            'ratio': a.mode.compress_ratio, 'heads': a.num_heads_local, 'groups': a.n_local_groups,
            'output_rank': a.o_lora_rank, 'window': target.cfg.window_size,
            'index_heads': target.cfg.index_n_heads, 'index_dim': target.cfg.index_head_dim,
            'index_topk': target.cfg.index_topk, 'candidate_source': target.cfg.candidate_source_layer_id,
            'candidate_blocks': target.cfg.candidate_topk_blocks,
            'candidate_block_size': target.cfg.candidate_block_size, 'epsilon': target.cfg.rms_norm_eps,
            'scale': a.scale, 'original_fp8_activation': True}.items():
            setattr(cfg, key, value)
        weights = native.NativeV41AttentionWeights()
        weights.projection = [a.q_a_proj.weight, a.q_a_layernorm.weight, a.q_b_proj.weight,
            a.kv_proj.weight, a.kv_a_layernorm.weight, a.o_a_proj.weight, a.o_b_proj.weight]
        weights.sink = a.attn_sink
        if a.mode.is_full:
            weights.compressor_kv, weights.compressor_norm = a.cmp_wkv.weight, a.cmp_norm.weight
            weights.compressor_gate = a.cmp_wgate.weight if a.mode.compress_ratio > 1 else None
        if a.indexer is not None:
            weights.indexer = [a.indexer.wq_b.weight, a.indexer.weights_proj.weight]
            if a.indexer.owns_k:
                weights.indexer += [a.indexer.wk.weight, a.indexer.k_norm.weight]
        decoder_cfg = native.NativeV41DecoderConfig()
        decoder_cfg.attention = cfg
        decoder_cfg.sinkhorn_iterations = target.cfg.hc_sinkhorn_iters
        decoder_cfg.hc_epsilon = target.cfg.hc_eps
        moe_cfg = native.NativeV41MoeConfig()
        for key, value in {'dense': False, 'original_fp8_activation': True,
            'first_expert': layer.mlp.start_expert_id, 'topk': layer.mlp.topk,
            'image_token_id': target.cfg.image_token_id, 'normalize': layer.mlp.norm_topk_prob,
            'scale': layer.mlp.routed_scaling, 'limit': layer.mlp.swiglu_limit,
            'bmm_so_path': str(args.bmm_so)}.items():
            setattr(moe_cfg, key, value)
        decoder_cfg.moe = moe_cfg
        moe_weights = native.NativeV41MoeWeights()
        moe_weights.shared = [layer.mlp.shared_w1.weight, layer.mlp.shared_w3.weight, layer.mlp.shared_w2.weight]
        moe_weights.expert = [layer.mlp.experts_w1, layer.mlp.experts_w3, layer.mlp.experts_w2]
        moe_weights.scale = [layer.mlp.experts_w1_scale, layer.mlp.experts_w3_scale, layer.mlp.experts_w2_scale]
        moe_weights.gate, moe_weights.bias = layer.mlp.gate.weight, layer.mlp.e_score_correction_bias
        moe_weights.vision_bias = layer.mlp.gate_bias_vl if layer.mlp.gate_bias_vl_loaded else None
        decoder_weights = native.NativeV41DecoderWeights()
        decoder_weights.attention, decoder_weights.moe = weights, moe_weights
        decoder_weights.hyper_connection = [getattr(layer.hc, f'hc_{part}_{key}')
            for part in ('attn', 'ffn') for key in ('fn', 'scale', 'base')]
        decoder_weights.input_norm = layer.input_layernorm.weight
        decoder_weights.post_attention_norm = layer.post_attention_layernorm.weight
        modules.append(native.NativeV41Decoder(decoder_cfg, decoder_weights))

    active: dict[str, Any] = {}
    captured_reference, captured_native = {}, {}
    for layer_id, layer in enumerate(reference.model.layers):
        layer.register_forward_hook(
            lambda module, inputs, output, index=layer_id: captured_reference.__setitem__(
                index, tuple(value.detach().clone() for value in output)))
    for layer_id, layer in enumerate(target.model.layers):
        def call_native(self: torch.nn.Module, hidden: torch.Tensor,
                        positions: torch.Tensor, table: torch.Tensor,
                        input_ids: torch.Tensor | None = None,
                        pre_mix: torch.Tensor | None = None,
                        engram_hashes: torch.Tensor | None = None,
                        engram_mask: torch.Tensor | None = None,
                        index: int = layer_id) -> tuple[torch.Tensor, torch.Tensor]:
            assert pre_mix is not None and engram_hashes is None and engram_mask is None
            harness = active['harness']
            ratio = target.cfg.compress_ratios[index]
            token_table = harness.tok1_bt if ratio == 1 else harness.tok2_bt
            output = modules[index].forward(hidden, pre_mix, positions, table,
                harness.native_cache(index), harness.swa_bt, token_table, token_table,
                harness.context_length, harness.native_runtime, input_ids, None, None)
            captured_native[index] = tuple(value.detach().clone() for value in output)
            return output
        layer.forward = MethodType(call_native, layer)

    result: dict[str, Any] = {'status': 'RUNNING', 'scope': __doc__, 'checks': [],
        'modes': [layer.mode.mode for layer in target.model.layers],
        'precision': 'strict bitwise native/Python outputs, hidden, logits and scheduler caches'}
    with torch.inference_mode():
        for chunks in ([1, 1, 3, 32, 1, 65, 1], [129, 1, 127, 3]):
            left, right = Harness(reference), Harness(target)
            active['harness'] = right
            ids = torch.randint(0, target.cfg.vocab_size, (sum(chunks),), device=device)
            previous = 0
            for count in chunks:
                end = previous+count
                positions = torch.arange(previous, end, device=device)
                old = left.step(ids[previous:end], positions, [end], [count], is_prefill=count>1,
                                is_chunked_prefill=previous>0 and count>1)
                new = right.native_step(ids[previous:end], positions, end, count)
                try:
                    for layer_id in range(target.cfg.n_layers):
                        for actual, expected in zip(captured_native[layer_id], captured_reference[layer_id]):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(new, old, rtol=0, atol=0)
                    torch.testing.assert_close(right.logits(new), left.logits(old), rtol=0, atol=0)
                    for a, b in zip(left.layer_caches, right.layer_caches):
                        for field in ('key', 'index', 'swa', 'compress_kv_state', 'compress_score_state'):
                            value = getattr(a, field)
                            if value is not None:
                                torch.testing.assert_close(getattr(b, field), value, rtol=0, atol=0)
                except AssertionError:
                    result.update(status='FAIL', failure={'start': previous, 'count': count,
                        'layer_max_abs': {str(index): float((captured_native[index][0].float()-captured_reference[index][0].float()).abs().max())
                                          for index in captured_native}})
                    args.output.write_text(json.dumps(result, indent=2))
                    torch.save({'native': {k:tuple(t.cpu() for t in v) for k,v in captured_native.items()},
                        'python': {k:tuple(t.cpu() for t in v) for k,v in captured_reference.items()},
                        'native_hidden': new.cpu(), 'python_hidden': old.cpu()}, args.output.with_suffix('.failure.pt'))
                    raise
                result['checks'].append({'start': previous, 'count': count,
                    'six_decoder_hidden_pre_mix_logits_caches': 'bitwise'})
                args.output.write_text(json.dumps(result, indent=2))
                previous = end
    result.update(status='PASS', count=len(result['checks']))
    args.output.write_text(json.dumps(result, indent=2))
    logger.info('Native full decoder integration passed: %s', result['count'])


if __name__ == '__main__':
    main()
