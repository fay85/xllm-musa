# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated native eight-rank gather contract and Graph microbenchmark."""
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch_musa
import v41_gather_candidate_test_module as native


def _main():
    rank = int(os.environ['RANK'])
    output = Path(os.environ['TEST_OUTPUT'])
    output.mkdir(exist_ok=True)
    result = {'rank': rank, 'status': 'RUNNING', 'checks': [], 'microbenchmarks': []}

    def save(stage):
        result['stage'] = stage
        result['at_unix'] = time.time()
        temp = output / f'rank{rank}.tmp'
        temp.write_text(json.dumps(result, indent=2))
        temp.replace(output / f'rank{rank}.json')

    save('imports_complete')
    torch.set_num_threads(1)
    torch.musa.set_device(rank)
    assert torch.musa.device_count() == 8
    dist.init_process_group('gloo', timeout=timedelta(seconds=90))
    config = json.loads(Path('/work/artifacts/model_view_w4afp8_mt_1017/config.json').read_text())['text_config']
    assert config['hidden_size'] == 5120 and config['vocab_size'] == 129280
    module = Path(native.__file__)
    result['identity'] = {'module': str(module), 'sha256': hashlib.sha256(module.read_bytes()).hexdigest(),
                          'torch': torch.__version__, 'torch_musa': torch_musa.__version__,
                          'hidden_size': config['hidden_size'], 'vocab_size': config['vocab_size']}
    save('before_native_group')
    group = native.GatherCandidate(rank, int(os.environ['NATIVE_PORT']))
    device = torch.device('musa', rank)
    save('native_group_ready')

    def inputs(rows, width, dtype, variant, specials=False):
        blocks = []
        for peer in range(8):
            # CPU rounding occurs before transfer, so the expected gather is
            # a concatenation of actual representable per-rank input values.
            value = ((torch.arange(rows * width).reshape(rows, width) % 97 - 48
                      + peer * 3 + variant) / 32).to(dtype)
            if variant == 111 and dtype == torch.bfloat16:
                value = ((torch.arange(rows * width).reshape(rows, width)
                          + peer * 7919) % 65536 - 32768).to(torch.int16).view(torch.bfloat16)
            if specials and value.numel():
                bit_type = torch.int16 if dtype == torch.bfloat16 else torch.int32
                patterns = ([0, -32768, 1, -32767, 128, -32640, 32640, -128, 32704]
                            if dtype == torch.bfloat16 else
                            [0, -2147483648, 1, -2147483647, 8388608, -2139095040,
                             2139095040, -8388608, 2143289344])
                value.view(bit_type).flatten()[:len(patterns)] = torch.tensor(patterns, dtype=bit_type)
            blocks.append(value)
        return blocks[rank], torch.cat(blocks, dim=-1)

    def check(actual, expected, label, strict=True):
        host = actual.cpu()
        bit_type = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
        mismatches = int((host.view(bit_type) != expected.view(bit_type)).sum())
        finite = torch.isfinite(expected)
        finite_mismatches = int(((host != expected) & finite).sum())
        same_nan = torch.equal(torch.isnan(host), torch.isnan(expected))
        same_inf = torch.equal(torch.isposinf(host), torch.isposinf(expected)) and torch.equal(torch.isneginf(host), torch.isneginf(expected))
        item = {'label': label, 'bit_mismatches': mismatches, 'finite_value_mismatches': finite_mismatches,
                'same_nan_mask': same_nan, 'same_inf_masks': same_inf, 'required_exact': strict}
        result['checks'].append(item)
        if strict:
            assert mismatches == 0, item
        return item

    with torch.inference_mode():
        for nondefault in [False, True]:
            stream = torch.musa.Stream() if nondefault else torch.musa.current_stream()
            stream.wait_stream(torch.musa.current_stream())
            with torch.musa.stream(stream):
                for width in [config['hidden_size'] // 8, config['vocab_size'] // 8]:
                    for dtype in [torch.bfloat16, torch.float32]:
                        for rows in [0, 1, 4, 6]:
                            value, expected = inputs(rows, width, dtype, 0)
                            value = value.to(device)
                            for custom in [False, True]:
                                label = f'eager/nondefault={nondefault}/width={width}/dtype={dtype}/q={rows}/custom={custom}'
                                save('before:' + label)
                                stream.synchronize()
                                dist.barrier()
                                actual = group.forward(value, custom)
                                stream.synchronize()
                                check(actual, expected, label)
                        value, expected = inputs(1, width, dtype, 0, specials=True)
                        value = value.to(device)
                        for custom in [False, True]:
                            label = f'special/nondefault={nondefault}/width={width}/dtype={dtype}/custom={custom}'
                            save('before:' + label)
                            stream.synchronize()
                            dist.barrier()
                            actual = group.forward(value, custom)
                            stream.synchronize()
                            # The encoded path must preserve every bit, including NaN payloads.
                            check(actual, expected, label, strict=True)
            stream.synchronize()

        # OFF -> ON -> OFF microbenchmark, same native communicator and stream.
        stream = torch.musa.Stream()
        stream.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(stream):
            for width in [config['hidden_size'] // 8, config['vocab_size'] // 8]:
                for dtype in [torch.bfloat16, torch.float32]:
                    for rows in [1, 4, 6]:
                        for mode, custom in [('off', False), ('on', True), ('reverse_off', False)]:
                            label = f'graph/width={width}/dtype={dtype}/q={rows}/{mode}'
                            value, expected = inputs(rows, width, dtype, 0)
                            value = value.to(device)
                            stream.synchronize()
                            dist.barrier()
                            save('before_capture:' + label)
                            calls = 8
                            actual = group.capture(value, custom, calls)
                            stream.synchronize()
                            check(actual, expected, label + '/capture')
                            for variant in [3, -7, 111]:
                                new_value, expected = inputs(rows, width, dtype, variant)
                                value.copy_(new_value.to(device))
                                stream.synchronize()
                                dist.barrier()
                                actual = group.replay()
                                stream.synchronize()
                                check(actual, expected, label + f'/variant={variant}')
                            for _ in range(3):
                                group.replay()
                            stream.synchronize()
                            samples = []
                            for _ in range(5):
                                dist.barrier()
                                begin = torch.musa.Event(enable_timing=True)
                                end = torch.musa.Event(enable_timing=True)
                                begin.record(stream)
                                for _ in range(5):
                                    group.replay()
                                end.record(stream)
                                end.synchronize()
                                samples.append(begin.elapsed_time(end) * 1000 / (5 * calls))
                            result['microbenchmarks'].append({'label': label, 'rows': rows, 'local_width': width,
                                'dtype': str(dtype), 'mode': mode, 'calls_per_graph': calls,
                                'replays_per_sample': 5, 'us_per_call_samples': samples,
                                'scope': 'Isolated gather without model compute or rank skew, not model critical-path share'})
                            group.release_graph()
                            stream.synchronize()
                            dist.barrier()
                            save('done:' + label)
        stream.synchronize()
    dist.barrier()
    del group
    dist.destroy_process_group()
    result['special_value_mismatches'] = [x for x in result['checks'] if not x['required_exact'] and x['bit_mismatches']]
    result['status'] = 'BIT_EXACT_AND_GRAPH_PASS'
    save('finished')


if __name__ == '__main__':
    _main()
