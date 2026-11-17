# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare finite-score all-visible IDs/masks with production scored selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch_musa  # noqa: F401
import v41_verify_graph_inputs_test_module as native

sys.path.insert(0, "/workspace/xllm-mt-main")
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexer-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.musa.device_count() == 1
    torch.set_num_threads(1)
    torch.manual_seed(111623)
    device = "musa:0"
    weights = [torch.zeros(4096, 128, device=device, dtype=torch.bfloat16),
               torch.zeros(32, 128, device=device, dtype=torch.bfloat16)]
    reference = native.NativeV41Indexer(weights, 32, 128, 1e-6, True, "")
    aot = native.NativeV41Indexer(weights, 32, 128, 1e-6, True, args.indexer_dir)
    results = []
    with torch.inference_mode():
        for width in (0, 1, 7, 8, 9, 127, 256, 511, 512):
            visible = torch.tensor([0, min(1, width), min(7, width), min(8, width),
                                    min(9, width), max(0, width-1), width],
                                   device=device, dtype=torch.int64)
            columns = torch.arange(width, device=device)
            scores = torch.randn(len(visible), width, device=device).bfloat16().float()
            scores.masked_fill_(columns[None, :] >= visible[:, None], -torch.inf)
            for masking in (False, True):
                mask = None
                if masking:
                    mask = torch.rand(scores.shape, device=device) > 0.35
                    mask[0].zero_()
                picks, blocks = native.v41_select_all_visible(visible, width, 8, mask)
                expected = reference.select(scores, visible, 512, mask)
                fast_expected = aot.select(scores, visible, 512, mask)
                expected_blocks = reference.candidates(scores, visible, 64, 8)
                assert torch.equal(picks, expected)
                assert torch.equal(picks, fast_expected)
                assert torch.equal(blocks, expected_blocks)
                results.append({"width": width, "masked": masking, "exact": True})
        stream = torch.musa.Stream()
        stream.wait_stream(torch.musa.default_stream())
        with torch.musa.stream(stream):
            for rows in (4, 6):
                for width in (256, 512):
                    visible = torch.zeros(rows, device=device, dtype=torch.int64)
                    mask = torch.ones(rows, width, device=device, dtype=torch.bool)
                    graph = torch.musa.MUSAGraph()
                    for _ in range(2):
                        native.v41_select_all_visible(visible, width, 8, mask)
                    stream.synchronize()
                    with torch.musa.graph(graph, stream=stream):
                        picks, blocks = native.v41_select_all_visible(visible, width, 8, mask)
                    for offset in (0, 7, width-rows):
                        visible.copy_(torch.arange(rows, device=device) + offset)
                        mask.copy_(torch.rand(rows, width, device=device) > 0.2)
                        graph.replay()
                        scores = torch.randn(rows, width, device=device).bfloat16().float()
                        scores.masked_fill_(torch.arange(width, device=device)[None, :] >= visible[:, None], -torch.inf)
                        assert torch.equal(picks, reference.select(scores, visible, 512, mask))
                        assert torch.equal(blocks, reference.candidates(scores, visible, 64, 8))
                        results.append({"rows": rows, "width": width, "offset": offset,
                                        "graph_replay_changed_inputs": True, "exact": True})
                    stream.synchronize()
                    graph.reset()
    args.output.write_text(json.dumps({"status": "PASS", "cases": results,
        "contract": "Finite valid model scores; invalid/future rows masked. Not a proof of whole-model accuracy."}, indent=2))
    logger.info("All-visible selection passed %d cases", len(results))


if __name__ == "__main__":
    main()
