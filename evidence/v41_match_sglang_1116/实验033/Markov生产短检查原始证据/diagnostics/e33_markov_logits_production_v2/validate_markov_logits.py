# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise native BF16 Markov projection and FP32 addition before sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch
import torch_musa
import v41_markov_test_module as native


def _chain(base: torch.Tensor, anchor: torch.Tensor, embedding: torch.Tensor,
           head: torch.Tensor, enabled: bool, reference: bool = False) -> torch.Tensor:
    previous = anchor
    tokens = []
    for step in range(base.shape[1]):
        if reference:
            bias = torch.nn.functional.linear(embedding[previous.long()], head)
            if enabled:
                bias = bias.float()
        else:
            bias = native.bias(previous, embedding, head, enabled)
        logits = base[:, step] + bias
        previous = logits.argmax(-1) if reference else native.argmax(logits)
        tokens.append(previous)
    return torch.stack(tokens, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(3301)
    torch.musa.set_device(0)
    stream = torch.musa.Stream()
    checks, graphs = [], []
    with torch.inference_mode(), torch.musa.stream(stream):
        for vocab in [1024, 129280]:
            embedding = torch.randn(vocab, 256, device="musa", dtype=torch.bfloat16) / 8
            head = torch.randn_like(embedding) / 8
            for batch in [1, 3]:
                for steps in [1, 3, 5]:
                    base = torch.randn(batch, steps, vocab, device="musa", dtype=torch.bfloat16)
                    anchor = torch.randint(vocab, (batch,), device="musa")
                    for enabled in [False, True]:
                        bias = native.bias(anchor, embedding, head, enabled)
                        expected_dtype = torch.float32 if enabled else torch.bfloat16
                        assert bias.dtype == expected_dtype
                        actual = _chain(base, anchor, embedding, head, enabled)
                        expected = _chain(base, anchor, embedding, head, enabled, True)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        checks.append(dict(vocab=vocab, batch=batch, steps=steps, fp32=enabled))
                    if batch == 1:
                        for _ in range(2):
                            _chain(base, anchor, embedding, head, True)
                        stream.synchronize()
                        graph = torch.musa.MUSAGraph()
                        with torch.musa.graph(graph):
                            actual = _chain(base, anchor, embedding, head, True)
                        for update in range(3):
                            base.copy_(torch.randn_like(base))
                            anchor.copy_(torch.randint(vocab, anchor.shape, device="musa"))
                            graph.replay()
                            expected = _chain(base, anchor, embedding, head, True, True)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            graphs.append(dict(vocab=vocab, steps=steps, update=update))
                        del graph, actual
        embedding = torch.zeros(8, 256, device="musa", dtype=torch.bfloat16)
        embedding[:, 0] = 1
        head = torch.zeros_like(embedding)
        head[0, 0], head[1, 0] = 1 / 512, 1 / 256
        base = torch.zeros(1, 3, 8, device="musa", dtype=torch.bfloat16)
        base[:, :, :2] = 1
        anchor = torch.zeros(1, device="musa", dtype=torch.int64)
        old = _chain(base, anchor, embedding, head, False).cpu().tolist()
        new = _chain(base, anchor, embedding, head, True).cpu().tolist()
        assert old == [[0, 0, 0]] and new == [[1, 1, 1]], (old, new)
        stream.synchronize()
    result = dict(status="PASS", chain_checks=checks, graph_checks=graphs,
                  near_tie=dict(bf16=old, fp32_add=new),
                  scope="Native Markov + production argmax, not model accuracy or acceptance rate")
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    logger.info("Markov chains=%d, dynamic graphs=%d, near-tie verified", len(checks), len(graphs))


if __name__ == "__main__":
    main()
