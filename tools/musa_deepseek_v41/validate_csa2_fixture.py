# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Run real MUSA model execution with a tiny random CSA2 paging fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_musa  # noqa: F401

from scripts.logger import logger
from xllm.python import initialize_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.musa.device_count() != 1:
        raise RuntimeError("Expose exactly one idle GPU")
    initialize_runtime()
    from tests.python.test_deepseek_v41_model import (
        _DEEPSEEK_V41_FORWARD_CONFIG,
        _V41ForwardHarness,
        _random_init_v41,
    )
    from xllm.python.models.deepseek_v41 import DeepseekV41ForCausalLM

    device = torch.device("musa:0")
    config = {
        **_DEEPSEEK_V41_FORWARD_CONFIG,
        "hidden_size": 64,
        "moe_intermediate_size": 32,
        "device": str(device),
        "dtype": "bfloat16",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
        "enable_graph": False,
        "num_speculative_tokens": 0,
    }
    fixture_dir = args.output.with_suffix(".fixture")
    fixture_dir.mkdir(parents=True, exist_ok=False)
    config["model_path"] = str(fixture_dir)
    (fixture_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.manual_seed(41)
    with torch.inference_mode():
        model = DeepseekV41ForCausalLM(config)
        _random_init_v41(model)
        for name, tensor in model.named_buffers():
            if "experts_" in name and name.endswith("_scale"):
                tensor.fill_(120)
        for name, parameter in model.named_parameters():
            if "experts_" in name and parameter.dtype == torch.uint8:
                parameter.copy_(torch.randint(0, 256, parameter.shape, dtype=torch.uint8).to(device))
        logger.info("Fixture initialized")
        nseq, length = 2, model.cfg.window_size + 1
        ids = torch.randint(0, model.cfg.vocab_size, (nseq, length)).to(device)
        prefill = _V41ForwardHarness(model, config, nseq, torch.bfloat16, device)
        hidden = prefill.step(ids.flatten(), torch.arange(length, device=device).repeat(nseq), [length] * nseq, [length] * nseq, is_prefill=True)
        logits = prefill.logits(hidden).reshape(nseq, length, -1)
        assert hidden.device.type == "musa" and bool(torch.isfinite(logits).all())
        magnitude = logits.float().abs().max().item()
        diversity = (logits[:, 1:].float() - logits[:, :-1].float()).abs().max().item()
        if magnitude <= 1e-4 or diversity <= 1e-5:
            raise AssertionError("Degenerate logits would make parity validation vacuous")
        logger.info("Prefill finished: magnitude=%s, adjacent-token diversity=%s", magnitude, diversity)
        decode = _V41ForwardHarness(model, config, nseq, torch.bfloat16, device)
        errors = []
        for pos in range(length):
            h = decode.step(ids[:, pos], torch.full((nseq,), pos, device=device), [pos + 1] * nseq, [1] * nseq, is_prefill=False)
            got = decode.logits(h)
            expected = logits[:, pos]
            error = (got - expected).abs().max().item()
            errors.append(error)
            torch.testing.assert_close(got, expected, rtol=0, atol=0.02)
            logger.info("Decode position %s: max_abs_logits_error=%s", pos, error)
        result = {"status": "PASS", "sequences": nseq, "tokens_each": length, "layers": model.cfg.n_layers, "engram_layers": model.cfg.engram_layer_ids, "max_abs_logits": magnitude, "adjacent_token_logits_difference": diversity, "max_abs_logits_error_by_position": errors, "scope": "real single-GPU MUSA execution; random tiny weights and Engram table, synthetic page metadata, no network transfer; not full-model accuracy"}
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
