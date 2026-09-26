#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
"""Process-local, non-mutating SGLang hooks; never patch the reference package."""

import os
from pathlib import Path
import sys

import torch
from sglang.srt.models import deepseek_v4 as model

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ["XLLM_SGL_PREFILL_PROBE_DIR"]).resolve()
if not OUTPUT.is_relative_to(ROOT):
    raise ValueError("Probe artifacts must remain in the xLLM checkout")


def _active(batch: object, tokens: int) -> bool:
    return tokens == 76 and batch.forward_mode.is_extend()


def _argument(args: tuple, kwargs: dict, index: int, name: str) -> object:
    return args[index] if len(args) > index else kwargs[name]


def _save(layer: int, stage: str, tensors: dict) -> None:
    if model.get_attention_tp_rank() != 0:
        return
    path = OUTPUT / f"layer_{layer}_{stage}_rank0.pt"
    if path.exists():
        return
    torch.save({key: value.detach().cpu() for key, value in tensors.items()
                if isinstance(value, torch.Tensor)}, path)


_prepare = model.MQALayer._forward_prepare


def _prepare_probe(self: object, *args: object, **kwargs: object) -> object:
    result = _prepare(self, *args, **kwargs)
    batch = args[2]
    if _active(batch, args[0].shape[0]):
        q, kv = result
        _save(self.layer_id, "prepare", {"q": q, "kv": kv, "input": args[0]})
        backend_type = type(args[3])
        if not getattr(backend_type, "_xllm_probe_installed", False):
            original = backend_type.forward

            def _core_probe(backend: object, *values: object, **named: object) -> object:
                out = original(backend, *values, **named)
                q_in = named.get("q")
                batch_in = named.get("forward_batch")
                if q_in is not None and batch_in is not None and _active(batch_in, q_in.shape[0]):
                    _save(named["layer"].layer_id, "core", {"output": out[:, :8, :]})
                return out

            backend_type.forward = _core_probe
            backend_type._xllm_probe_installed = True
    return result


model.MQALayer._forward_prepare = _prepare_probe
_mqa_forward = model.MQALayer.forward


def _mqa_probe(self: object, *args: object, **kwargs: object) -> object:
    output = _mqa_forward(self, *args, **kwargs)
    batch = _argument(args, kwargs, 2, "forward_batch")
    x = _argument(args, kwargs, 0, "x")
    if _active(batch, x.shape[0]):
        _save(self.layer_id, "attention", {"output": output})
    return output


model.MQALayer.forward = _mqa_probe
_model_forward = model.DeepseekV4ForCausalLM.forward


def _model_probe(self: object, *args: object, **kwargs: object) -> object:
    output = _model_forward(self, *args, **kwargs)
    batch = _argument(args, kwargs, 2, "forward_batch")
    tokens = _argument(args, kwargs, 0, "input_ids")
    if _active(batch, tokens.numel()):
        _save(-1, "logits", {"logits": output.next_token_logits, "input_ids": tokens})
    return output


model.DeepseekV4ForCausalLM.forward = _model_probe

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
