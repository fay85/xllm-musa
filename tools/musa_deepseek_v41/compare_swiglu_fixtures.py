# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Compare saved A/B tiny-model tensors, including greedy token agreement."""

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("fused", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--axis", choices=("shared_swiglu", "mhc"), default="shared_swiglu")
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    fused = torch.load(args.fused, map_location="cpu", weights_only=True)
    if args.axis == "shared_swiglu":
        assert reference["shared_swiglu"] == "reference" and fused["shared_swiglu"] == "v4_fused"
        assert reference.get("mhc", "reference") == fused.get("mhc", "reference")
    else:
        assert reference["shared_swiglu"] == fused["shared_swiglu"]
        assert (reference["mhc"], fused["mhc"]) in (("reference", "post"), ("post", "post_pre"), ("post_pre", "all"))
    torch.testing.assert_close(reference["ids"], fused["ids"], rtol=0, atol=0)
    rows = []
    for key in ("hidden", "prefill_logits", "decode_logits"):
        expected, actual = reference[key], fused[key]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0.02)
        row = {"tensor": key, "elements": expected.numel(),
               "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
               "bitwise_equal": bool(torch.equal(actual, expected)), "atol": 0.02, "rtol": 0}
        if "logits" in key:
            agreement = actual.argmax(-1) == expected.argmax(-1)
            assert bool(agreement.all()), "Greedy token changed in tiny-model A/B"
            row["argmax_agreement"] = agreement.float().mean().item()
        rows.append(row)
    args.output.write_text(json.dumps({"status": "PASS", "cases": rows,
                                      "scope": "random tiny-model A/B; not checkpoint or 1P1D acceptance"}, indent=2))


if __name__ == "__main__":
    main()
