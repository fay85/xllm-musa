# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small GPU checks for forward-local native compressed-cache read views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_musa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import musa_python_kernel_test_module as native

    torch.set_num_threads(1)
    torch.musa.set_device(0)
    cases = []
    with torch.inference_mode():
        for width in [128, 512]:
            for dtype in [torch.int32, torch.int64]:
                for committed in [0, 1, 127, 128, 129, 255, 256, 257]:
                    capacity = ((committed + 127) // 128) * 128
                    table = torch.tensor([[3, 1, 4, 2]], dtype=dtype, device="musa")
                    physical = torch.full((5, 128, width), float("nan"), dtype=torch.bfloat16, device="musa")
                    rows = torch.arange(committed, device="musa", dtype=torch.int64)
                    values = ((torch.arange(committed * width, device="musa", dtype=torch.float32) % 29) - 14).reshape(committed, width).to(torch.bfloat16)
                    cache = native.NativeV41TokenCache(physical)
                    cache.commit(values, rows, table)
                    # Negative page entries are allowed for unread padding;
                    # the existing gather masks them even when physical data
                    # in block zero are NaN.
                    table[:, (committed + 127) // 128:] = -1
                    reference = cache.gather(table, committed, capacity)
                    view = native.NativeV41TokenCacheReadView(physical, table, committed, capacity)
                    actual = view.materialize()
                    again = view.materialize()
                    expected = torch.zeros((capacity, width), dtype=torch.bfloat16)
                    expected[:committed] = values.cpu()
                    assert torch.equal(actual.cpu(), reference.cpu())
                    assert torch.equal(actual.cpu(), expected)
                    assert actual.data_ptr() == again.data_ptr()
                    assert not bool(torch.isnan(actual).any().item())
                    # A new forward/source gets a fresh view after its writes.
                    changed = values + 1
                    cache.commit(changed, rows, table)
                    fresh = native.NativeV41TokenCacheReadView(physical, table, committed, capacity).materialize()
                    expected[:committed] = changed.cpu()
                    assert torch.equal(fresh.cpu(), expected)
                    cases.append({"width": width, "table_dtype": str(dtype), "committed": committed, "capacity": capacity, "bitwise_reference": True, "repeat_same_storage": True, "new_view_observes_commit": True, "padding_nan_removed": True})
        torch.musa.synchronize()
    result = {"status": "PASS_NATIVE_GPU_READ_VIEW_BASIC_CHECKS", "cases": cases, "case_count": len(cases), "contract": "Read-only source within one forward. New FULL publication or token/request must discard the old view.", "accuracy_scope": "仅通过基础功能检查，未完成模型精度验收"}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"status": result["status"], "case_count": len(cases)}))


if __name__ == "__main__":
    main()
