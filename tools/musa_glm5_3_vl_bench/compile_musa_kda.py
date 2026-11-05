# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export MATE KDA modules for xLLM's C++ TVM-FFI runtime.

Run before serving, with the same MATE version used by the C++ workspace ABI.
Only dtype/gate/state variants are specialized; heads, batch and lengths are
runtime parameters. No Python or compiler is invoked by xLLM at runtime.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mate.jit.core import JitSpec


def _build_and_export(spec: "JitSpec", ops_path: Path) -> str:
    import tvm_ffi

    library = spec.build()
    name = spec.name
    module = tvm_ffi.load_module(str(library))
    if module.get_function(name) is None:
        raise RuntimeError(f"MATE module {library} has no function {name}")
    target = ops_path / name / f"{name}.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".so.tmp")
    shutil.copy2(library, temporary)
    temporary.replace(target)
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-path", type=Path,
                        default=os.environ.get("FLASHINFER_OPS_PATH"))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.ops_path is None:
        parser.error("set FLASHINFER_OPS_PATH or pass --ops-path")
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    import torch
    import torch_musa  # noqa: F401 - register the MUSA device backend
    from mate import kda
    from mate.jit.kda_ops import (
        gen_kda_prepare_ops_spec,
        gen_kda_recurrence_ops_spec,
        make_kda_ops_config,
        resolve_kda_arch,
    )

    if resolve_kda_arch() != "mp31":
        raise RuntimeError("The current xLLM KDA module ABI targets MATE MP31")
    # Fail before export if MATE changes the chunk-major workspace ABI.
    actual = (kda._KDA_WS_CHUNK, kda._KDA_WS_HEAD_DIM,
              kda._KDA_WS_KRESTORED_OFFSET, kda._KDA_WS_P_OFFSET,
              kda._KDA_WS_INVERSE_OFFSET, kda._KDA_WS_TOTAL_OFFSET,
              kda._KDA_WS_RECORD_STRIDE)
    if actual != (16, 128, 4096, 6144, 6400, 6656, 6784):
        raise RuntimeError(f"MATE workspace ABI changed: {actual}")

    specs = {}
    for dtype in (torch.float16, torch.bfloat16):
        for state_dtype in (dtype, torch.float32):
            for has_gate in (False, True):
                config = make_kda_ops_config(
                    dtype, state_dtype=state_dtype,
                    cu_seqlens_dtype=torch.int64,
                    has_state_in=True, has_state_out=True,
                    has_chunk_states_out=False,
                    state_fp32=state_dtype == torch.float32,
                    has_gate_params=has_gate, is_varlen=True,
                    normalize_qk=True,
                )
                for spec in (gen_kda_prepare_ops_spec(config),
                             gen_kda_recurrence_ops_spec(config)):
                    specs[spec.name] = spec
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for path in executor.map(
            lambda spec: _build_and_export(spec, args.ops_path), specs.values()
        ):
            print(path, flush=True)


if __name__ == "__main__":
    main()
