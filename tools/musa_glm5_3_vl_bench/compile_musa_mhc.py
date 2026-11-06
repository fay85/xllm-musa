# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE

"""Export MATE HyperConnection prenorm modules for the generic MUSA MHC API."""

import argparse
from pathlib import Path
import os
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mate.jit.core import JitSpec


def _build_and_export(spec: "JitSpec", ops_path: Path) -> Path:
    import tvm_ffi

    library = spec.build()
    module = tvm_ffi.load_module(str(library))
    if module.get_function("tf32_hc_prenorm_gemm") is None:
        raise RuntimeError(f"MATE module {library} has no prenorm entry point")
    target = ops_path / spec.name / f"{spec.name}.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".{os.getpid()}.tmp.so")
    shutil.copy2(library, temporary)
    temporary.replace(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-path", type=Path,
                        default=os.environ.get("FLASHINFER_OPS_PATH"))
    args = parser.parse_args()
    if args.ops_path is None:
        parser.error("set FLASHINFER_OPS_PATH or pass --ops-path")

    from mate.jit.gemm.deep_gemm.hyperconnection import gen_hyperconnection_aot
    from mate.jit.env import MATE_MUSA_ARCH_LIST

    if tuple(MATE_MUSA_ARCH_LIST) != ("mp31",):
        raise RuntimeError(
            "The generic MUSA MHC ABI currently targets exactly MP31; "
            f"got {MATE_MUSA_ARCH_LIST}"
        )
    for spec in gen_hyperconnection_aot():
        _build_and_export(spec, args.ops_path)


if __name__ == "__main__":
    main()
