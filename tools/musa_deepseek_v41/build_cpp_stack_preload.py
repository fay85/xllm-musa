# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build an optional profiling-only libtorch RecordFunction native unwinder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiler = shutil.which("g++")
    if compiler is None:
        raise ValueError("Build container must provide g++")
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("native_cpp_stack_preload.cpp")
    torch_root = Path(torch.__file__).resolve().parent
    library = args.output / "libxllm_native_cpp_stack.so"
    command = [
        compiler,
        "-std=c++20",
        "-O2",
        "-fPIC",
        "-shared",
        "-fno-omit-frame-pointer",
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        "-I" + str(torch_root / "include"),
        "-I" + str(torch_root / "include/torch/csrc/api/include"),
        str(source),
        "-L" + str(torch_root / "lib"),
        "-ltorch_cpu",
        "-lc10",
        "-ldl",
        "-pthread",
        "-Wl,-rpath," + str(torch_root / "lib"),
        "-o",
        str(library),
    ]
    logger.info("Compiling optional native-stack preload helper")
    result = subprocess.run(command, capture_output=True, text=True)
    (args.output / "build.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    metadata = {
        "status": "COMPILED_NOT_YET_VALIDATED",
        "command": command,
        "torch": torch.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "bytes": library.stat().st_size,
        "scope": "Actual execinfo native PCs, RecordFunction IDs and clock anchors; profiling only, not enabled for formal performance",
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Native-stack helper compiled: %s", library)


if __name__ == "__main__":
    main()
