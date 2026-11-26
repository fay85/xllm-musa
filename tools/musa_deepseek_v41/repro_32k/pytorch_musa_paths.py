# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve MUSA build inputs from the Python interpreter used for the build."""
import argparse
import json
import sysconfig
from importlib.metadata import distribution
from pathlib import Path


def package_paths() -> dict[str, str]:
    torch = Path(distribution("torch").locate_file("torch")).resolve(strict=True)
    musa = Path(distribution("torch_musa").locate_file("torch_musa")).resolve(strict=True)
    cmake = musa / "share" / "cmake"
    for path in (torch / "share/cmake/Torch/TorchConfig.cmake",
                 torch / "include/torch/csrc/api/include/torch/torch.h",
                 torch / "lib/libtorch.so",
                 cmake / "TorchMusa/TorchMusaConfig.cmake",
                 musa / "lib/libmusa_python.so"):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {"torch": str(torch), "torch_musa": str(musa), "cmake": str(cmake),
            "include": sysconfig.get_path("include")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", action="store_true")
    paths = package_paths()
    if parser.parse_args().lines:
        print(*(paths[key] for key in ("torch", "torch_musa", "include")), sep="\n")
    else:
        print(json.dumps(paths, sort_keys=True))
