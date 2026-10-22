#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build the reproducible q6 legacy-Attention A/B candidate.

The normal checkout keeps the accuracy-protected Attention implementation.
This helper links a historical Attention object under private symbols, so only
the explicitly enabled Graph target-q6 route can call it.  The old object's
constructors, loader, and helper members are renamed as well; otherwise weak
duplicate members can contaminate the q2 Eager path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


FORWARD_SYMBOL = (
    "_ZN4xllm5layer23DeepseekV4AttentionImpl20forward_legacy_graph"
    "ERKNS0_11DSAMetadataERN2at6TensorERNS_7KVCacheERSt5tupleIJS6_S6_S6_S6_S6_EEbb"
    "RKSA_IJS6_S6_S6_S6_EE"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="new directory under build/")
    parser.add_argument(
        "--legacy-object",
        default=(
            "build/gsm8k_candidates/real_old_attn_current_core_v2_20260923/"
            "deepseek_v4_attention.cpp.o"
        ),
        help="historical Attention object compiled from the d787 snapshot",
    )
    parser.add_argument(
        "--kernel-archive",
        default="",
        help="optional kernel archive to use for the linked candidate",
    )
    parser.add_argument(
        "--no-legacy-kernel-compat",
        action="store_true",
        help="do not add the old-kernel ABI wrapper; use the checkout archive",
    )
    parser.add_argument(
        "--legacy-kernel-compat", action="store_true",
        help="explicit diagnostic-only missing-ABI shims (fail if executed)",
    )
    parser.add_argument(
        "--kernel-source",
        action="append",
        default=[],
        help="compile one additional kernel source into the private archive",
    )
    args = parser.parse_args()

    dest = (root / args.output_dir).resolve()
    if not dest.is_relative_to(root / "build") or dest.exists():
        parser.error("output must be a NEW directory under this checkout's build/")
    legacy_object = (root / args.legacy_object).resolve()
    if not legacy_object.is_file():
        parser.error(f"legacy object does not exist: {legacy_object}")

    builder = root / "tools/build_gsm8k_isolated_candidate.py"
    builder_command = [
        sys.executable,
        str(builder),
        str(dest.relative_to(root)),
        "xllm/core/layers/musa/deepseek_v4/deepseek_v4_attention.cpp",
        "xllm/core/runtime/musa_graph_executor_impl.cpp",
    ]
    if args.kernel_source:
        builder_command.extend(args.kernel_source)
    if args.legacy_kernel_compat and args.no_legacy_kernel_compat:
        parser.error("conflicting legacy compatibility flags")
    if args.legacy_kernel_compat:
        builder_command.append("--legacy-kernel-compat")
    if args.kernel_archive:
        builder_command.extend(["--kernel-archive", args.kernel_archive])
    subprocess.run(builder_command, cwd=root, check=True)

    if args.kernel_archive:
        archive = Path(args.kernel_archive).resolve()
        if not archive.is_file():
            parser.error(f"kernel archive does not exist: {archive}")
        link_script = dest / "link.sh"
        link = link_script.read_text()
        default_archive = str(
            root / "build/dspark-musa-local/xllm/core/kernels/musa/libcuda_kernels.a"
        )
        link = link.replace(default_archive, str(archive))
        link_script.write_text(link)

    isolated_object = dest / "legacy_graph_attention_isolated.o"
    shutil.copyfile(legacy_object, isolated_object)
    rows = subprocess.check_output(
        ["nm", "-g", "--defined-only", str(isolated_object)], text=True
    ).splitlines()
    renamed = []
    for row in rows:
        fields = row.split()
        if len(fields) < 3 or fields[1] not in {"T", "W"}:
            continue
        symbol = fields[2]
        demangled = subprocess.check_output(
            ["c++filt", symbol], text=True
        ).strip()
        if "DeepseekV4AttentionImpl::" not in demangled:
            continue
        target = (
            FORWARD_SYMBOL
            if "::forward(" in demangled
            else f"__dsv4_legacy_member_{len(renamed)}"
        )
        renamed.append({"symbol": symbol, "target": target, "name": demangled})
        subprocess.run(
            ["objcopy", "--redefine-sym", f"{symbol}={target}", str(isolated_object)],
            check=True,
        )

    link_script = dest / "link.sh"
    link = link_script.read_text()
    link = link.replace(" -o ", f" {isolated_object} -o ", 1)
    link_script.write_text(link)
    build_dir = root / "build/dspark-musa-local"
    subprocess.run(["bash", str(link_script)], cwd=build_dir, check=True)
    binary = dest / "xllm"
    if not binary.is_file():
        raise RuntimeError(f"missing linked executable: {binary}")

    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["legacy_object"] = {
        "source": str(legacy_object),
        "sha256": sha256(legacy_object),
        "isolated_object": str(isolated_object),
        "isolated_sha256": sha256(isolated_object),
        "renamed_members": renamed,
        "forward_symbol": FORWARD_SYMBOL,
    }
    manifest["binary_sha256"] = sha256(binary)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest["binary_sha256"], binary)


if __name__ == "__main__":
    main()
