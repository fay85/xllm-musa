# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Publish and verify the offline operator bundle and its provenance."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from xllm.core.kernels.musa.aot_config import (
    FFI_SIGNATURES,
    FUSED_MOE_RECIPES,
    OPERATOR_RECIPES,
    PINNED_VERSIONS,
    PROFILE,
    SCHEMA_VERSION,
)


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def bundle_identity(inputs: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def validate_export(library: Path, name: str, kind: str, tir_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check the compiled TVM-FFI signature without executing the operator."""
    import tvm_ffi

    is_tilelang = kind in ("gdn_prefill", "gdn_kkt_fixed", "gdn_kkt_varlen")
    if is_tilelang:
        if tir_schema is None:
            raise RuntimeError(f"MUSA operator is missing verified TIR signature: {name}")
        if tir_schema != FFI_SIGNATURES[kind]:
            raise RuntimeError(f"MUSA operator TIR signature mismatch: {name}: {tir_schema}")
        # Register TileLang's embedded MUSA-module loader on cache-only builds.
        importlib.import_module("tilelang")
    module = tvm_ffi.load_module(str(library))
    module.get_function(name)
    metadata = module.get_function_metadata(name)
    # Untyped TileLang exports require the actual checked TIR interface,
    # carried through publication and cache verification in the manifest.
    if metadata is None and is_tilelang:
        return tir_schema
    if metadata is None or "type_schema" not in metadata:
        raise RuntimeError(f"MUSA operator is missing FFI signature metadata: {name}")
    schema = json.loads(metadata["type_schema"])
    if schema != FFI_SIGNATURES[kind]:
        raise RuntimeError(f"MUSA operator FFI signature mismatch: {name}: {schema}")
    return schema


def check_unmanaged_operators(root: Path) -> None:
    names = {r["name"] for r in OPERATOR_RECIPES}
    names.update(Path(r["path"]).parts[0] for r in FUSED_MOE_RECIPES)
    expected = {f"{r['name']}/{r['name']}.so" for r in OPERATOR_RECIPES}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*.so")}
    unexpected = actual - expected
    bundle_files = {"manifest.json", ".bundle-id", ".build.lock"}
    unexpected.update(
        path.name
        for path in root.iterdir()
        if path.is_symlink()
        or not ((path.is_dir() and path.name in names) or (path.is_file() and path.name in bundle_files))
    )
    for recipe in OPERATOR_RECIPES:
        directory = root / recipe["name"]
        allowed = {f"{recipe['name']}.so", *recipe.get("assets", [])}
        if directory.is_dir():
            unexpected.update(
                path.relative_to(root).as_posix()
                for path in directory.iterdir()
                if path.is_symlink() or (path.name not in allowed and path.suffix != ".tmp")
            )
    expected_mubins = {r["path"] for r in FUSED_MOE_RECIPES}
    unexpected.update(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.mubin")
        if path.relative_to(root).as_posix() not in expected_mubins
    )
    for name in {Path(path).parts[0] for path in expected_mubins}:
        directory = root / name
        if directory.is_dir():
            unexpected.update(
                path.relative_to(root).as_posix()
                for path in directory.iterdir()
                if path.is_symlink()
                or (path.relative_to(root).as_posix() not in expected_mubins and path.suffix != ".tmp")
            )
    if unexpected:
        raise RuntimeError(
            f"Unmanaged MUSA operators in {root}: {sorted(unexpected)}. "
            "Remove these operators or use a clean bundle output directory."
        )


def validate_bundle(root: Path, expected_id: str | None = None) -> dict[str, Any]:
    """Verify the compiler output inventory and provenance."""
    check_unmanaged_operators(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("profile") != PROFILE:
        raise RuntimeError(f"Unsupported MUSA operator manifest: {root}")
    bundle_id = manifest["bundle_id"]
    inputs = manifest["inputs"]
    if (
        inputs.get("versions") != PINNED_VERSIONS
        or inputs.get("architecture") != "mp31"
        or inputs.get("cxx11_abi") is not True
        or inputs.get("profile") != PROFILE
        or inputs.get("operators") != OPERATOR_RECIPES
        or inputs.get("fused_moe_recipes") != FUSED_MOE_RECIPES
        or bundle_identity(inputs) != bundle_id
    ):
        raise RuntimeError(f"MUSA operator provenance mismatch: {root}")
    if expected_id is not None and bundle_id != expected_id:
        raise RuntimeError(f"MUSA operator inputs changed: {root}")
    if (root / ".bundle-id").read_text().strip() != bundle_id:
        raise RuntimeError(f"MUSA operator bundle marker mismatch: {root}")
    records = manifest["operators"]
    expected = {r["name"]: r for r in OPERATOR_RECIPES}
    if len(records) != len(expected) or {r["name"] for r in records} != set(expected):
        raise RuntimeError(f"Incomplete MUSA operator inventory: {root}")
    for record in records:
        name = record["name"]
        recipe = expected[name]
        if (
            record["function"] != recipe.get("function", name)
            or record["arity"] != recipe["arity"]
            or record["type_schema"] != FFI_SIGNATURES[recipe["kind"]]
        ):
            raise RuntimeError(f"MUSA operator contract mismatch: {name}")
        if recipe["kind"] in ("gdn_prefill", "gdn_kkt_fixed", "gdn_kkt_varlen"):
            if record.get("tir_schema") != FFI_SIGNATURES[recipe["kind"]]:
                raise RuntimeError(f"MUSA operator verified TIR signature mismatch: {name}")
        library = root / name / f"{name}.so"
        if not library.is_file() or hash_file(library, "sha512") != record["sha512"]:
            raise RuntimeError(f"Missing or corrupted MUSA operator: {library}")
        assets = record.get("assets", {})
        if set(assets) != set(recipe.get("assets", [])):
            raise RuntimeError(f"Incomplete MUSA operator assets: {name}")
        for filename, digest in assets.items():
            asset = root / name / filename
            if not asset.is_file() or hash_file(asset, "sha512") != digest:
                raise RuntimeError(f"Missing or corrupted MUSA operator asset: {asset}")
    kernels = manifest["fused_moe_kernels"]
    if len(kernels) != len(FUSED_MOE_RECIPES) or {r["path"] for r in kernels} != {r["path"] for r in FUSED_MOE_RECIPES}:
        raise RuntimeError(f"Incomplete MUSA fused MoE kernel inventory: {root}")
    recipes = {r["path"]: r for r in FUSED_MOE_RECIPES}
    for record in kernels:
        recipe = recipes[record["path"]]
        if (
            any(record.get(key) != value for key, value in recipe.items())
            or record.get("function") != "fused_moe_kernel"
            or record.get("arity") != 22
            or record.get("threads_per_block") != recipe["num_warps"] * 32
            or record.get("shared_memory")
            != (recipe["block_m"] + recipe["block_n"]) * recipe["block_k"] * (1 if recipe["fp8"] else 2)
        ):
            raise RuntimeError(f"MUSA fused MoE launch contract mismatch: {record['path']}")
        binary = root / record["path"]
        if not binary.is_file() or hash_file(binary, "sha512") != record["sha512"]:
            raise RuntimeError(f"Missing or corrupted MUSA fused MoE kernel: {binary}")
    return manifest


def _copy_file(source: Path, destination: Path) -> str:
    """Atomically publish an artifact and hash the bytes actually deployed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return hash_file(destination, "sha512")


def publish_operator(
    output: Path,
    recipe: dict[str, Any],
    library: Path,
    assets: dict[str, Path],
    tir_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = recipe["name"]
    function = recipe.get("function", name)
    schema = validate_export(library, function, recipe["kind"], tir_schema)
    directory = output / name
    record = {
        "name": name,
        "function": function,
        "arity": recipe["arity"],
        "type_schema": schema,
        "sha512": _copy_file(library, directory / f"{name}.so"),
        "assets": {filename: _copy_file(source, directory / filename) for filename, source in assets.items()},
    }
    if tir_schema is not None:
        record["tir_schema"] = tir_schema
    return record
