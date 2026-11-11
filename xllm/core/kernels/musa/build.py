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

"""CMake entry and compiler adapters for the offline MUSA operator bundle."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.logger import logger
from xllm.core.kernels.musa import aot_bundle
from xllm.core.kernels.musa.aot_config import (
    FFI_SIGNATURES,
    FUSED_MOE_RECIPES,
    OPERATOR_RECIPES,
    PINNED_VERSIONS,
    PROFILE,
    SCHEMA_VERSION,
    header_content,
)

_TRITON_COMPILER_HASHES = {
    "backends/mtgpu/compiler.py": "1fd6922102a7e93c123026a19995965cc9a5c4fb9f985e9cbf52967a65eb1e02",
    "_C/libtriton.so": "d824a3a71db07db1d5bca3dc0e5fe6c1662a2b25a62b9b03bcc9aa49dac7b4a2",
}
_INPUT_FILES: set[Path] = set()


def _hash_input(path: Path) -> str:
    path = path.absolute()
    resolved = path.resolve(strict=True)
    _INPUT_FILES.update((path, resolved, path.parent, resolved.parent))
    return aot_bundle.hash_file(resolved)


def _hash_inputs(paths: dict[str, Path]) -> str:
    contents = {}
    hashes = {}
    for name, path in sorted(paths.items()):
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise RuntimeError(f"MUSA compiler input is not a file: {path}")
        if resolved not in hashes:
            hashes[resolved] = _hash_input(path)
        contents[name] = hashes[resolved]
    return hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()


def _hash_tree(root: Path) -> str:
    if not root.is_dir():
        raise RuntimeError(f"MUSA compiler input directory is missing: {root}")
    _INPUT_FILES.update((root, root.parent))
    _INPUT_FILES.update(path for path in root.rglob("*") if path.is_dir() and "__pycache__" not in path.parts)
    return _hash_inputs(
        {
            path.relative_to(root).as_posix(): path
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
    )


def _compiler_identity(command: str) -> dict[str, str]:
    compiler = shutil.which(command)
    if compiler is None:
        raise RuntimeError(f"MUSA operator compiler is missing: {command}")
    return {
        "version": subprocess.check_output([compiler, "--version"], text=True).strip(),
        "sha256": _hash_input(Path(compiler)),
    }


def _source_identity(mate_root: Path) -> dict[str, Any]:
    if os.getenv("MUSA_REPLACE_IR", "0") == "1" or os.getenv("TRITON_KERNEL_OVERRIDE", "0") == "1":
        raise RuntimeError("The versioned MUSA operator build does not accept compiler IR overrides")
    try:
        versions = {package: importlib.metadata.version(package) for package in PINNED_VERSIONS}
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "MUSA operator toolchain is incomplete; use an image with the pinned Triton-MUSA 3.2.0 compiler"
        ) from error
    if versions != PINNED_VERSIONS:
        raise RuntimeError(f"MUSA operator toolchain differs from the supported release: {versions}")
    for package in PINNED_VERSIONS:
        distribution = importlib.metadata.distribution(package)
        for path in distribution.files or ():
            if path.name == "METADATA" and path.parent.suffix == ".dist-info":
                _INPUT_FILES.add(Path(distribution.locate_file(path)))
    from mate.jit import cpp_ext

    musa_home = cpp_ext.get_musa_home()
    compiler = _compiler_identity(cpp_ext.get_mcc())
    compiler_version = compiler["version"]
    if "mcc version 5.1.0" not in compiler_version:
        raise RuntimeError(f"This operator recipe requires MUSA 5.1.0: {compiler_version}")
    digest = hashlib.sha256()
    # Hash package sources/headers, not just the distribution version: vendor
    # releases with the same version have shipped incompatible FFI signatures.
    for directory in ("jit", "gdn_kernels", "data/csrc", "data/include", "data/mutlass/include"):
        tree = mate_root / directory
        if not tree.is_dir():
            raise RuntimeError(f"MATE compiler input is missing: {tree}")
        _INPUT_FILES.update((tree, tree.parent))
        _INPUT_FILES.update(path for path in tree.rglob("*") if path.is_dir() and "__pycache__" not in path.parts)
        for path in sorted(p for p in tree.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            digest.update(path.relative_to(mate_root).as_posix().encode())
            digest.update(bytes.fromhex(_hash_input(path)))
    import torch

    libraries = {
        path.name: path
        for pattern in ("libmusa.so*", "libmusart.so*", "libmudnn*.so*")
        for path in (musa_home / "lib").glob(pattern)
    }
    for name in ("libmusa.so", "libmusart.so", "libmudnncxx.so"):
        if name not in libraries:
            raise RuntimeError(f"MUSA operator runtime library is missing: {name}")
    packages = {}
    for package, directory in (
        ("apache-tvm-ffi", "tvm_ffi"),
        ("tilelang-musa", "tilelang"),
        ("mate-mubin", "mate_mubin"),
        ("triton", "triton"),
    ):
        root = Path(importlib.metadata.distribution(package).locate_file(directory))
        packages[package] = _hash_tree(root)
        if package == "triton":
            for filename, expected_hash in _TRITON_COMPILER_HASHES.items():
                if not (root / filename).is_file() or _hash_input(root / filename) != expected_hash:
                    raise RuntimeError("Fused MoE AOT requires the pinned Triton-MUSA 3.2.0 compiler, not PyPI Triton")
    for package in ("torch", "torch_musa"):
        root = Path(importlib.metadata.distribution(package).locate_file(package))
        packages[package] = _hash_tree(root / "lib")
    replacement_source = os.getenv("TILELANG_REPLACE_MUSAC", "")
    toolchain = {
        "mcc": compiler,
        "tilelang_mcc": _compiler_identity(str(musa_home / "bin" / "mcc")),
        "cxx": _compiler_identity(cpp_ext.get_cxx()),
        "sdk_headers_sha256": _hash_tree(musa_home / "include"),
        "triton_libdevice_sha256": _hash_input(
            Path(os.getenv("MUSA_LIBDEVICE_PATH", str(musa_home / "mtgpu/bitcode/libdevice.31.bc")))
        ),
        "runtime_libraries_sha256": _hash_inputs(libraries),
        "packages_sha256": packages,
        "replacement_source_sha256": _hash_inputs({"source": Path(replacement_source)}) if replacement_source else "",
        "flags": {
            name: os.getenv(name, "")
            for name in (
                "MATE_EXTRA_CFLAGS",
                "MATE_EXTRA_MUSAFLAGS",
                "MATE_EXTRA_LDFLAGS",
                "TILELANG_MCC_EXTRA_ARGS",
                "TILELANG_OPT_LEVEL",
                "TILELANG_REPLACE_MUSAC",
                "ENABLE_MUSA_MMA",
                "DISABLE_SQMMA",
                "TRITON_DISABLE_LINE_INFO",
                "MUSA_ENABLE_LLC_OPT",
            )
        },
    }
    return {
        "versions": versions,
        "mate_sources_sha256": digest.hexdigest(),
        "compiler_version": compiler_version,
        "toolchain": toolchain,
        "architecture": "mp31",
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "build_sha256": _hash_inputs(
            {name: Path(__file__).parent / name for name in ("build.py", "aot_config.py", "aot_bundle.py")}
        ),
        "fused_moe_source_sha256": _hash_input(Path(__file__).parent / "triton_kernel/fused_moe.py"),
        "profile": PROFILE,
        "operators": OPERATOR_RECIPES,
        "fused_moe_recipes": FUSED_MOE_RECIPES,
    }


def _build_fused_moe_kernels(output: Path, build: Path) -> list[dict[str, Any]]:
    import triton
    from triton.backends.compiler import AttrsDescriptor, GPUTarget
    from triton.compiler import ASTSource

    source = Path(__file__).parent / "triton_kernel/fused_moe.py"
    spec = importlib.util.spec_from_file_location("xllm_musa_fused_moe_source", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    function = module.fused_moe_kernel
    os.environ["TRITON_CACHE_DIR"] = str(build / "triton")
    records = []
    for recipe in FUSED_MOE_RECIPES:
        fp8 = recipe["fp8"]
        constants = {
            "BLOCK_SIZE_M": recipe["block_m"],
            "BLOCK_SIZE_N": recipe["block_n"],
            "BLOCK_SIZE_K": recipe["block_k"],
            "GROUP_SIZE_M": recipe["group_m"],
            "MUL_ROUTED_WEIGHT": recipe["stage"] == "down",
            "TOP_K": 1 if recipe["stage"] == "down" else 8,
            "USE_FP8": fp8,
        }
        types = ["*fp8e4nv" if fp8 else "*bf16"] * 2 + ["*bf16"] + ["*fp32"] * 3 + ["*i32"] * 3 + ["i32"] * 13
        signature = dict(zip(function.arg_names[:22], types, strict=True))
        aligned = (*range(11), 13, 14, 15, 16, 17, 18)
        attrs = AttrsDescriptor.from_dict(
            {"arg_properties": {"tt.divisibility": aligned, "tt.equal_to": ()}, "cls": "AttrsDescriptor"}
        )
        compiled = triton.compile(
            ASTSource(function, signature, constants, attrs),
            target=GPUTarget("musa", 31, 32),
            options={"num_warps": recipe["num_warps"], "num_stages": recipe["num_stages"]},
        )
        declaration = re.search(r"define mtgpu_kernel void @fused_moe_kernel\((.*)\) [^{\n]*\{", compiled.asm["llir"])
        if declaration is None or len(declaration[1].split(",")) != 22 or compiled.metadata.name != "fused_moe_kernel":
            raise RuntimeError("Compiled fused MoE kernel does not match the native 22-argument launch contract")
        binary = output / recipe["path"]
        binary.parent.mkdir(parents=True, exist_ok=True)
        temporary = binary.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_bytes(compiled.asm["mubin"])
        temporary.replace(binary)
        records.append(
            {
                **recipe,
                "function": compiled.metadata.name,
                "arity": 22,
                "threads_per_block": compiled.metadata.num_warps * 32,
                "shared_memory": compiled.metadata.shared,
                "sha512": aot_bundle.hash_file(binary, "sha512"),
            }
        )
    return records


def _moe_gemm_spec(recipe: dict[str, Any]) -> tuple[Any, Path]:
    """Compile a launcher for a checked, installed vendor kernel; never download."""
    import torch
    from mate.artifacts import KernelMapHash, load_kernel_map
    from mate.jit.mubin.gemm.dispatch import get_moe_gemm_mubin_dispatcher
    from mate.jit.mubin.gemm.types import TensorMajor, TensorQuantMode
    from mate.jit.mubin.launcher import gen_mubin_launcher_spec, render_gemm_mubin_launcher

    root = Path(importlib.metadata.distribution("mate-mubin").locate_file("mate_mubin/mubin/gemm"))
    maps = sorted(root.glob("*/kernel_map.json"))
    if len(maps) != 1:
        raise RuntimeError(
            f"The pinned MATE GEMM package requires exactly one kernel map in {root}; "
            f"found {len(maps)}: {[str(path) for path in maps]}"
        )
    if aot_bundle.hash_file(maps[0]) != KernelMapHash.GEMM:
        raise RuntimeError(f"The pinned, installed MATE GEMM kernel map is missing or corrupted: {maps[0]}")
    dispatcher = get_moe_gemm_mubin_dispatcher(maps[0])
    config = recipe["config"]
    fp8 = config["dtype"].startswith("fp8_")
    output_dtype = torch.float16 if config["dtype"].endswith("fp16") else torch.bfloat16
    options = {
        "type_a": torch.float8_e4m3fn if fp8 else output_dtype,
        "type_b": torch.float8_e4m3fn if fp8 else output_dtype,
        "type_d": output_dtype,
        "quant_tile": 128,
        "quant_mode_a": TensorQuantMode.GROUP if fp8 else TensorQuantMode.NO_QUANT,
        "quant_mode_b": TensorQuantMode.BLOCK if fp8 else TensorQuantMode.NO_QUANT,
    }
    if config["mode"] == "contiguous":
        asm_id = dispatcher.get_ragged_expert_layout_asm_id(**options, candidate_block_id=1, major_b=TensorMajor.K)
    else:
        select = dispatcher.get_masked_asm_id if config["mode"] == "masked" else dispatcher.get_ragged_asm_id
        asm_id = select(**options, alignment_m=config["alignment"], tme_cache_hint_b=False)
    # Resolve locally rather than using ensure_mubin_kernel_artifact, which may
    # fetch an uninstalled asset. The map pins both dispatch and file hashes.
    from mate.jit.mubin.gemm.dispatch import get_gemm_mubin_id_hash

    dispatch_hash = get_gemm_mubin_id_hash(asm_id)
    entries = {entry.dispatch_hash: entry for entry in load_kernel_map(maps[0])}
    entry = entries.get(dispatch_hash)
    if entry is None:
        raise RuntimeError(f"The pinned MATE release has no GEMM kernel for {config}")
    kernel = maps[0].parent / "mubin" / entry.file_name
    if not kernel.is_file() or aot_bundle.hash_file(kernel) != entry.file_hash:
        raise RuntimeError(f"The pinned MATE GEMM kernel is missing or corrupted: {kernel}")
    source = render_gemm_mubin_launcher(asm_id, entry.kernel_name)
    source = source.replace(f"run_{entry.kernel_name}", recipe["name"])
    return gen_mubin_launcher_spec("gemm", recipe["name"], source), kernel


def _build_gdn_operator(recipe: dict[str, Any], library: Path) -> dict[str, Any]:
    import torch
    from mate.gdn_kernels.tilelang.gdn_kkt_solve import tilelang_kkt_solve
    from mate.gdn_kernels.tilelang.gdn_prefill import tilelang_fused_chunk_gdn_prefill

    options = {
        **recipe["config"],
        "qkva_dtype": torch.bfloat16,
        "b_dtype": torch.float32,
        "seqlen_dtype": torch.int32,
        "accum_dtype": "float32",
    }
    if recipe["kind"] == "gdn_prefill":
        kernel = tilelang_fused_chunk_gdn_prefill(
            **options,
            g_dtype=torch.float32,
            h0_dtype=torch.float32,
            ht_dtype=torch.float32,
            o_dtype=torch.bfloat16,
        )
    else:
        kernel = tilelang_kkt_solve(**options)
    args = []
    for param in kernel.prim_func.params:
        if param in kernel.prim_func.buffer_map:
            args.append({"type": "DLTensor*"})
        elif str(param.dtype) == "int32":
            args.append({"type": "int"})
        else:
            raise RuntimeError(f"Unsupported TileLang parameter in {recipe['name']}: {param}")
    schema = {"type": "ffi.Function", "args": [{"type": "None"}, *args]}
    if kernel.adapter.result_idx or schema != FFI_SIGNATURES[recipe["kind"]]:
        raise RuntimeError(f"MUSA operator TIR signature mismatch: {recipe['name']}: {schema}")
    library.parent.mkdir(parents=True, exist_ok=True)
    cached_library = getattr(kernel.adapter, "libpath", None)
    if cached_library is not None:
        cached_path = Path(cached_library).resolve()
        cache_root = Path(os.environ["TILELANG_CACHE_DIR"]).resolve()
        if not cached_path.is_file() or not cached_path.is_relative_to(cache_root):
            raise RuntimeError(f"TileLang operator is not from this build's compiler cache: {cached_path}")
        shutil.copy2(cached_path, library)
    else:
        kernel.adapter.get_exportable_executable().export_library(str(library))
    return schema


def _build_mate_operators(workspace: Path, jobs: int) -> tuple[dict[str, Path], dict[str, dict[str, Path]]]:
    from mate.jit import build_jit_specs
    from mate.jit import env as jit_env
    from mate.jit.attention.fmha.fmha_combine import gen_fmha_fwd_combine_spec
    from mate.jit.attention.fmha.fmha_fwd import gen_fmha_fwd_spec
    from mate.jit.attention.fmha.fmha_get_metadata import gen_fmha_metadata_spec
    from mate.jit.gemm_ops import gen_gemm_ops_spec

    jit_env.MATE_GEN_SRC_DIR = workspace / "generated"
    jit_env.MATE_JIT_DIR = workspace / "compiled"
    generators = {
        "forward": gen_fmha_fwd_spec,
        "metadata": gen_fmha_metadata_spec,
        "combine": gen_fmha_fwd_combine_spec,
    }
    specs = []
    assets = {}
    mate_recipes = [recipe for recipe in OPERATOR_RECIPES if recipe["kind"] in (*generators, "bmm", "moe_gemm")]
    for recipe in mate_recipes:
        config = {**recipe["config"], "func_name": recipe["name"]}
        if recipe["kind"] == "moe_gemm":
            original, kernel = _moe_gemm_spec(recipe)
            assets[recipe["name"]] = {"kernel.o": kernel}
        elif recipe["kind"] == "bmm":
            original = gen_gemm_ops_spec()
        else:
            original = generators[recipe["kind"]](config)
        specs.append(
            dataclasses.replace(
                original,
                name=recipe["name"],
                aot_path=None,
                extra_cflags=[*(original.extra_cflags or []), "-DTVM_FFI_DLL_EXPORT_INCLUDE_METADATA=1"],
                extra_cuda_cflags=[*(original.extra_cuda_cflags or []), "-DTVM_FFI_DLL_EXPORT_INCLUDE_METADATA=1"],
            )
        )
    build_jit_specs(specs, jobs=jobs, verbose=True, skip_prebuilt=False)

    return {spec.name: spec.jit_library_path for spec in specs}, assets


def _build_bundle(output: Path, build: Path, header: Path, jobs: int, check_only: bool = False) -> None:
    if jobs < 1:
        raise ValueError("MUSA operator build jobs must be positive")
    if output == build or output in build.parents or build in output.parents:
        raise ValueError("MUSA bundle output and compiler workspace must be separate directories")
    if os.getenv("MATE_MUSA_ARCH_LIST", "3.1").strip() != "3.1":
        raise RuntimeError("The supported MUSA operator profile targets MP31; use MATE_MUSA_ARCH_LIST=3.1")
    os.environ["MATE_MUSA_ARCH_LIST"] = "3.1"
    os.environ["MATE_WORKSPACE_BASE"] = str(build.resolve())
    os.environ["TILELANG_DEFAULT_TARGET"] = "musa"
    dist = importlib.metadata.distribution("mate")
    mate_root = Path(dist.locate_file("mate"))
    identity = _source_identity(mate_root)
    if not identity["cxx11_abi"]:
        raise RuntimeError("The pinned MUSA operator profile requires the C++11 ABI")
    bundle_id = aot_bundle.bundle_identity(identity)
    os.environ["TILELANG_CACHE_DIR"] = str(build.resolve() / "tilelang" / bundle_id)
    if check_only:
        if not (output / ".build.lock").is_file():
            raise RuntimeError(f"MUSA operator bundle has not been built: {output}")
        with (output / ".build.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            manifest = aot_bundle.validate_bundle(output, bundle_id)
            records = {record["name"]: record for record in manifest["operators"]}
            for recipe in OPERATOR_RECIPES:
                library = output / recipe["name"] / f"{recipe['name']}.so"
                aot_bundle.validate_export(
                    library,
                    recipe.get("function", recipe["name"]),
                    recipe["kind"],
                    records[recipe["name"]].get("tir_schema"),
                )
        logger.info("MUSA operator bundle verified without building: %s", bundle_id)
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        aot_bundle.check_unmanaged_operators(output)
        if (output / "manifest.json").is_file():
            try:
                manifest = aot_bundle.validate_bundle(output, bundle_id)
            except (OSError, ValueError, KeyError, RuntimeError):
                logger.info("MUSA operator bundle is stale or incomplete; rebuilding from pinned inputs.")
            else:
                records = {record["name"]: record for record in manifest["operators"]}
                for recipe in OPERATOR_RECIPES:
                    library = output / recipe["name"] / f"{recipe['name']}.so"
                    aot_bundle.validate_export(
                        library,
                        recipe.get("function", recipe["name"]),
                        recipe["kind"],
                        records[recipe["name"]].get("tir_schema"),
                    )
                aot_bundle.write_if_changed(
                    header,
                    header_content(
                        bundle_id,
                        manifest["operators"],
                        manifest["fused_moe_kernels"],
                        aot_bundle.hash_file(output / "manifest.json", "sha512"),
                    ),
                )
                logger.info("MUSA operator bundle verified: %s", bundle_id)
                return
        workspace = build / bundle_id
        libraries, assets = _build_mate_operators(workspace, jobs)
        records = []
        for recipe in OPERATOR_RECIPES:
            name = recipe["name"]
            tir_schema = None
            if name not in libraries:
                library = workspace / "compiled" / name / f"{name}.so"
                tir_schema = _build_gdn_operator(recipe, library)
                libraries[name] = library
            records.append(
                aot_bundle.publish_operator(output, recipe, libraries[name], assets.get(name, {}), tir_schema)
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "profile": PROFILE,
            "bundle_id": bundle_id,
            "inputs": identity,
            "operators": records,
            "fused_moe_kernels": _build_fused_moe_kernels(output, build / bundle_id),
        }
        aot_bundle.write_if_changed(output / ".bundle-id", bundle_id + "\n")
        aot_bundle.write_if_changed(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        aot_bundle.validate_bundle(output, bundle_id)
        aot_bundle.write_if_changed(
            header,
            header_content(
                bundle_id,
                manifest["operators"],
                manifest["fused_moe_kernels"],
                aot_bundle.hash_file(output / "manifest.json", "sha512"),
            ),
        )
        logger.info(
            "Built and verified %d MUSA FFI operators and %d fused MoE kernels.", len(records), len(FUSED_MOE_RECIPES)
        )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument(
        "--jobs", type=int, default=os.getenv("XLLM_MUSA_AOT_JOBS", os.getenv("MAX_JOBS", os.cpu_count() or 1))
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stamp", type=Path)
    parser.add_argument("--depfile", type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if (args.stamp is None) != (args.depfile is None) or (args.check_only and args.stamp is not None):
        parser.error("--stamp and --depfile must be supplied together, only when building")
    _build_bundle(
        args.output_dir.resolve(), args.build_dir.resolve(), args.header.resolve(), args.jobs, args.check_only
    )
    if args.stamp is not None:
        # Track hashed compiler inputs, package inventories and published outputs.
        # Directory dependencies also invalidate additions/removals and unknown assets.
        dependencies = _INPUT_FILES | {args.output_dir.resolve(), Path(sys.executable).resolve()}
        dependencies.update(path for path in args.output_dir.resolve().rglob("*") if path.suffix != ".tmp")

        def _escape(path: Path) -> str:
            return (
                str(path.absolute())
                .replace("\\", "\\\\")
                .replace("$", "$$")
                .replace("#", "\\#")
                .replace(" ", "\\ ")
                .replace(":", "\\:")
            )

        aot_bundle.write_if_changed(
            args.depfile, _escape(args.stamp) + ": " + " ".join(_escape(path) for path in sorted(dependencies)) + "\n"
        )
        args.stamp.parent.mkdir(parents=True, exist_ok=True)
        args.stamp.touch()


if __name__ == "__main__":
    _main()
