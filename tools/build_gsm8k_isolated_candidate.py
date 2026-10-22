#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Reuse dependencies, rebuild named TUs into private archives, record provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


DSPARK_CURRENT_SOURCES = (
    "xllm/core/layers/musa/deepseek_v4/deepseek_v4_attention.cpp",
    "xllm/core/layers/musa/deepseek_v4/dsa_metadata_builder_musa.cpp",
    "xllm/core/runtime/musa_graph_executor_impl.cpp",
    "xllm/core/runtime/dflash_worker_impl.cpp",
)


def _complete_dspark_sources(sources: list[str]) -> list[str]:
    """Do not silently combine current Attention with stale DSpark runtime."""
    requested = list(dict.fromkeys(sources))
    if any(source in DSPARK_CURRENT_SOURCES for source in requested):
        return list(dict.fromkeys([*requested, *DSPARK_CURRENT_SOURCES]))
    return requested


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else digest_stream(stream)


def digest_stream(stream):
    digest = hashlib.sha256()
    for data in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(data)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("sources", nargs="+")
    parser.add_argument(
        "--no-dspark-source-complete",
        action="store_true",
        help="Compile exactly the supplied sources for historical A/B candidates.",
    )
    parser.add_argument(
        "--kernel-archive",
        default="",
        help="Use an explicit MUSA kernel archive for a reproducible archive A/B.",
    )
    parser.add_argument(
        "--kernel-base-archive", type=Path,
        help="Read-only donor for a private kernel archive with selected TUs rebuilt.",
    )
    parser.add_argument(
        "--legacy-kernel-compat",
        action="store_true",
        help="Add the old-kernel state-gather ABI compatibility wrapper.",
    )
    args = parser.parse_args()
    if args.kernel_base_archive and (args.kernel_archive or
                                    not args.kernel_base_archive.is_file()):
        parser.error("kernel base must exist and cannot be combined with kernel override")
    sources = (list(dict.fromkeys(args.sources)) if args.no_dspark_source_complete
               else _complete_dspark_sources(args.sources))
    if args.legacy_kernel_compat:
        sources.append("tools/deepseek_v4_legacy_kernel_compat.cpp")
    dest = (root / args.output_dir).resolve()
    if not dest.is_relative_to(root / "build") or dest.exists():
        parser.error("output must be a NEW directory under this checkout's build/")
    dest.mkdir(parents=True)
    (dest / "tmp").mkdir()
    env = dict(os.environ, TMPDIR=str(dest / "tmp"), PYTHONDONTWRITEBYTECODE="1")
    build = root / "build/dspark-musa-local"
    commands = json.loads((build / "compile_commands.json").read_text())
    donor = "/data/nfs_shared/jiaming.wang/xllmmerge/xllm0814/xllm-dsv4-musa-dspark/third_party"
    include_args = ["-I/usr/local/musa/include", "-I/usr/local/lib/python3.10/dist-packages",
                    "-I/usr/local/lib/python3.10/dist-packages/tvm_ffi/include",
                    "-I/data/nfs_shared/jiaming.wang/xllmmerge/xllm/build/musa-standalone-db33797/musa-msgpack-manual/include"]
    manifest = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "requested_sources": args.sources,
                "required_dspark_sources": list(DSPARK_CURRENT_SOURCES)
                if not args.no_dspark_source_complete and
                any(source in DSPARK_CURRENT_SOURCES for source in sources)
                else [],
                "kernel_archive": str(Path(args.kernel_archive).resolve())
                if args.kernel_archive
                else "",
                "sources": [], "archives": {}}
    (dest / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=root))
    manifest["local_policy_headers"] = []
    for name in ("xllm/core/layers/musa/deepseek_v4/deepseek_v4_graph_policy.h",
                 "xllm/core/distributed_runtime/speculative_engine.h"):
        header = root / name
        if header.is_file():
            snapshot = dest / "headers" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(header, snapshot)
            manifest["local_policy_headers"].append(
                {"path": name, "sha256": sha(header), "snapshot": str(snapshot)})
    archives = {}
    for source_name in sources:
        source = (root / source_name).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        source_text = source.as_posix()
        if source.name in {"deepseek_v4_legacy_kernel_compat.cpp",
                           "dsv4_metadata_compat.cpp"} or \
                source.name.endswith("_hybrid_attention.cpp"):
            source_suffix = "/xllm/core/layers/musa/deepseek_v4/deepseek_v4_attention.cpp"
        else:
            source_suffix = (
                "/xllm/" + source_text.split("/xllm/", 1)[1]
                if "/xllm/" in source_text
                else "/" + source.name
            )
        entry = next(
            item
            for item in commands
            if Path(item["file"]).as_posix().endswith(source_suffix)
        )
        argv = shlex.split(entry["command"])
        is_kernel = source.suffix == ".cu"
        if is_kernel:
            compiler = argv.pop(0)
        else:
            compiler = "g++"
            while not argv[0].startswith("-D"):
                argv.pop(0)
            argv = [arg for arg in argv if not arg.startswith("-fplugin=")]
            if "-x" in argv:
                language = argv.index("-x")
                assert argv[language + 1] == "musa"
                del argv[language:language + 2]
        argv = [arg.replace(str(root / "third_party"), donor) for arg in argv]
        # A compile-database entry supplies flags, NOT the requested source.
        # Historical A/B used to hash a snapshot while compiling the live TU.
        source_index = argv.index("-c") + 1
        compile_source = source
        if not source.is_relative_to(root):
            # Compile snapshots from this checkout's include root. Quoted
            # includes otherwise resolve against the snapshot directory first
            # and silently select an older ABI header.
            compile_source = dest / source.name
            shutil.copyfile(source, compile_source)
        argv[source_index] = str(compile_source)
        assert Path(argv[source_index]).resolve() == compile_source.resolve()
        output_index = argv.index("-o") + 1
        base_object = build / argv[output_index]
        obj = dest / (source.name + ".o")
        argv[output_index] = str(obj)
        command = [compiler, *include_args, *argv]
        (dest / (source.name + ".command.txt")).write_text(shlex.join(command) + "\n")
        if compile_source != dest / source.name:
            shutil.copyfile(source, dest / source.name)
        compiled_source_sha = sha(compile_source)
        if sha(dest / source.name) != compiled_source_sha:
            raise RuntimeError(f"Source changed while snapshotting: {source}")
        with (dest / (source.name + ".compile.log")).open("w") as log:
            print("Compiling", source.name, flush=True)
            subprocess.run(command, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if sha(compile_source) != compiled_source_sha or sha(source) != compiled_source_sha:
            raise RuntimeError(f"Source changed during compile; do not link mixed provenance: {source}")
        # Some musamapping wrappers return zero even when the device compiler
        # fails. Each candidate uses a new directory; never link an old object.
        if not obj.is_file() or obj.stat().st_size == 0:
            raise RuntimeError(f"Compiler produced no object: {obj}; inspect compile.log")
        if source.name in {"deepseek_v4_legacy_kernel_compat.cpp",
                           "dsv4_metadata_compat.cpp"}:
            archive_relative = "xllm/core/runtime/libruntime.a"
        elif "musa_layers.dir" in str(base_object):
            archive_relative = "xllm/core/layers/musa/libmusa_layers.a"
        elif "runtime.dir" in str(base_object):
            archive_relative = (
                "xllm/core/distributed_runtime/libdistributed_runtime.a"
                if "distributed_runtime.dir" in str(base_object)
                else "xllm/core/runtime/libruntime.a")
        elif "parallel_state.dir" in str(base_object):
            archive_relative = "xllm/core/framework/parallel_state/libparallel_state.a"
        elif "CMakeFiles/sampler.dir" in str(base_object):
            archive_relative = "xllm/core/framework/sampling/libsampler.a"
        elif "deepseek_v4_cpp_template.dir" in str(base_object):
            archive_relative = "xllm/core/framework/chat_template/libdeepseek_v4_cpp_template.a"
        elif "cuda_kernels.dir" in str(base_object):
            archive_relative = "xllm/core/kernels/musa/libcuda_kernels.a"
        else:
            raise ValueError("Add explicit archive mapping for " + str(base_object))
        archive = dest / Path(archive_relative).name
        if archive_relative not in archives:
            original = build / archive_relative
            if (archive_relative == "xllm/core/kernels/musa/libcuda_kernels.a"
                    and args.kernel_base_archive):
                original = args.kernel_base_archive.resolve()
            manifest["archives"][archive_relative] = {"base_sha256": sha(original)}
            manifest["archives"][archive_relative]["base_path"] = str(original)
            shutil.copyfile(original, archive)
            archives[archive_relative] = archive
        # Explicitly replace the donor member and verify its actual bytes.
        subprocess.run(["ar", "d", str(archive), obj.name], check=False)
        subprocess.run(["ar", "r", str(archive), str(obj)], check=True)
        member = subprocess.check_output(["ar", "p", str(archive), obj.name])
        assert hashlib.sha256(member).hexdigest() == sha(obj)
        manifest["sources"].append({"path": source_name, "compiled_source": str(source), "sha256": compiled_source_sha, "object_sha256": sha(obj)})
    for relative, archive in archives.items():
        subprocess.run(["ranlib", str(archive)], check=True)
        manifest["archives"][relative]["candidate_sha256"] = sha(archive)
    link = subprocess.check_output(["ninja", "-C", str(build), "-t", "commands", "xllm"], text=True).splitlines()[-1]
    link = link.replace("/opt/xllm-build-deps/compiler_launcher.sh ", "")
    link = link.replace("/usr/local/lib/libzstd.so", "/usr/lib/x86_64-linux-gnu/libzstd.so.1")
    for relative, archive in archives.items():
        # Handle BOTH whole-archive absolute paths and repeated relative group members.
        link = link.replace(str(build / relative), str(archive)).replace(relative, str(archive))
        assert relative not in link, relative
    if args.kernel_archive:
        if "xllm/core/kernels/musa/libcuda_kernels.a" in archives:
            parser.error("cannot combine kernel sources with --kernel-archive")
        kernel_archive = Path(args.kernel_archive).resolve()
        if not kernel_archive.is_file():
            parser.error(f"kernel archive does not exist: {kernel_archive}")
        kernel_relative = "xllm/core/kernels/musa/libcuda_kernels.a"
        link = link.replace(str(build / kernel_relative), str(kernel_archive))
        link = link.replace(kernel_relative, str(kernel_archive))
        manifest["kernel_archive_sha256"] = sha(kernel_archive)
    assert " -o xllm/xllm " in link
    link = link.replace(" -o xllm/xllm ", " -o " + shlex.quote(str(dest / "xllm")) + " ")
    (dest / "link.sh").write_text(link + "\n")
    with (dest / "link.log").open("w") as log:
        subprocess.run(["bash", str(dest / "link.sh")], cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not (dest / "xllm").is_file():
        raise RuntimeError(f"Link driver returned success without executable; inspect {dest / 'link.log'}")
    manifest["binary_sha256"] = sha(dest / "xllm")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest["binary_sha256"], str(dest / "xllm"), flush=True)


if __name__ == "__main__":
    main()
