#!/usr/bin/env python3
"""Reuse dependencies, rebuild named TUs into private archives, record provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


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
    args = parser.parse_args()
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
                "sources": [], "archives": {}}
    (dest / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=root))
    archives = {}
    for source_name in args.sources:
        source = (root / source_name).resolve()
        assert source.is_relative_to(root), source
        entry = next(item for item in commands if Path(item["file"]).resolve() == source)
        argv = shlex.split(entry["command"])
        while not argv[0].startswith("-D"):
            argv.pop(0)
        argv = [arg.replace(str(root / "third_party"), donor) for arg in argv
                if not arg.startswith("-fplugin=")]
        if "-x" in argv:
            language = argv.index("-x")
            assert argv[language + 1] == "musa"
            del argv[language:language + 2]
        output_index = argv.index("-o") + 1
        base_object = build / argv[output_index]
        obj = dest / (source.name + ".o")
        argv[output_index] = str(obj)
        command = ["g++", *include_args, *argv]
        (dest / (source.name + ".command.txt")).write_text(shlex.join(command) + "\n")
        shutil.copyfile(source, dest / source.name)
        with (dest / (source.name + ".compile.log")).open("w") as log:
            print("Compiling", source.name, flush=True)
            subprocess.run(command, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if "musa_layers.dir" in str(base_object):
            archive_relative = "xllm/core/layers/musa/libmusa_layers.a"
        elif "runtime.dir" in str(base_object):
            archive_relative = "xllm/core/runtime/libruntime.a"
        else:
            raise ValueError("Add explicit archive mapping for " + str(base_object))
        archive = dest / Path(archive_relative).name
        if archive_relative not in archives:
            original = build / archive_relative
            manifest["archives"][archive_relative] = {"base_sha256": sha(original)}
            shutil.copyfile(original, archive)
            archives[archive_relative] = archive
        subprocess.run(["ar", "r", str(archive), str(obj)], check=True)
        manifest["sources"].append({"path": source_name, "sha256": sha(source), "object_sha256": sha(obj)})
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
    assert " -o xllm/xllm " in link
    link = link.replace(" -o xllm/xllm ", " -o " + shlex.quote(str(dest / "xllm")) + " ")
    (dest / "link.sh").write_text(link + "\n")
    with (dest / "link.log").open("w") as log:
        subprocess.run(["bash", str(dest / "link.sh")], cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest["binary_sha256"] = sha(dest / "xllm")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest["binary_sha256"], str(dest / "xllm"), flush=True)


if __name__ == "__main__":
    main()
