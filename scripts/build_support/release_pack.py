# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assemble the xllm release directory consumed as xllm_pack/.

The pack keeps the deployment layout: the xllm wheel, the full CANN
``opp/vendors`` archive and flash-linear-attention wheel, and the Mooncake
runtime built by this xllm compile. etcd is intentionally left out.
"""

import datetime
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile

from scripts.build_support.utils import (
    _get_cann_version_info,
    get_ascend_platform,
    get_cpu_arch,
)
from scripts.logger import logger

# xxhash and zstd stay on the system linker path during the xllm build, so the
# runtime archive has to carry the sonames the binaries were linked against.
_MOONCAKE_SYSTEM_SONAMES = ("libxxhash.so.0", "libzstd.so.1")
# CANN 9.0.0 serves both of these from opp/vendors. custom_xllm_math is the
# xllm_ops install; custom_transformer is the ops-transformer install.
_REQUIRED_VENDOR_PACKAGES = ("custom_transformer", "custom_xllm_math")
_ASCEND_CHIP_TAGS = {"a2": "a2", "a3": "a3", "a5": "a5"}
_SYSTEM_LIBRARY_DIRS = (
    "/usr/lib64",
    "/lib64",
    "/usr/lib/aarch64-linux-gnu",
    "/lib/aarch64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu",
    "/lib/x86_64-linux-gnu",
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_path(path: str) -> None:
    if not os.path.lexists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
        return
    os.remove(path)


def _reset_archive(archive_path: str) -> None:
    _remove_path(archive_path)
    _remove_path(archive_path + ".sha256")


def _write_sha256_sidecar(archive_path: str) -> None:
    archive_name = os.path.basename(archive_path)
    with open(archive_path + ".sha256", "w", encoding="utf-8") as checksum:
        checksum.write(f"{_sha256_file(archive_path)}  {archive_name}\n")


def _clear_release_pack(pack_dir: str) -> None:
    """Remove a previous or partial release directory and its archive."""
    _remove_path(pack_dir)
    _reset_archive(pack_dir + ".tar.gz")


def _git_command(repo_dir: str, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_dir}", "-C", repo_dir, *args],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def _git_text(repo_dir: str, *args: str) -> str:
    """Return git output, or ``unknown`` when this source tree has no git metadata.

    A release pack is assembled after the wheel already exists. An sdist, git
    archive, or cleared submodule must not fail that wheel.
    """
    try:
        output = _git_command(repo_dir, *args)
    except (OSError, subprocess.CalledProcessError):
        logger.exception(f"Git metadata is unavailable for {repo_dir}: git {' '.join(args)}")
        return "unknown"
    return output or "unknown"


def _git_short_hash(repo_dir: str) -> str:
    return _git_text(repo_dir, "rev-parse", "--short=8", "HEAD")


def _git_recent_log(repo_dir: str, count: int = 5) -> str:
    """Return ``git log -n`` text, or ``unknown`` when git metadata is absent."""
    return _git_text(repo_dir, "log", f"-n{count}")


def _git_branch(repo_dir: str) -> str:
    branch = _git_text(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
    if branch == "unknown":
        return "unknown"
    if branch == "HEAD":
        try:
            branch = _git_command(repo_dir, "describe", "--tags", "--exact-match", "HEAD")
        except (OSError, subprocess.CalledProcessError):
            # Detached HEAD without an exact tag is a valid checkout, not missing metadata.
            branch = "HEAD"
    return branch or "HEAD"


def _write_pack_version_file(base_dir: str, pack_dir: str, flash_wheel: str = "") -> str:
    """Record the xllm branch, recent history, and the packed flash-linear-attention wheel."""
    sections = [
        ("xllm", base_dir, True),
        ("xllm_ops", os.path.join(base_dir, "third_party", "xllm_ops"), False),
        ("mooncake", os.path.join(base_dir, "third_party", "Mooncake"), False),
    ]
    parts: list[str] = []
    for name, repo_dir, include_branch in sections:
        lines = [f"===== {name} ====="]
        if include_branch:
            lines.append(f"branch: {_git_branch(repo_dir)}")
            lines.append("")
        lines.append(_git_recent_log(repo_dir))
        parts.append("\n".join(lines))
    if flash_wheel:
        parts.append(
            "\n".join(
                [
                    "===== flash-linear-attention-npu =====",
                    f"wheel: {os.path.basename(flash_wheel)}",
                    f"sha256: {_sha256_file(flash_wheel)}",
                ]
            )
        )

    path = os.path.join(pack_dir, "xllm_version.txt")
    with open(path, "w", encoding="utf-8") as version_file:
        version_file.write("\n\n".join(parts))
        version_file.write("\n")
    logger.info(f"Wrote pack version file: {path}")
    return path


def _read_mooncake_version(base_dir: str) -> str:
    pyproject = os.path.join(base_dir, "third_party", "Mooncake", "mooncake-wheel", "pyproject.toml")
    try:
        with open(pyproject, encoding="utf-8") as source:
            match = re.search(r'^version\s*=\s*"([^"]+)"', source.read(), re.MULTILINE)
    except OSError as error:
        raise RuntimeError(f"Failed to read Mooncake version from {pyproject}: {error}") from error
    if match is None:
        raise RuntimeError(f"Mooncake version was not found in {pyproject}")
    return match.group(1)


def _resolve_system_soname(soname: str) -> str:
    """Return the real file behind a system soname symlink."""
    search_dirs: list[str] = []
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if entry and entry not in search_dirs:
            search_dirs.append(entry)
    search_dirs.extend(path for path in _SYSTEM_LIBRARY_DIRS if path not in search_dirs)
    for directory in search_dirs:
        candidate = os.path.join(directory, soname)
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    raise RuntimeError(f"Mooncake system runtime library was not found: {soname}")


def _chip_tag(device: str) -> str:
    if device != "npu":
        return device
    try:
        platform = get_ascend_platform()
    except Exception:
        logger.exception("Failed to detect the Ascend chip for the release pack name.")
        raise
    chip = _ASCEND_CHIP_TAGS.get(platform)
    if chip is None:
        raise RuntimeError(f"Unsupported Ascend chip for the release pack name: {platform}")
    return chip


def _ascend_home() -> str:
    return os.environ.get("ASCEND_HOME_PATH") or "/usr/local/Ascend/ascend-toolkit/latest"


def _opp_root() -> str:
    """Return the OPP root CMake and utils use to find the xllm_ops vendor."""
    configured = os.environ.get("ASCEND_OPP_PATH", "")
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(_ascend_home(), "opp")


def _cann_tag() -> str:
    """Return the CANN version token used in the vendors archive name.

    ``ascend-toolkit/latest`` resolves to a directory such as ``8.3.RC1``, which
    has no ``cann-`` prefix. ``compiler/version.info`` publishes the full
    ``Version=`` value, including the RC segment.
    """
    version_info_path, version = _get_cann_version_info(_ascend_home())
    if not version:
        raise RuntimeError(
            f"CANN Version= was not found in {version_info_path}. The vendors archive name needs the full CANN version."
        )
    sanitized = re.sub(r"[^A-Za-z0-9.]", "", version)
    if not sanitized:
        raise RuntimeError(f"CANN version {version!r} from {version_info_path} cannot be used in an archive name.")
    return f"cann{sanitized}"


def _select_newest_file(pattern: str, description: str) -> str:
    """Return the newest file matching ``pattern``.

    The flash-linear-attention dist directory is not cleaned by the xllm build,
    so an upgrade can leave the previous wheel beside the new one. Pick the
    newest mtime, matching the wheel installed by build_flash_linear_attention.sh.
    """
    matches = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    if not matches:
        raise RuntimeError(f"No {description} matching {pattern}.")
    selected = max(matches, key=lambda path: (os.path.getmtime(path), os.path.basename(path)))
    if len(matches) > 1:
        skipped = sorted(
            os.path.basename(path) for path in matches if os.path.abspath(path) != os.path.abspath(selected)
        )
        logger.warning(
            f"Found {len(matches)} {description} files. Packing the newest {os.path.basename(selected)} "
            f"and skipping: {', '.join(skipped)}."
        )
    else:
        logger.info(f"Packing {description}: {os.path.basename(selected)}")
    return selected


# The NPU wheel ships three statically linked copies of the engine. brpc keeps
# WITH_DEBUG_SYMBOLS on, so each copy carries the same ~600 MiB DWARF image.
# Deployment installs that wheel on every start and then scans the server
# binary; the debug sections are not needed to run it.
def _is_release_debug_binary(filename: str) -> bool:
    if filename in {"xllm", "spawn_worker", "mooncake_master", "mooncake_client"}:
        return True
    return filename.startswith("xllm_export") and ".so" in filename


def strip_release_debug_symbols(package_dir: str) -> None:
    """Drop DWARF from the binaries that bdist_wheel is about to package.

    ``strip --strip-debug`` keeps the dynamic symbol table, so pybind and the
    runtime sonames still resolve. A missing strip tool leaves the binaries
    unchanged and fails the wheel instead of publishing the slow debug copies.
    """
    if not os.path.isdir(package_dir):
        raise RuntimeError(f"Release package directory does not exist: {package_dir}")
    binaries = [
        os.path.join(package_dir, filename)
        for filename in sorted(os.listdir(package_dir))
        if _is_release_debug_binary(filename) and os.path.isfile(os.path.join(package_dir, filename))
    ]
    missing = [name for name in ("xllm", "spawn_worker") if not os.path.isfile(os.path.join(package_dir, name))]
    export_found = any(os.path.basename(path).startswith("xllm_export") for path in binaries)
    if missing or not export_found:
        raise RuntimeError(
            f"Release binaries missing under {package_dir}: {missing or ['xllm_export']}. "
            "Refusing to pack debug copies."
        )
    for binary in binaries:
        before = os.path.getsize(binary)
        subprocess.check_call(["strip", "--strip-debug", binary])
        after = os.path.getsize(binary)
        logger.info(f"Stripped debug symbols: {binary} ({before} -> {after} bytes)")


def _copy_release_file(source: str, destination: str, executable: bool = False) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)
    if executable:
        mode = os.stat(destination).st_mode
        os.chmod(destination, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_mooncake_runtime_scripts(root: str) -> None:
    env_script = """#!/usr/bin/env bash

MOONCAKE_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOONCAKE_HOME
export PATH="$MOONCAKE_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MOONCAKE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"""
    check_script = """#!/usr/bin/env bash

set -euo pipefail

MOONCAKE_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$MOONCAKE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for binary in mooncake_master mooncake_client; do
  binary_path="$MOONCAKE_HOME/bin/$binary"
  [[ -x "$binary_path" ]] || {
    echo "ERROR: missing executable: $binary_path" >&2
    exit 1
  }
  # Keep ldd's own status. A pipeline inside if hides a failed ldd behind grep.
  ldd_status=0
  ldd_output="$(ldd "$binary_path" 2>&1)" || ldd_status=$?
  if [[ "$ldd_status" -ne 0 ]]; then
    echo "ERROR: ldd could not check $binary_path (exit ${ldd_status})" >&2
    printf '%s\n' "$ldd_output" >&2
    exit 1
  fi
  if grep -q 'not found' <<<"$ldd_output"; then
    echo "ERROR: unresolved libraries for $binary_path" >&2
    grep 'not found' <<<"$ldd_output" >&2
    exit 1
  fi
done

if [[ -f "$MOONCAKE_HOME/SHA256SUMS" ]]; then
  (cd "$MOONCAKE_HOME" && sha256sum -c SHA256SUMS)
fi

echo "Mooncake runtime check passed: $MOONCAKE_HOME"
"""
    for name, content in (("env.sh", env_script), ("check.sh", check_script)):
        path = os.path.join(root, name)
        with open(path, "w", encoding="utf-8") as script:
            script.write(content)
        os.chmod(path, 0o755)


def _build_mooncake_runtime_archive(
    base_dir: str,
    extdir: str,
    cmake_dir: str,
    output_dir: str,
    xllm_wheel: str,
) -> str:
    """Pack the Mooncake master/client built with this xllm wheel."""
    version = _read_mooncake_version(base_dir)
    package_name = f"mooncake-v{version}"
    archive_name = f"mooncake-runtime-v{version}.tar.gz"
    staging_root = os.path.join(output_dir, package_name)
    if os.path.isdir(staging_root):
        shutil.rmtree(staging_root)

    binaries = {
        "mooncake_master": os.path.join(extdir, "mooncake_master"),
        "mooncake_client": os.path.join(extdir, "mooncake_client"),
    }
    libraries = {
        "libasio.so": os.path.join(cmake_dir, "mooncake-common", "libasio.so"),
        "libetcd_wrapper.so": os.path.join(extdir, "libetcd_wrapper.so"),
        "liburing.so.2": os.path.join(extdir, "liburing.so.2"),
    }
    for soname in _MOONCAKE_SYSTEM_SONAMES:
        libraries[soname] = _resolve_system_soname(soname)

    origins: dict[str, dict[str, str]] = {}
    for name, source in binaries.items():
        if not os.path.isfile(source):
            raise RuntimeError(f"Mooncake binary was not staged by the wheel build: {source}")
        destination = os.path.join(staging_root, "bin", name)
        _copy_release_file(source, destination, executable=True)
        origins[f"bin/{name}"] = {"wheel_member": f"xllm/{name}", "sha256": _sha256_file(destination)}
    for soname, source in libraries.items():
        if not os.path.isfile(source):
            raise RuntimeError(f"Mooncake runtime library was not found: {source}")
        destination = os.path.join(staging_root, "lib", soname)
        _copy_release_file(source, destination, executable=True)
        member = f"lib/{soname}"
        if soname in _MOONCAKE_SYSTEM_SONAMES:
            origins[member] = {"system_source": source, "sha256": _sha256_file(destination)}
        else:
            origins[member] = {"wheel_member": f"xllm/{soname}", "sha256": _sha256_file(destination)}

    with open(os.path.join(staging_root, "VERSION"), "w", encoding="utf-8") as version_file:
        version_file.write(f"v{version}\n")
    mooncake_dir = os.path.join(base_dir, "third_party", "Mooncake")
    build_info = {
        "MOONCAKE_PACKAGE_LAYOUT": "1",
        "MOONCAKE_VERSION": f"v{version}",
        "MOONCAKE_SOURCE_COMMIT": _git_short_hash(mooncake_dir),
        "XLLM_SOURCE_COMMIT": _git_short_hash(base_dir),
    }
    with open(os.path.join(staging_root, "build-info.env"), "w", encoding="utf-8") as info:
        for key, value in build_info.items():
            info.write(f"{key}={value}\n")
    _write_mooncake_runtime_scripts(staging_root)
    manifest = {
        "package": package_name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mooncake_commit": build_info["MOONCAKE_SOURCE_COMMIT"],
        "xllm_commit": build_info["XLLM_SOURCE_COMMIT"],
        "source_wheel_sha256": _sha256_file(xllm_wheel),
        "architecture": platform.machine(),
        "origins": origins,
    }
    with open(os.path.join(staging_root, "MANIFEST.txt"), "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")

    checksum_path = os.path.join(staging_root, "SHA256SUMS")
    with open(checksum_path, "w", encoding="utf-8") as checksums:
        for dirpath, _, filenames in os.walk(staging_root):
            for filename in sorted(filenames):
                path = os.path.join(dirpath, filename)
                if os.path.abspath(path) == os.path.abspath(checksum_path):
                    continue
                relative = os.path.relpath(path, staging_root)
                checksums.write(f"{_sha256_file(path)}  {relative}\n")

    archive_path = os.path.join(output_dir, archive_name)
    _reset_archive(archive_path)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(staging_root, arcname=package_name)
    shutil.rmtree(staging_root)
    _write_sha256_sidecar(archive_path)
    logger.info(f"Packed Mooncake runtime: {archive_path}")
    return archive_path


def _stage_cann90_aicpu_alias(vendor_root: str, cann_tag: str) -> None:
    """Keep CANN 9.0's math AICPU lookup inside the staged math vendor.

    Older xllm_ops exports name the SMLA CPU library ``libtransformer`` in
    both the binary and JSON, but CANN 9.0 resolves ``libmath`` at launch.
    A relative alias preserves those exports without changing shared CANN
    files or borrowing the unrelated custom_transformer CPU implementation.
    """
    if not cann_tag.startswith("cann9.0."):
        return
    cpu_root = os.path.join(vendor_root, "custom_xllm_math", "op_impl", "cpu")
    config_path = os.path.join(cpu_root, "config", "cust_aicpu_kernel.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    metadata = config.get("SmlaVaMetadata", {}).get("opInfo", {})
    source_name = "libtransformer_aicpu_kernels.so"
    if metadata.get("kernelSo") != source_name:
        return

    library_dir = os.path.join(cpu_root, "aicpu_kernel", "impl")
    alias_path = os.path.join(library_dir, "libmath_aicpu_kernels.so")
    if os.path.lexists(alias_path):
        if not os.path.isfile(alias_path):
            raise RuntimeError(f"CANN 9.0 math AICPU library is not a readable file: {alias_path}")
        return
    source_path = os.path.join(library_dir, source_name)
    if not os.path.isfile(source_path):
        raise RuntimeError(f"SmlaVaMetadata CPU library is missing from its own vendor: {source_path}")
    os.symlink(source_name, alias_path)
    logger.info(f"Staged CANN 9.0 math AICPU compatibility alias: {alias_path} -> {source_name}")


def _stage_vendors_archive(base_dir: str, output_dir: str, device: str) -> str:
    """Repack the whole installed ``opp/vendors`` tree consumed by deployment.

    CANN 9.0.0 keeps every custom operator package in this directory, including
    ``custom_transformer``, ``custom_xllm_math``, and ``config.ini``. Deployment
    installs that directory as one unit, so a single vendor package is not enough.
    """
    vendor_root = os.path.join(_opp_root(), "vendors")
    if not os.path.isdir(vendor_root):
        raise RuntimeError(
            f"CANN opp/vendors directory was not found: {vendor_root}. "
            "The xllm NPU build installs it through third_party/xllm_ops."
        )
    if not os.path.isfile(os.path.join(vendor_root, "config.ini")):
        raise RuntimeError(f"CANN opp/vendors is missing config.ini: {vendor_root}")
    missing = [name for name in _REQUIRED_VENDOR_PACKAGES if not os.path.isdir(os.path.join(vendor_root, name))]
    if missing:
        raise RuntimeError(
            f"CANN opp/vendors is missing required packages {missing} under {vendor_root}. "
            "custom_xllm_math comes from third_party/xllm_ops/build.sh and "
            "custom_transformer comes from third_party/xllm_ops/build_ops_transformer.sh."
        )

    ops_dir = os.path.join(base_dir, "third_party", "xllm_ops")
    cann_tag = _cann_tag()
    archive_name = f"vendors-{_git_short_hash(ops_dir)}-{cann_tag}-{get_cpu_arch()}-{_chip_tag(device)}.tar.gz"
    staging_root = os.path.join(output_dir, "vendors")
    if os.path.isdir(staging_root):
        shutil.rmtree(staging_root)
    shutil.copytree(vendor_root, staging_root, symlinks=True)
    _stage_cann90_aicpu_alias(staging_root, cann_tag)

    archive_path = os.path.join(output_dir, archive_name)
    _reset_archive(archive_path)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(staging_root, arcname="vendors")
    shutil.rmtree(staging_root)
    _write_sha256_sidecar(archive_path)
    logger.info(f"Packed CANN opp/vendors: {archive_path}")
    return archive_path


def collect_xllm_release_pack(
    base_dir: str,
    extdir: str,
    cmake_dir: str,
    device: str,
    dist_dir: str,
    wheel_path: str,
) -> str:
    """Assemble the NPU release directory next to the source tree.

    MLU and the other device builds still produce their wheel. The release
    directory and archive are NPU-only until those platforms have a pack layout.
    """
    if device != "npu":
        logger.info(f"Skip xllm release pack for device={device}. The release pack is NPU-only.")
        return ""

    pack_dir = os.path.join(base_dir, "xllm_pack")
    archive_path = pack_dir + ".tar.gz"
    _clear_release_pack(pack_dir)
    try:
        os.makedirs(pack_dir)

        wheel_destination = os.path.join(pack_dir, os.path.basename(wheel_path))
        shutil.copy2(wheel_path, wheel_destination)

        ops_dir = os.path.join(base_dir, "third_party", "xllm_ops")
        flash_wheel = _select_newest_file(
            os.path.join(
                ops_dir, "third_party", "flash-linear-attention-npu", "dist", "flash_linear_attention_npu_*.whl"
            ),
            "flash-linear-attention-npu wheel",
        )
        shutil.copy2(flash_wheel, os.path.join(pack_dir, os.path.basename(flash_wheel)))
        _stage_vendors_archive(base_dir, pack_dir, device)
        _build_mooncake_runtime_archive(base_dir, extdir, cmake_dir, pack_dir, wheel_destination)
        _write_pack_version_file(base_dir, pack_dir, flash_wheel)

        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(pack_dir, arcname=os.path.basename(pack_dir))
        _write_sha256_sidecar(archive_path)
    except Exception:
        _clear_release_pack(pack_dir)
        raise

    logger.info("Release pack contents:")
    for name in sorted(os.listdir(pack_dir)):
        logger.info(f"  {name}")
    logger.info(f"✅ xllm release pack: {pack_dir}")
    logger.info(f"✅ xllm release archive: {archive_path} (wheel from {dist_dir})")
    return pack_dir
