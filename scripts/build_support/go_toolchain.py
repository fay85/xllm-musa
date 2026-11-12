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

"""Select a healthy, pinned Go toolchain without modifying the development image."""

import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

from scripts.logger import logger

_GO_DOWNLOAD_MIRRORS = ("https://go.dev/dl/", "https://golang.google.cn/dl/", "https://mirrors.aliyun.com/golang/")


def _probe_environment(go_root: str | None = None) -> dict[str, str]:
    environment = dict(os.environ, GOTOOLCHAIN="local", GOENV="off", GOWORK="off", GOFLAGS="", CGO_ENABLED="0")
    environment.pop("GOROOT", None)
    if go_root:
        environment["GOROOT"] = go_root
    return environment


def _probe_toolchain(go_binary: str, required_version: str) -> str | None:
    """A matching binary version does not guarantee that its standard library is intact."""
    try:
        with tempfile.TemporaryDirectory(prefix="xllm-go-probe-") as temp_dir:
            result = subprocess.run(
                [go_binary, "env", "GOROOT", "GOVERSION"],
                cwd=temp_dir,
                env=_probe_environment(),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            values = result.stdout.strip().splitlines()
            if result.returncode or len(values) != 2 or values[1] != f"go{required_version}":
                return None
            go_root = values[0]
            source = Path(temp_dir) / "probe.go"
            source.write_text('package main\nimport "runtime"\nfunc main() { _ = runtime.GOOS }\n', encoding="utf-8")
            result = subprocess.run(
                [go_binary, "build", "-o", str(Path(temp_dir) / "probe"), str(source)],
                cwd=temp_dir,
                env=_probe_environment(go_root),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode:
                logger.warning(f"Reject broken Go {required_version} at {go_root}: {result.stderr.strip()}")
                return None
            return go_root
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.debug(f"Go toolchain probe failed for {go_binary}: {error}")
        return None


def _export_toolchain(go_root: str) -> None:
    bin_dir = os.path.join(go_root, "bin")
    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry and entry != bin_dir]
    os.environ.update(PATH=os.pathsep.join([bin_dir, *path_entries]), GOROOT=go_root, GOTOOLCHAIN="local")
    logger.info(f"Use Mooncake Go toolchain: {go_root}")


def _get_archive_checksum(filename: str) -> str:
    # Official release metadata covers old pins as well as currently supported releases.
    for mirror in _GO_DOWNLOAD_MIRRORS[:2]:
        try:
            with urlopen(f"{mirror}?mode=json&include=all", timeout=30) as response:
                releases = json.load(response)
            for release in releases:
                for archive in release["files"]:
                    if archive["filename"] == filename and re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]):
                        return archive["sha256"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning(f"Failed to read Go release metadata from {mirror}: {error}")
    raise RuntimeError(f"Cannot find an official SHA256 checksum for {filename}")


def _download_archive(filename: str, destination: Path) -> None:
    checksum = _get_archive_checksum(filename)
    for mirror in _GO_DOWNLOAD_MIRRORS:
        try:
            digest = hashlib.sha256()
            with urlopen(f"{mirror}{filename}", timeout=30) as response, destination.open("wb") as archive:
                while chunk := response.read(1024 * 1024):
                    archive.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != checksum:
                raise ValueError("Go archive SHA256 mismatch")
            return
        except (OSError, ValueError) as error:
            logger.warning(f"Failed to download Go from {mirror}: {error}")
    raise RuntimeError(f"Failed to download and verify {filename} from all mirrors")


def _extract_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        # Official Go archives only contain regular files and directories under go/.
        # Check all members before extraction, including on supported Python 3.10 builds.
        for member in tar.getmembers():
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != "go"
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe Go archive member: {member.name}")
        tar.extractall(destination)


def ensure_go_toolchain(required_version: str) -> None:
    """Reuse a healthy image toolchain or install a clean, versioned user-cache copy."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", required_version):
        raise ValueError(f"Invalid Mooncake Go version: {required_version}")
    architecture = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64", "amd64": "amd64"}.get(platform.machine())
    if platform.system() != "Linux" or architecture is None:
        raise RuntimeError(f"Unsupported Go platform: {platform.system()} {platform.machine()}")

    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    cache_dir = cache_home / "xllm" / "toolchains"
    install_dir = cache_dir / f"go{required_version}-linux-{architecture}"
    candidates = dict.fromkeys(
        filter(None, [shutil.which("go"), str(install_dir / "bin" / "go"), "/usr/local/go/bin/go"])
    )
    for go_binary in candidates:
        go_root = _probe_toolchain(go_binary, required_version)
        if go_root:
            _export_toolchain(go_root)
            return

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Serialize replacement so parallel setup.py invocations never merge Go trees.
    with (cache_dir / f".{install_dir.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        go_root = _probe_toolchain(str(install_dir / "bin" / "go"), required_version)
        if go_root:
            _export_toolchain(go_root)
            return
        with tempfile.TemporaryDirectory(prefix=f".{install_dir.name}-", dir=cache_dir) as temp_dir:
            staging_dir = Path(temp_dir)
            archive = staging_dir / "go.tar.gz"
            filename = f"go{required_version}.linux-{architecture}.tar.gz"
            logger.info(f"Install clean Mooncake Go {required_version} in {install_dir}")
            _download_archive(filename, archive)
            _extract_archive(archive, staging_dir)
            go_root = _probe_toolchain(str(staging_dir / "go" / "bin" / "go"), required_version)
            if not go_root:
                raise RuntimeError(f"Downloaded Go {required_version} failed its compilation probe")
            if install_dir.is_symlink():
                install_dir.unlink()
            elif install_dir.exists():
                shutil.rmtree(install_dir)
            (staging_dir / "go").rename(install_dir)
        _export_toolchain(str(install_dir))
