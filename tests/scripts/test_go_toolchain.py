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

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.build_support import go_toolchain as go


class GoToolchainTest(unittest.TestCase):
    def _make_archive(self, destination: Path) -> None:
        with tarfile.open(destination, "w:gz") as archive:
            for name, content in (("go/bin/go", b"new"), ("go/src/internal/abi/map.go", b"new map")):
                member = tarfile.TarInfo(name)
                member.size = len(content)
                member.mode = 0o755 if name.endswith("/bin/go") else 0o644
                archive.addfile(member, io.BytesIO(content))

    def test_healthy_updated_image_is_reused_without_download(self) -> None:
        with (
            mock.patch.object(go.shutil, "which", return_value="/opt/go/bin/go"),
            mock.patch.object(go, "_probe_toolchain", return_value="/opt/go"),
            mock.patch.object(go, "_download_archive") as download,
            mock.patch.dict(os.environ, {"PATH": "/usr/bin", "GOROOT": "/old/go", "GOTOOLCHAIN": "auto"}),
        ):
            go.ensure_go_toolchain("1.26.10")
            self.assertEqual(os.environ["PATH"], "/opt/go/bin:/usr/bin")
            self.assertEqual(os.environ["GOROOT"], "/opt/go")
            self.assertEqual(os.environ["GOTOOLCHAIN"], "local")
            download.assert_not_called()

    def test_probe_ignores_inherited_root_and_automatic_toolchain(self) -> None:
        with (
            mock.patch.dict(os.environ, {"GOROOT": "/old/go", "GOTOOLCHAIN": "auto", "GOFLAGS": "-invalid"}),
            mock.patch.object(
                go.subprocess,
                "run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, "/new/go\ngo1.26.10\n", ""),
                    subprocess.CompletedProcess([], 0, "", ""),
                ],
            ) as run,
        ):
            self.assertEqual(go._probe_toolchain("/new/go/bin/go", "1.26.10"), "/new/go")
            self.assertNotIn("GOROOT", run.call_args_list[0].kwargs["env"])
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["env"]["GOTOOLCHAIN"], "local")
                self.assertEqual(call.kwargs["env"]["GOENV"], "off")
                self.assertEqual(call.kwargs["env"]["GOFLAGS"], "")
            self.assertEqual(run.call_args_list[1].kwargs["env"]["GOROOT"], "/new/go")

    def test_matching_version_with_broken_standard_library_is_rejected(self) -> None:
        with mock.patch.object(
            go.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess([], 0, "/mixed/go\ngo1.26.10\n", ""),
                subprocess.CompletedProcess([], 1, "", "ctrlEmpty redeclared in this block"),
            ],
        ):
            self.assertIsNone(go._probe_toolchain("/mixed/go/bin/go", "1.26.10"))

    def test_old_or_newer_image_version_is_not_selected(self) -> None:
        for version in ("1.25.10", "1.27.3"):
            with (
                self.subTest(version=version),
                mock.patch.object(
                    go.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, f"/image/go\ngo{version}\n", ""),
                ) as run,
            ):
                self.assertIsNone(go._probe_toolchain("/image/go/bin/go", "1.26.10"))
                run.assert_called_once()

    def test_missing_or_timed_out_toolchain_is_rejected(self) -> None:
        for error in (FileNotFoundError("no go"), subprocess.TimeoutExpired("go", 30)):
            with self.subTest(error=error), mock.patch.object(go.subprocess, "run", side_effect=error):
                self.assertIsNone(go._probe_toolchain("/missing/go", "1.26.10"))

    def test_broken_cache_is_replaced_without_touching_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "image-go"
            image_root.mkdir()
            (image_root / "old-image-file").write_text("keep", encoding="utf-8")
            install_dir = root / "xllm/toolchains/go1.26.10-linux-arm64"
            (install_dir / "src/internal/abi").mkdir(parents=True)
            stale_file = install_dir / "src/internal/abi/map_swiss.go"
            stale_file.write_text("old map", encoding="utf-8")
            tarball = root / "download.tar.gz"
            self._make_archive(tarball)

            def probe(binary: str, version: str) -> str | None:
                go_root = Path(binary).parent.parent
                if (go_root / "bin/go").is_file() and not (go_root / "src/internal/abi/map_swiss.go").exists():
                    return str(go_root)
                return None

            def download(filename: str, destination: Path) -> None:
                self.assertEqual(filename, "go1.26.10.linux-arm64.tar.gz")
                shutil.copyfile(tarball, destination)

            with (
                mock.patch.dict(
                    os.environ, {"XDG_CACHE_HOME": temp_dir, "PATH": "/usr/bin", "GOROOT": str(image_root)}
                ),
                mock.patch.object(go.platform, "machine", return_value="aarch64"),
                mock.patch.object(go.shutil, "which", return_value=str(image_root / "bin/go")),
                mock.patch.object(go, "_probe_toolchain", side_effect=probe),
                mock.patch.object(go, "_download_archive", side_effect=download) as fetch,
            ):
                go.ensure_go_toolchain("1.26.10")
                self.assertFalse(stale_file.exists())
                self.assertEqual((install_dir / "bin/go").read_bytes(), b"new")
                self.assertTrue((image_root / "old-image-file").exists())
                self.assertEqual(os.environ["GOROOT"], str(install_dir))
                go.ensure_go_toolchain("1.26.10")
                fetch.assert_called_once()

    def test_failed_download_does_not_remove_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            install_dir = Path(temp_dir) / "xllm/toolchains/go1.26.10-linux-amd64"
            install_dir.mkdir(parents=True)
            marker = install_dir / "keep-on-failure"
            marker.touch()
            with (
                mock.patch.dict(os.environ, {"XDG_CACHE_HOME": temp_dir}),
                mock.patch.object(go.platform, "machine", return_value="x86_64"),
                mock.patch.object(go, "_probe_toolchain", return_value=None),
                mock.patch.object(go, "_download_archive", side_effect=RuntimeError("download failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    go.ensure_go_toolchain("1.26.10")
                self.assertTrue(marker.exists())
                self.assertEqual(
                    sorted(p.name for p in install_dir.parent.iterdir()),
                    [f".{install_dir.name}.lock", install_dir.name],
                )

    def test_download_checks_official_checksum_and_falls_back(self) -> None:
        content = b"verified archive"
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "go.tar.gz"
            with (
                mock.patch.object(go, "_get_archive_checksum", return_value=hashlib.sha256(content).hexdigest()),
                mock.patch.object(
                    go, "urlopen", side_effect=[io.BytesIO(b"wrong content"), io.BytesIO(content)]
                ) as download,
            ):
                go._download_archive("go1.26.10.linux-arm64.tar.gz", destination)
                self.assertEqual(destination.read_bytes(), content)
                self.assertEqual(download.call_count, 2)

    def test_unverified_archives_are_not_accepted(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(go, "_get_archive_checksum", return_value="0" * 64),
            mock.patch.object(go, "urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(b"wrong content")),
            self.assertRaisesRegex(RuntimeError, "verify"),
        ):
            go._download_archive("go1.26.10.linux-arm64.tar.gz", Path(temp_dir) / "go.tar.gz")

    def test_official_metadata_falls_back_after_invalid_response(self) -> None:
        filename = "go1.26.10.linux-arm64.tar.gz"
        metadata = [{"files": [{"filename": filename, "sha256": "a" * 64}]}]
        with mock.patch.object(
            go, "urlopen", side_effect=[io.BytesIO(b"<html>"), io.BytesIO(json.dumps(metadata).encode())]
        ):
            self.assertEqual(go._get_archive_checksum(filename), "a" * 64)

    def test_unsafe_archive_members_are_rejected_before_extraction(self) -> None:
        for name, member_type in (
            ("go/../../outside", tarfile.REGTYPE),
            ("/outside", tarfile.REGTYPE),
            ("go/link", tarfile.SYMTYPE),
            ("go/link", tarfile.LNKTYPE),
        ):
            with self.subTest(name=name, member_type=member_type), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                tarball = root / "unsafe.tar.gz"
                with tarfile.open(tarball, "w:gz") as archive:
                    member = tarfile.TarInfo(name)
                    member.type = member_type
                    member.linkname = "/outside"
                    archive.addfile(member)
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    go._extract_archive(tarball, root / "extract")
                self.assertFalse((root / "extract").exists())

    def test_invalid_pin_and_platform_fail_before_installation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid"):
            go.ensure_go_toolchain("../../go")
        with (
            mock.patch.object(go.platform, "machine", return_value="unsupported"),
            self.assertRaisesRegex(RuntimeError, "Unsupported"),
        ):
            go.ensure_go_toolchain("1.26.10")


if __name__ == "__main__":
    unittest.main()
