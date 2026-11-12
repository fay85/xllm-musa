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

"""xllm release pack assembled at the end of bdist_wheel."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

from scripts.build_support import release_pack


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        output.write(content)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        digest.update(source.read())
    return digest.hexdigest()


class ReleasePackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pack = release_pack

    def test_release_pack_keeps_xllm_ops_and_mooncake_only(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base_dir = os.path.join(root, "xllm")
            extdir = os.path.join(root, "ext")
            cmake_dir = os.path.join(root, "cmake")
            dist_dir = os.path.join(root, "dist")
            vendor_root = os.path.join(root, "ascend", "opp", "vendors")
            os.makedirs(extdir)
            os.makedirs(os.path.join(cmake_dir, "mooncake-common"))
            os.makedirs(dist_dir)
            for name in ("custom_transformer", "custom_xllm_math", "glm_next_transformer", "kpool_transformer"):
                os.makedirs(os.path.join(vendor_root, name, "op_impl"))
                _write(os.path.join(vendor_root, name, "op_impl", "kernel.json"), "{}\n")
            os.makedirs(os.path.join(vendor_root, "custom_xllm_math", "op_api", "include", "aclnnop"))
            _write(
                os.path.join(vendor_root, "config.ini"),
                "load_priority=glm_next_transformer,custom_transformer,custom_xllm_math\n",
            )

            mooncake_pyproject = os.path.join(base_dir, "third_party", "Mooncake", "mooncake-wheel", "pyproject.toml")
            _write(mooncake_pyproject, 'name = "mooncake-transfer-engine"\nversion = "0.3.12"\n')
            flash_wheel = os.path.join(
                base_dir,
                "third_party",
                "xllm_ops",
                "third_party",
                "flash-linear-attention-npu",
                "dist",
                "flash_linear_attention_npu_a3-26.9.1-py3-none-manylinux_2_34_aarch64.whl",
            )
            _write(flash_wheel, "flash-wheel")
            stale_flash_wheel = os.path.join(
                os.path.dirname(flash_wheel),
                "flash_linear_attention_npu_a3-26.8.0-py3-none-manylinux_2_34_aarch64.whl",
            )
            _write(stale_flash_wheel, "stale-flash-wheel")
            os.utime(stale_flash_wheel, (1_700_000_000, 1_700_000_000))
            xllm_wheel = os.path.join(dist_dir, "xllm_npu_torch2.9.0-0.11.0-cp311-cp311-linux_aarch64.whl")
            _write(xllm_wheel, "xllm-wheel")

            for name, content in {
                "mooncake_master": "master",
                "mooncake_client": "client",
                "libetcd_wrapper.so": "etcd-wrapper",
                "liburing.so.2": "uring",
            }.items():
                _write(os.path.join(extdir, name), content)
            _write(os.path.join(cmake_dir, "mooncake-common", "libasio.so"), "asio")
            _write(os.path.join(root, "libxxhash.so.0"), "xxhash")
            _write(os.path.join(root, "libzstd.so.1"), "zstd")
            stale_pack = os.path.join(base_dir, "xllm_pack")
            os.makedirs(stale_pack)
            _write(os.path.join(stale_pack, "stale.txt"), "stale")
            _write(stale_pack + ".tar.gz", "old-archive")
            _write(stale_pack + ".tar.gz.sha256", "old-checksum")

            with (
                mock.patch.object(
                    self.pack,
                    "_git_short_hash",
                    side_effect=lambda path: "ops1234" if path.endswith("xllm_ops") else "xllm1234",
                ),
                mock.patch.object(self.pack, "_cann_tag", return_value="cann9.0"),
                mock.patch.object(self.pack, "get_cpu_arch", return_value="arm"),
                mock.patch.object(self.pack, "_chip_tag", return_value="a3"),
                mock.patch.object(
                    self.pack, "_resolve_system_soname", side_effect=lambda soname: os.path.join(root, soname)
                ),
                mock.patch.dict(
                    os.environ,
                    {
                        "ASCEND_HOME_PATH": os.path.join(root, "ascend"),
                        "ASCEND_OPP_PATH": os.path.join(root, "ascend", "opp"),
                    },
                    clear=False,
                ),
                mock.patch.object(self.pack, "_write_pack_version_file", return_value="xllm_version.txt"),
            ):
                pack_dir = self.pack.collect_xllm_release_pack(base_dir, extdir, cmake_dir, "npu", dist_dir, xllm_wheel)

            names = set(os.listdir(pack_dir))
            self.assertEqual(
                names,
                {
                    os.path.basename(xllm_wheel),
                    os.path.basename(flash_wheel),
                    "vendors-ops1234-cann9.0-arm-a3.tar.gz",
                    "vendors-ops1234-cann9.0-arm-a3.tar.gz.sha256",
                    "mooncake-runtime-v0.3.12.tar.gz",
                    "mooncake-runtime-v0.3.12.tar.gz.sha256",
                },
            )
            self.assertFalse(any(name in {"etcd", "etcdctl", "stale.txt"} for name in names))
            self._assert_sha256_sidecar(pack_dir, "mooncake-runtime-v0.3.12.tar.gz")
            self._assert_sha256_sidecar(pack_dir, "vendors-ops1234-cann9.0-arm-a3.tar.gz")
            self._assert_sha256_sidecar(os.path.dirname(pack_dir), "xllm_pack.tar.gz")

            archive_path = os.path.join(pack_dir, "mooncake-runtime-v0.3.12.tar.gz")
            with tarfile.open(archive_path, "r:gz") as archive:
                members = set(archive.getnames())
            self.assertIn("mooncake-v0.3.12/bin/mooncake_master", members)
            self.assertIn("mooncake-v0.3.12/bin/mooncake_client", members)
            self.assertIn("mooncake-v0.3.12/lib/libasio.so", members)
            self.assertIn("mooncake-v0.3.12/lib/libxxhash.so.0", members)
            self.assertIn("mooncake-v0.3.12/check.sh", members)
            self.assertNotIn("mooncake-v0.3.12/bin/etcd", members)

            with tarfile.open(os.path.join(pack_dir, "vendors-ops1234-cann9.0-arm-a3.tar.gz"), "r:gz") as archive:
                vendor_members = set(archive.getnames())
            self.assertIn("vendors/config.ini", vendor_members)
            self.assertIn("vendors/custom_transformer/op_impl/kernel.json", vendor_members)
            self.assertIn("vendors/custom_xllm_math/op_impl/kernel.json", vendor_members)
            self.assertIn("vendors/glm_next_transformer/op_impl/kernel.json", vendor_members)
            self.assertIn("vendors/kpool_transformer/op_impl/kernel.json", vendor_members)

            pack_archive = pack_dir + ".tar.gz"
            self.assertTrue(os.path.isfile(pack_archive))
            with tarfile.open(pack_archive, "r:gz") as archive:
                packed_names = set(archive.getnames())
            self.assertIn("xllm_pack/" + os.path.basename(xllm_wheel), packed_names)
            self.assertIn("xllm_pack/vendors-ops1234-cann9.0-arm-a3.tar.gz", packed_names)
            self.assertIn("xllm_pack/vendors-ops1234-cann9.0-arm-a3.tar.gz.sha256", packed_names)
            self.assertIn("xllm_pack/mooncake-runtime-v0.3.12.tar.gz", packed_names)
            self.assertIn("xllm_pack/mooncake-runtime-v0.3.12.tar.gz.sha256", packed_names)
            self.assertNotIn("xllm_pack/xllm_pack.tar.gz.sha256", packed_names)

    def test_release_strip_removes_debug_sections_from_shipped_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as package_dir:
            sources = {
                "xllm": "server-binary-debug-info",
                "spawn_worker": "worker-binary-debug-info",
                "xllm_export.cpython-311-aarch64-linux-gnu.so": "export-debug-info",
                "mooncake_master": "master-debug-info",
                "mooncake_client": "client-debug-info",
                "libetcd_wrapper.so": "keep-this-shared-library",
            }
            for name, content in sources.items():
                _write(os.path.join(package_dir, name), content)
                os.chmod(os.path.join(package_dir, name), 0o755)

            def fake_strip(command: list[str]) -> None:
                self.assertEqual(command[:2], ["strip", "--strip-debug"])
                _write(command[2], "stripped")

            with mock.patch.object(self.pack.subprocess, "check_call", side_effect=fake_strip) as strip_call:
                self.pack.strip_release_debug_symbols(package_dir)

            stripped = {os.path.basename(call.args[0][2]) for call in strip_call.call_args_list}
            self.assertEqual(
                stripped,
                {
                    "xllm",
                    "spawn_worker",
                    "xllm_export.cpython-311-aarch64-linux-gnu.so",
                    "mooncake_master",
                    "mooncake_client",
                },
            )
            self.assertNotIn("libetcd_wrapper.so", stripped)
            with open(os.path.join(package_dir, "libetcd_wrapper.so"), encoding="utf-8") as kept:
                self.assertEqual(kept.read(), "keep-this-shared-library")
            with open(os.path.join(package_dir, "xllm"), encoding="utf-8") as stripped_binary:
                self.assertEqual(stripped_binary.read(), "stripped")

    def test_release_strip_refuses_to_pack_when_a_binary_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as package_dir:
            _write(os.path.join(package_dir, "xllm"), "server")
            with self.assertRaises(RuntimeError):
                self.pack.strip_release_debug_symbols(package_dir)

    def test_version_file_records_xllm_branch_and_recent_commits(self) -> None:
        xllm_log = "\n".join(
            [
                "commit aaaaaaa",
                "Author: xllm <xllm@example.com>",
                "Date:   Tue Nov 10 12:00:00 2026 +0800",
                "",
                "    xllm commit",
            ]
        )
        ops_log = "commit bbbbbbb\n\n    xllm_ops commit"
        mooncake_log = "commit ccccccc\n\n    mooncake commit"

        def recent_log(repo_dir: str, count: int = 5) -> str:
            self.assertEqual(count, 5)
            if repo_dir.endswith("xllm_ops"):
                return ops_log
            if repo_dir.endswith("Mooncake"):
                return mooncake_log
            return xllm_log

        with tempfile.TemporaryDirectory() as root:
            with (
                mock.patch.object(self.pack, "_git_branch", return_value="feature/pack"),
                mock.patch.object(self.pack, "_git_recent_log", side_effect=recent_log),
            ):
                flash_wheel = os.path.join(root, "flash_linear_attention_npu_a3-26.9.1.whl")
                _write(flash_wheel, "flash-wheel")
                flash_digest = _sha256(flash_wheel)
                path = self.pack._write_pack_version_file(root, root, flash_wheel)

            with open(path, encoding="utf-8") as version_file:
                content = version_file.read()

        self.assertIn("===== xllm =====\nbranch: feature/pack\n", content)
        self.assertIn(
            "===== flash-linear-attention-npu =====\nwheel: flash_linear_attention_npu_a3-26.9.1.whl\n", content
        )
        self.assertIn(f"sha256: {flash_digest}", content)
        self.assertIn("    xllm commit", content)
        self.assertIn("===== xllm_ops =====\ncommit bbbbbbb", content)
        self.assertNotIn("xllm_ops =====\nbranch:", content)
        self.assertIn("===== mooncake =====\ncommit ccccccc", content)
        self.assertNotIn("mooncake =====\nbranch:", content)
        self.assertNotIn("xllm-service", content)
        self.assertTrue(path.endswith(os.path.join(root, "xllm_version.txt")))
        self.assertEqual(content.count("====="), 8)

    def test_vendors_archive_packs_the_whole_opp_vendors_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            opp_root = os.path.join(root, "opp")
            vendor_root = os.path.join(opp_root, "vendors")
            for name in ("custom_transformer", "custom_xllm_math", "glm_next_transformer", "kpool_transformer"):
                os.makedirs(os.path.join(vendor_root, name, "op_impl"))
                _write(os.path.join(vendor_root, name, "op_impl", "kernel.json"), "{}\n")
            _write(
                os.path.join(vendor_root, "config.ini"),
                "load_priority=glm_next_transformer,custom_transformer,custom_xllm_math\n",
            )
            output_dir = os.path.join(root, "out")
            os.makedirs(output_dir)
            with (
                mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": opp_root}, clear=False),
                mock.patch.object(self.pack, "_git_short_hash", return_value="ops1234"),
                mock.patch.object(self.pack, "_cann_tag", return_value="cann9.0.0"),
                mock.patch.object(self.pack, "get_cpu_arch", return_value="arm"),
                mock.patch.object(self.pack, "_chip_tag", return_value="a3"),
            ):
                archive = self.pack._stage_vendors_archive(root, output_dir, "npu")
            with tarfile.open(archive, "r:gz") as packed:
                names = set(packed.getnames())
            self.assertIn("vendors/config.ini", names)
            self.assertIn("vendors/custom_transformer/op_impl/kernel.json", names)
            self.assertIn("vendors/custom_xllm_math/op_impl/kernel.json", names)
            self.assertIn("vendors/glm_next_transformer/op_impl/kernel.json", names)
            self.assertIn("vendors/kpool_transformer/op_impl/kernel.json", names)
            self.assertEqual(os.path.basename(archive), "vendors-ops1234-cann9.0.0-arm-a3.tar.gz")

            os.remove(os.path.join(vendor_root, "config.ini"))
            with (
                mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": opp_root}, clear=False),
                self.assertRaisesRegex(RuntimeError, "config.ini"),
            ):
                self.pack._stage_vendors_archive(root, output_dir, "npu")

            shutil.rmtree(os.path.join(vendor_root, "custom_transformer"))
            _write(os.path.join(vendor_root, "config.ini"), "load_priority=custom_xllm_math\n")
            with (
                mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": opp_root}, clear=False),
                self.assertRaisesRegex(RuntimeError, "custom_transformer"),
            ):
                self.pack._stage_vendors_archive(root, output_dir, "npu")

    def _write_smla_cpu_vendor(self, root: str) -> tuple[str, str]:
        """Create distinct CPU implementations to catch cross-vendor aliases."""
        vendor = os.path.join(root, "custom_xllm_math", "op_impl", "cpu")
        library = os.path.join(vendor, "aicpu_kernel", "impl", "libtransformer_aicpu_kernels.so")
        _write(library, "math CPU implementation")
        config = {"SmlaVaMetadata": {"opInfo": {"kernelSo": "libtransformer_aicpu_kernels.so"}}}
        _write(os.path.join(vendor, "config", "cust_aicpu_kernel.json"), json.dumps(config))
        transformer = os.path.join(root, "custom_transformer", "op_impl", "cpu", "aicpu_kernel", "impl")
        _write(os.path.join(transformer, "libtransformer_aicpu_kernels.so"), "different transformer implementation")
        return library, os.path.join(os.path.dirname(library), "libmath_aicpu_kernels.so")

    def test_cann90_aicpu_alias_uses_only_matching_math_library(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            library, alias = self._write_smla_cpu_vendor(root)
            before = _sha256(library)
            self.pack._stage_cann90_aicpu_alias(root, "cann9.0.0")
            self.assertTrue(os.path.islink(alias))
            self.assertEqual(os.readlink(alias), "libtransformer_aicpu_kernels.so")
            self.assertEqual(_sha256(alias), before)
            self.assertEqual(_sha256(library), before)
            self.pack._stage_cann90_aicpu_alias(root, "cann9.0.0")
            self.assertEqual(_sha256(alias), before)

    def test_cann90_aicpu_alias_preserves_existing_library(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            _, alias = self._write_smla_cpu_vendor(root)
            _write(alias, "newer canonical math CPU implementation")
            before = _sha256(alias)
            self.pack._stage_cann90_aicpu_alias(root, "cann9.0.0")
            self.assertFalse(os.path.islink(alias))
            self.assertEqual(_sha256(alias), before)

    def test_cann90_aicpu_alias_rejects_missing_source_and_broken_alias(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            library, alias = self._write_smla_cpu_vendor(root)
            os.remove(library)
            with self.assertRaisesRegex(RuntimeError, "missing from its own vendor"):
                self.pack._stage_cann90_aicpu_alias(root, "cann9.0.0")
            _write(library, "math CPU implementation")
            os.symlink("missing.so", alias)
            with self.assertRaisesRegex(RuntimeError, "not a readable file"):
                self.pack._stage_cann90_aicpu_alias(root, "cann9.0.0")

    def test_cann90_aicpu_alias_does_not_modify_other_cann_versions(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            _, alias = self._write_smla_cpu_vendor(root)
            self.pack._stage_cann90_aicpu_alias(root, "cann9.1.0")
            self.assertFalse(os.path.lexists(alias))

    def test_vendors_archive_keeps_aicpu_alias_without_mutating_installed_tree(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            opp_root = os.path.join(root, "opp")
            vendor_root = os.path.join(opp_root, "vendors")
            library, alias = self._write_smla_cpu_vendor(vendor_root)
            _write(os.path.join(vendor_root, "config.ini"), "load_priority=custom_xllm_math,custom_transformer\n")
            output_dir = os.path.join(root, "out")
            os.makedirs(output_dir)
            original_hash = _sha256(library)
            with (
                mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": opp_root}, clear=False),
                mock.patch.object(self.pack, "_git_short_hash", return_value="ops1234"),
                mock.patch.object(self.pack, "_cann_tag", return_value="cann9.0.0"),
                mock.patch.object(self.pack, "get_cpu_arch", return_value="arm"),
                mock.patch.object(self.pack, "_chip_tag", return_value="a3"),
            ):
                archive = self.pack._stage_vendors_archive(root, output_dir, "npu")
            self.assertFalse(os.path.lexists(alias))
            self.assertEqual(_sha256(library), original_hash)
            relative = "vendors/custom_xllm_math/op_impl/cpu/aicpu_kernel/impl/"
            with tarfile.open(archive, "r:gz") as packed:
                member = packed.getmember(relative + "libmath_aicpu_kernels.so")
                self.assertTrue(member.issym())
                self.assertEqual(member.linkname, "libtransformer_aicpu_kernels.so")
                self.assertEqual(packed.extractfile(member).read(), b"math CPU implementation")
            self._assert_sha256_sidecar(output_dir, os.path.basename(archive))

    def test_cann_tag_uses_full_version_info(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            toolkit = os.path.join(root, "ascend-toolkit", "8.3.RC1")
            version_info = os.path.join(toolkit, "compiler", "version.info")
            _write(version_info, "Version=8.3.RC1\nTimestamp=20260101\n")
            with mock.patch.dict(os.environ, {"ASCEND_HOME_PATH": toolkit}, clear=False):
                self.assertEqual(self.pack._cann_tag(), "cann8.3.RC1")
            _write(version_info, "Version=9.0.0\n")
            with mock.patch.dict(os.environ, {"ASCEND_HOME_PATH": toolkit}, clear=False):
                self.assertEqual(self.pack._cann_tag(), "cann9.0.0")

            empty_toolkit = os.path.join(root, "ascend-toolkit", "latest")
            os.makedirs(empty_toolkit)
            with (
                mock.patch.dict(os.environ, {"ASCEND_HOME_PATH": empty_toolkit}, clear=False),
                self.assertRaisesRegex(RuntimeError, "version.info"),
            ):
                self.pack._cann_tag()

    def test_flash_wheel_selects_newest_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            older_mtime = os.path.join(root, "flash_linear_attention_npu_a3-9.whl")
            newer_mtime = os.path.join(root, "flash_linear_attention_npu_a3-1.whl")
            _write(older_mtime, "old")
            _write(newer_mtime, "new")
            os.utime(older_mtime, (2_000, 2_000))
            os.utime(newer_mtime, (1_000, 1_000))

            selected = self.pack._select_newest_file(
                os.path.join(root, "flash_linear_attention_npu_*.whl"),
                "flash-linear-attention-npu wheel",
            )

            self.assertEqual(selected, older_mtime)
            with self.assertRaisesRegex(RuntimeError, "No flash-linear-attention-npu wheel"):
                self.pack._select_newest_file(
                    os.path.join(root, "missing", "flash_linear_attention_npu_*.whl"),
                    "flash-linear-attention-npu wheel",
                )

    def test_missing_git_metadata_does_not_fail_the_pack(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(self.pack.logger, "exception") as log_exception:
                self.assertEqual(self.pack._git_short_hash(root), "unknown")
                self.assertEqual(self.pack._git_branch(root), "unknown")
                self.assertEqual(self.pack._git_recent_log(root), "unknown")
                path = self.pack._write_pack_version_file(root, root)
            self.assertGreaterEqual(log_exception.call_count, 1)

            with open(path, encoding="utf-8") as version_file:
                content = version_file.read()
        self.assertIn("===== xllm =====\nbranch: unknown\n", content)
        self.assertIn("===== xllm_ops =====\nunknown", content)
        self.assertIn("===== mooncake =====\nunknown", content)

    def test_detached_head_without_tag_stays_head(self) -> None:
        def git_command(repo_dir: str, *args: str) -> str:
            del repo_dir
            if args[:2] == ("rev-parse", "--abbrev-ref"):
                return "HEAD"
            raise subprocess.CalledProcessError(128, ["git", *args])

        with mock.patch.object(self.pack, "_git_command", side_effect=git_command):
            self.assertEqual(self.pack._git_branch("/tmp/detached"), "HEAD")

    def _assert_sha256_sidecar(self, directory: str, archive_name: str) -> None:
        archive_path = os.path.join(directory, archive_name)
        with open(archive_path + ".sha256", encoding="utf-8") as checksum:
            recorded, recorded_name = checksum.read().split()
        self.assertEqual(recorded, _sha256(archive_path))
        self.assertEqual(recorded_name, archive_name)

    def test_failed_release_pack_removes_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base_dir = os.path.join(root, "xllm")
            pack_dir = os.path.join(base_dir, "xllm_pack")
            os.makedirs(pack_dir)
            _write(os.path.join(pack_dir, "stale.txt"), "stale")
            _write(pack_dir + ".tar.gz", "old-archive")
            _write(pack_dir + ".tar.gz.sha256", "old-checksum")

            with self.assertRaises(FileNotFoundError):
                self.pack.collect_xllm_release_pack(
                    base_dir,
                    "/tmp",
                    "/tmp",
                    "npu",
                    "/tmp",
                    os.path.join(root, "missing.whl"),
                )

            self.assertFalse(os.path.exists(pack_dir))
            self.assertFalse(os.path.exists(pack_dir + ".tar.gz"))
            self.assertFalse(os.path.exists(pack_dir + ".tar.gz.sha256"))

    def test_mooncake_check_script_rejects_failed_ldd(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self.pack._write_mooncake_runtime_scripts(root)
            for name in ("mooncake_master", "mooncake_client"):
                binary = os.path.join(root, "bin", name)
                _write(binary, "not an elf\n")
                os.chmod(binary, os.stat(binary).st_mode | stat.S_IXUSR)

            failed_ldd = self._run_mooncake_check(root, "echo 'not a dynamic executable' >&2\nexit 1\n")
            self.assertNotEqual(failed_ldd.returncode, 0)
            self.assertNotIn("Mooncake runtime check passed", failed_ldd.stdout)
            self.assertIn("ldd could not check", failed_ldd.stderr)

            missing_library = self._run_mooncake_check(root, "echo 'libzstd.so.1 => not found'\nexit 0\n")
            self.assertNotEqual(missing_library.returncode, 0)
            self.assertNotIn("Mooncake runtime check passed", missing_library.stdout)
            self.assertIn("unresolved libraries", missing_library.stderr)
            self.assertIn("not found", missing_library.stderr)

            passed = self._run_mooncake_check(root, "echo 'libzstd.so.1 => /lib/libzstd.so.1'\nexit 0\n")
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn("Mooncake runtime check passed", passed.stdout)

            os.makedirs(os.path.join(root, "empty-path"))
            hidden_ldd = subprocess.run(
                ["/bin/bash", os.path.join(root, "check.sh")],
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": os.path.join(root, "empty-path")},
            )
            self.assertNotEqual(hidden_ldd.returncode, 0)
            self.assertNotIn("Mooncake runtime check passed", hidden_ldd.stdout)

    def _run_mooncake_check(self, root: str, ldd_body: str) -> subprocess.CompletedProcess:
        ldd_dir = os.path.join(root, "stub-bin")
        ldd_path = os.path.join(ldd_dir, "ldd")
        _write(ldd_path, "#!/bin/sh\n" + ldd_body)
        os.chmod(ldd_path, 0o755)
        return subprocess.run(
            ["bash", os.path.join(root, "check.sh")],
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": ldd_dir + os.pathsep + os.environ.get("PATH", "")},
        )

    def test_chip_detection_failure_keeps_traceback_and_archive_name(self) -> None:
        with (
            mock.patch.object(self.pack, "get_ascend_platform", side_effect=RuntimeError("acl init failed")),
            mock.patch.object(self.pack.logger, "exception") as log_exception,
            self.assertRaisesRegex(RuntimeError, "acl init failed"),
        ):
            self.pack._chip_tag("npu")
        log_exception.assert_called_once()
        self.assertIn("Ascend chip", log_exception.call_args.args[0])
        self.assertNotIn("a3", log_exception.call_args.args[0])

        with mock.patch.object(self.pack, "get_ascend_platform", return_value="a5"):
            self.assertEqual(self.pack._chip_tag("npu"), "a5")
        with (
            mock.patch.object(self.pack, "get_ascend_platform", return_value="a9"),
            self.assertRaisesRegex(RuntimeError, "a9"),
        ):
            self.pack._chip_tag("npu")
        self.assertEqual(self.pack._chip_tag("mlu"), "mlu")

    def test_release_pack_skips_non_npu_builds(self) -> None:
        for device in ("mlu", "cuda"):
            with self.subTest(device=device):
                self.assertEqual(
                    self.pack.collect_xllm_release_pack("/tmp", "/tmp", "/tmp", device, "/tmp", "/tmp/x.whl"),
                    "",
                )


if __name__ == "__main__":
    unittest.main()
