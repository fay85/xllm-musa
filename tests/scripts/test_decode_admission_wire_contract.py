# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU wire-contract regression for native Decode admission observations."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from google.protobuf import descriptor_pb2


class DecodeAdmissionWireContractTest(unittest.TestCase):
    def test_service_observation_fields_have_stable_tags_and_presence(self) -> None:
        root = Path(__file__).resolve().parents[2]
        protoc = os.environ.get("PROTOC") or shutil.which("protoc")
        if not protoc:
            protoc = str(
                root / "build/cmake.linux-x86_64-cpython-310/vcpkg_installed"
                / "x64-linux/tools/protobuf/protoc"
            )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pd.pb"
            subprocess.run(
                [
                    protoc,
                    "--proto_path=" + str(root / "xllm/proto"),
                    "--include_imports",
                    "--descriptor_set_out=" + str(output),
                    str(root / "xllm/proto/disagg_pd.proto"),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            descriptor = descriptor_pb2.FileDescriptorSet.FromString(output.read_bytes())
        generation = next(
            message
            for file in descriptor.file
            if file.name == "disagg_pd.proto"
            for message in file.message_type
            if message.name == "DisaggStreamGeneration"
        )
        fields = {field.name: field for field in generation.field}
        # These tags are consumed by the Service RPC adapter as an all-or-none
        # observation. Absence turns a successful native admission into UNKNOWN.
        expected = {
            "decode_admission_disposition": (
                12,
                descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
                ".xllm.proto.AdmissionDisposition",
            ),
            "decode_admission_reason": (
                13,
                descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
                ".xllm.proto.AdmissionReason",
            ),
            "decode_admission_attempts": (
                14,
                descriptor_pb2.FieldDescriptorProto.TYPE_UINT32,
                "",
            ),
            "decode_admission_rpc_duration_ns": (
                15,
                descriptor_pb2.FieldDescriptorProto.TYPE_UINT64,
                "",
            ),
        }
        for name, (tag, field_type, type_name) in expected.items():
            with self.subTest(field=name):
                self.assertTrue(
                    name in fields, f"Missing native admission evidence: {name}"
                )
                field = fields[name]
                self.assertEqual(field.number, tag)
                self.assertEqual(field.type, field_type)
                self.assertEqual(field.type_name, type_name)
                self.assertTrue(field.proto3_optional)
        self.assertEqual(fields["prefill_resources_released"].number, 16)


if __name__ == "__main__":
    unittest.main()
