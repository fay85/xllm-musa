# Copyright 2026 The xLLM Authors.
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

"""Interface contract for the ops-transformer operators xllm depends on.

The operators are compiled from ``third_party/xllm_ops/third_party/
ops-transformer`` into the ``custom_transformer`` vendor.  A gitlink bump can
change their def input/output/attr lists or their hand-written aclnn parameter
lists without any xllm compile error, because xllm resolves the aclnn entry
through ``dlsym`` and hand-writes the expected function pointer type.

This test parses the ops-transformer source at the pinned commit and compares
the interfaces with the contract xllm relies on.  When it fails after a gitlink
bump, do not only update this file: first diff the vendor interface, then
update the xllm wrapper listed next to the operator, and finally update the
expectation here.

Scope: only the operators xllm actually calls from the ``custom_transformer``
vendor: ``mega_moe`` and ``pool_key_indexer``.  The other operators produced by
``build_ops_transformer.sh`` have no xllm caller and are intentionally not
guarded here; add them when a model starts using them.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OPS_TRANSFORMER_ROOT = (
    _REPO_ROOT / "third_party" / "xllm_ops" / "third_party" / "ops-transformer"
)

_MEGA_MOE_INPUTS = [
    "context",
    "x",
    "topk_ids",
    "topk_weights",
    "weight1",
    "weight2",
    "weight_scales1",
    "weight_scales2",
    "bias1",
    "bias2",
    "x_active_mask",
    "scales",
    "shared_weight1",
    "shared_weight2",
    "shared_weight_scales1",
    "shared_weight_scales2",
    "shared_bias1",
    "shared_bias2",
    "mask_buffer",
]
_MEGA_MOE_OUTPUTS = ["y", "expert_token_nums"]
_MEGA_MOE_ATTRS = [
    "moe_expert_num",
    "ep_world_size",
    "ccl_buffer_size",
    "max_recv_token_num",
    "dispatch_quant_mode",
    "dispatch_quant_out_dtype",
    "shared_expert_quant_out_dtype",
    "combine_quant_mode",
    "comm_alg",
    "num_max_tokens_per_rank",
    "activation",
    "activation_params",
    "activation_out_dtype",
    "transpose_weight1",
    "transpose_weight2",
    "weight1_interleave",
    "topo_type",
    "rank_num_per_server",
    "topk_weights_type",
    "combine_comm_mode",
]

_POOL_KEY_INDEXER_INPUTS = [
    "query",
    "pool_key",
    "weights",
    "pool_tail_k",
    "actual_seq_q",
    "actual_seq_k",
    "block_table",
    "q_descale",
    "k_descale",
]
_POOL_KEY_INDEXER_OUTPUTS = ["sparse_indices", "sparse_values"]
_POOL_KEY_INDEXER_ATTRS = [
    "topk",
    "pool_size",
    "layout_q",
    "layout_k",
    "mask_mode",
    "quant_mode",
    "return_value",
    "key_stride0",
    "k_descale_stride0",
    "sparse_indices_mode",
]

# op name -> (def path, expected inputs, outputs, attrs, xllm wrapper to update)
_DEF_INTERFACES = {
    "mega_moe": (
        "mc2/mega_moe/op_host/mega_moe_def.cpp",
        _MEGA_MOE_INPUTS,
        _MEGA_MOE_OUTPUTS,
        _MEGA_MOE_ATTRS,
        "xllm/core/kernels/npu/npu_mega_moe.cpp",
    ),
    "pool_key_indexer": (
        "attention/pool_key_indexer/op_host/pool_key_indexer_def.cpp",
        _POOL_KEY_INDEXER_INPUTS,
        _POOL_KEY_INDEXER_OUTPUTS,
        _POOL_KEY_INDEXER_ATTRS,
        "xllm/core/kernels/npu/xllm_ops/pool_key_indexer.cpp",
    ),
}


def _expected_acl_tensor(count: int) -> list[str]:
    return ["const aclTensor*"] * count


def _expected_acl_tensor_list(count: int) -> list[str]:
    return ["const aclTensorList*"] * count


# The C ABI sequence of aclnnMegaMoeGetWorkspaceSize.  xllm hand-writes this
# exact pointer type in npu_mega_moe.cpp.
_MEGA_MOE_ACLNN_TYPES = (
    _expected_acl_tensor(4)
    + _expected_acl_tensor_list(6)
    + _expected_acl_tensor(2)
    + _expected_acl_tensor_list(6)
    + _expected_acl_tensor(1)
    + ["int64_t"] * 8
    + ["const char*", "int64_t", "const char*", "const aclFloatArray*"]
    + ["int64_t"] * 4
    + ["aclTensor*"] * 2
    + ["uint64_t*", "aclOpExecutor**"]
)

# aclnnPoolKeyIndexerTensorGetWorkspaceSize: V1 is frozen to the legacy
# signature, so it must not gain sparseIndicesMode.
_POOL_KEY_INDEXER_V1_ACLNN_TYPES = (
    _expected_acl_tensor(4)
    + _expected_acl_tensor(5)
    + ["int64_t"] * 2
    + ["char*"] * 2
    + ["int64_t"] * 2
    + ["bool"]
    + ["int64_t"] * 2
    + _expected_acl_tensor(2)
    + ["uint64_t*", "aclOpExecutor**"]
)

# aclnnPoolKeyIndexerV2TensorGetWorkspaceSize: V2 adds sparseIndicesMode before
# the output tensors.
_POOL_KEY_INDEXER_V2_ACLNN_TYPES = (
    _expected_acl_tensor(4)
    + _expected_acl_tensor(5)
    + ["int64_t"] * 2
    + ["char*"] * 2
    + ["int64_t"] * 2
    + ["bool"]
    + ["int64_t"] * 3
    + _expected_acl_tensor(2)
    + ["uint64_t*", "aclOpExecutor**"]
)

# (label, source path, function, expected types, required names, forbidden names)
_ACLNN_SIGNATURES = (
    (
        "aclnnMegaMoe",
        "mc2/mega_moe/op_api/aclnn_mega_moe.cpp",
        "aclnnMegaMoeGetWorkspaceSize",
        _MEGA_MOE_ACLNN_TYPES,
        (
            "sharedWeight1Optional",
            "sharedWeight2Optional",
            "sharedWeightScales1Optional",
            "sharedWeightScales2Optional",
            "sharedBias1Optional",
            "sharedBias2Optional",
            "maskBufferOptional",
            "activationParams",
            "topoType",
            "rankNumPerServer",
        ),
        ("activationClamp",),
    ),
    (
        "aclnnPoolKeyIndexer",
        "attention/pool_key_indexer/op_host/op_api/aclnn_pool_key_indexer.h",
        "aclnnPoolKeyIndexerTensorGetWorkspaceSize",
        _POOL_KEY_INDEXER_V1_ACLNN_TYPES,
        ("kDescaleStride0",),
        ("sparseIndicesMode",),
    ),
    (
        "aclnnPoolKeyIndexerV2",
        "attention/pool_key_indexer/op_host/op_api/aclnn_pool_key_indexer_v2.h",
        "aclnnPoolKeyIndexerV2TensorGetWorkspaceSize",
        _POOL_KEY_INDEXER_V2_ACLNN_TYPES,
        ("kDescaleStride0", "sparseIndicesMode"),
        (),
    ),
)


def _ops_transformer_root() -> Path:
    env_root = os.environ.get("XLLM_OPS_TRANSFORMER_ROOT")
    candidates = [Path(env_root)] if env_root else []
    candidates.append(_DEFAULT_OPS_TRANSFORMER_ROOT)
    for root in candidates:
        # Do not use a guarded file as the presence probe: if the vendor moves
        # that file, the whole contract would skip instead of failing.  The
        # top-level operator directories distinguish "source absent" from
        # "layout changed"; the latter fails in _read().
        if (root / "mc2").is_dir() or (root / "attention").is_dir():
            return root
    pytest.skip(
        "ops-transformer source not found; initialize third_party/xllm_ops "
        "(git submodule update --init --recursive) or set "
        "XLLM_OPS_TRANSFORMER_ROOT"
    )
    raise AssertionError("unreachable")


def _read(root: Path, relative_path: str) -> str:
    path = root / relative_path
    assert path.is_file(), (
        f"guarded ops-transformer file {relative_path} is missing under {root}; "
        "update this contract and the xllm wrapper if the vendor layout changed."
    )
    return path.read_text(encoding="utf-8")


def _def_names(source: str, kind: str) -> list[str]:
    return re.findall(rf'this->{kind}\("([^"]+)"\)', source)


def _aclnn_params(source: str, function: str) -> list[str]:
    # The ops-transformer aclnn declarations contain no nested parentheses,
    # function-pointer parameters or default values.  If that changes, the
    # per-parameter type comparison below will fail on the resulting
    # count/type mismatch, which is the intended signal to review the
    # interface and update this parser.
    match = re.search(rf"\b{function}\s*\((.*?)\)\s*(?:\{{|;)", source, re.DOTALL)
    assert match, f"{function} declaration not found"
    body = match.group(1)
    return [" ".join(param.split()) for param in body.split(",") if param.strip()]


def _param_type(declaration: str) -> str:
    # Drop the trailing parameter name, then normalize pointer spacing so a
    # formatting-only change such as "const aclTensor *x" does not fail the ABI
    # check.
    declaration = re.sub(r"\s*[A-Za-z_]\w*\s*$", "", declaration)
    declaration = re.sub(r"\s*\*\s*", "*", declaration)
    return declaration.strip()


def _assert_list(label: str, actual: list[str], expected: list[str], op: str, caller: str) -> None:
    assert actual == expected, (
        f"{op} {label} interface changed in ops-transformer.\n"
        f"expected ({len(expected)}): {expected}\n"
        f"actual   ({len(actual)}): {actual}\n"
        f"If the change is intentional, update {caller} and this expectation "
        "together."
    )


def test_operator_def_interfaces_are_unchanged() -> None:
    root = _ops_transformer_root()
    for op, (relative_path, inputs, outputs, attrs, caller) in _DEF_INTERFACES.items():
        source = _read(root, relative_path)
        _assert_list("input", _def_names(source, "Input"), inputs, op, caller)
        _assert_list("output", _def_names(source, "Output"), outputs, op, caller)
        _assert_list("attribute", _def_names(source, "Attr"), attrs, op, caller)


def test_aclnn_signatures_are_unchanged() -> None:
    root = _ops_transformer_root()
    for label, relative_path, function, expected_types, required, forbidden in _ACLNN_SIGNATURES:
        declarations = _aclnn_params(_read(root, relative_path), function)
        types = [_param_type(declaration) for declaration in declarations]
        assert types == expected_types, (
            f"{label} aclnn signature changed in ops-transformer.\n"
            f"expected ({len(expected_types)}): {expected_types}\n"
            f"actual   ({len(types)}): {types}\n"
            "If the change is intentional, update the xllm call site and this "
            "expectation together."
        )
        for name in required:
            assert any(name in declaration for declaration in declarations), (
                f"{label} lost the required parameter '{name}'. "
                "Check the vendor interface and the xllm call site."
            )
        for name in forbidden:
            assert not any(name in declaration for declaration in declarations), (
                f"{label} unexpectedly contains '{name}'; this changes the ABI "
                "that xllm's frozen pointer type relies on."
            )


def test_probe_detects_layout_change_without_guarded_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mc2").mkdir()
    monkeypatch.setenv("XLLM_OPS_TRANSFORMER_ROOT", str(tmp_path))
    assert _ops_transformer_root() == tmp_path
    with pytest.raises(AssertionError, match="guarded ops-transformer file"):
        _read(tmp_path, "mc2/mega_moe/op_host/mega_moe_def.cpp")


def test_direct_run_converts_skip_to_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def skip() -> None:
        pytest.skip("missing source")

    monkeypatch.setattr(sys.modules[__name__], "_ops_transformer_root", skip)
    assert main() == 0
    assert "skipped" in capsys.readouterr().out


def main() -> int:
    try:
        test_operator_def_interfaces_are_unchanged()
        test_aclnn_signatures_are_unchanged()
    except pytest.skip.Exception as skipped:
        print(f"ops-transformer interface contract: skipped ({skipped})")
        return 0
    print("ops-transformer interface contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
