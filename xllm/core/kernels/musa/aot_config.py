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

"""Compile-time operator configurations and native ABI definitions."""

from __future__ import annotations

from typing import Any

PROFILE = "bf16_attention_h256_gqa6_8_gdn_h128_hq16_hv32_48_fp8_bmm_moe_decode_aot"
SCHEMA_VERSION = 6
PINNED_VERSIONS = {
    "mate": "0.2.7",
    "mate-mubin": "0.2.7",
    "apache-tvm-ffi": "0.1.11.post1+musa.1",
    "tilelang-musa": "0.1.12+musa.2",
    "torch": "2.9.1",
    "torch_musa": "2.9.1",
    "triton": "3.2.0",
}

_TENSOR = {"type": "DLTensor*"}
_OPTIONAL_TENSOR = {"type": "Optional", "args": [_TENSOR]}
_INT = {"type": "int"}
_OPTIONAL_INT = {"type": "Optional", "args": [_INT]}
_FLOAT = {"type": "float"}
_BOOL = {"type": "bool"}
_NONE = {"type": "None"}
# Match the actual TVM-FFI types, including return values and Optional arguments.
FFI_SIGNATURES = {
    "forward": {
        "type": "ffi.Function",
        "args": [
            {"type": "ffi.Array", "args": [{"type": "Any"}]},
            *[_TENSOR] * 3,
            *[_OPTIONAL_TENSOR] * 8,
            *[_OPTIONAL_INT] * 2,
            *[_OPTIONAL_TENSOR] * 9,
            _FLOAT,
            _BOOL,
            _INT,
            _INT,
            _INT,
            _FLOAT,
            _INT,
            _INT,
            *[_OPTIONAL_TENSOR] * 2,
            *[_TENSOR] * 2,
            *[_INT] * 2,
            _OPTIONAL_TENSOR,
            _BOOL,
        ],
    },
    "metadata": {
        "type": "ffi.Function",
        "args": [
            _NONE,
            *[_INT] * 8,
            *[_OPTIONAL_TENSOR] * 5,
            _INT,
            _INT,
            _OPTIONAL_TENSOR,
            _TENSOR,
            *[_INT] * 4,
        ],
    },
    "combine": {
        "type": "ffi.Function",
        "args": [
            _NONE,
            _OPTIONAL_TENSOR,
            _OPTIONAL_TENSOR,
            _OPTIONAL_INT,
            *[_TENSOR] * 4,
            _OPTIONAL_TENSOR,
            _INT,
        ],
    },
    "gdn_prefill": {"type": "ffi.Function", "args": [_NONE, *[_TENSOR] * 10]},
    "gdn_kkt_fixed": {"type": "ffi.Function", "args": [_NONE, *[_TENSOR] * 3, _INT]},
    "gdn_kkt_varlen": {"type": "ffi.Function", "args": [_NONE, *[_TENSOR] * 4]},
    "bmm": {
        "type": "ffi.Function",
        "args": [
            _NONE,
            *[_TENSOR] * 3,
            *[{"type": "std::optional", "args": [_TENSOR]}] * 3,
            *[{"type": "std::tuple", "args": [_INT, _INT]}] * 2,
            *[_BOOL] * 3,
        ],
    },
    "moe_gemm": {
        "type": "ffi.Function",
        "args": [
            _NONE,
            {"type": "std::string"},
            *[_TENSOR] * 3,
            *[{"type": "std::optional", "args": [_TENSOR]}] * 6,
            {"type": "std::optional", "args": [_INT]},
        ],
    },
}


def _forward_config(ratio: int, mode: str) -> dict[str, Any]:
    """Describe only the signatures called by xLLM, not MATE's entire AOT sweep."""
    paged = mode != "dense"
    return {
        **dict.fromkeys(
            (
                "has_seqused_q",
                "has_kv_batch_idx",
                "has_leftpad_k",
                "has_q_descale",
                "has_k_descale",
                "has_v_descale",
                "has_softcap",
                "is_append_kv",
                "has_learnable_sink",
                "is_local",
                "enable_cp",
                "is_rotary",
                "is_rotary_interleaved",
                "has_seqlens_rotary",
                "has_attention_chunk",
                "has_qv",
                "only_qv",
            ),
            False,
        ),
        "has_cu_seqlens_q": True,
        "has_cu_seqlens_k": not paged,
        # xLLM writes projected K/V into its paged cache before attention;
        # this is not MATE's append-KV API.
        "has_cu_seqlens_k_new": False,
        "has_seqused_k": paged,
        "paged_kv": paged,
        "is_causal": True,
        "is_packgqa": paged,
        "head_ratio": ratio,
        "element": "mutlass::bfloat16_t",
        "use_lsu_load_k": not paged,
        "use_lsu_load_v": False,
        "tile_m": 32 if paged else 192,
        "tile_n": 64,
        "stages_k": 1,
        "stages_v": 1,
        "head_dim_qk": 256,
        "head_dim_vo": 256,
        "consumers_qk": 1 if paged else 3,
        "consumers_pv": 1 if paged else 3,
        "tile_head_dim": 0,
        "is_even_headdim": True,
        "has_metadata": paged,
    }


def _operator_recipes() -> list[dict[str, Any]]:
    recipes = []
    for ratio in (6, 8):
        for mode, label in (("dense", "PrefillFwd"), ("decode", "Fwd")):
            recipes.append(
                {
                    "name": f"xllm_mate_fmha_{mode}_bf16_h256_gqa{ratio}_v1",
                    "constant": f"kFa3{label}UriGqa{ratio}",
                    "kind": "forward",
                    "config": _forward_config(ratio, mode),
                    "arity": 38,
                }
            )
        recipes.append(
            {
                "name": f"xllm_mate_fmha_metadata_decode_bf16_gqa{ratio}_v1",
                "constant": f"kFa3MetadataUriGqa{ratio}",
                "kind": "metadata",
                "config": {
                    "has_cu_seqlens_q": True,
                    "has_cu_seqlens_k": False,
                    "has_seqused_q": False,
                    "has_seqused_k": True,
                    "has_cu_seqlens_k_new": False,
                    "has_leftpad_k": False,
                    "is_causal": True,
                    "is_local": False,
                    "is_packgqa": True,
                    "head_ratio": ratio,
                    "element": "mutlass::bfloat16_t",
                    "element_size": 2,
                    "sort": True,
                    "num_warps": 1,
                },
                "arity": 21,
            }
        )
    for splits in (16, 32, 64, 128):
        recipes.append(
            {
                "name": f"xllm_mate_fmha_combine_bf16_splits{splits}_v1",
                "constant": f"kFa3FwdCombine{splits}Uri",
                "kind": "combine",
                "config": {
                    "element": "mutlass::bfloat16_t",
                    "has_cu_seqlens_q": True,
                    "has_seqused_q": False,
                    "has_metadata": True,
                    "tile_m": 16,
                    "tile_n": 64,
                    "max_splits": splits,
                },
                "arity": 9,
            }
        )
    for value_heads in (32, 48):
        for is_varlen, label in ((False, "Fixed"), (True, "Varlen")):
            suffix = "varlen_strided" if is_varlen else "strided"
            common = {"H": value_heads, "Hg": 16, "DK": 128, "chunk_size": 64, "is_varlen": is_varlen}
            recipes.extend(
                [
                    {
                        "name": f"mate_kkt_solve_{suffix}_hq16_hv{value_heads}_bf16",
                        "constant": f"kGdnKkt{label}UriHv{value_heads}",
                        "function": "main",
                        "kind": "gdn_kkt_varlen" if is_varlen else "gdn_kkt_fixed",
                        "config": common,
                        "arity": 4,
                    },
                    {
                        "name": f"mate_gdn_prefill_full_{suffix}_hq16_hv{value_heads}_bf16",
                        "constant": f"kGdnPrefill{label}UriHv{value_heads}",
                        "function": "main",
                        "kind": "gdn_prefill",
                        "config": {
                            **common,
                            "DV": 128,
                            "scale": 128**-0.5,
                            "use_initial_state": True,
                            "store_final_state": True,
                            "is_log_space": True,
                        },
                        "arity": 10,
                    },
                ]
            )
    recipes.append(
        {
            "name": "gemm_ops",
            "constant": "kBmmUri",
            "function": "bmm",
            "kind": "bmm",
            "config": {},
            "arity": 11,
        }
    )
    for dtype in ("bf16", "fp16", "fp8_bf16", "fp8_fp16"):
        for mode in ("ragged", "masked", "contiguous"):
            alignments = (128,) if mode == "contiguous" else (128, 256)
            for alignment in alignments:
                name = f"xllm_mate_moe_{mode}_{dtype}_m{alignment}"
                recipes.append(
                    {
                        "name": name,
                        "constant": "kMoe" + "".join(part.title() for part in name.split("_")[3:]),
                        "kind": "moe_gemm",
                        "config": {"dtype": dtype, "mode": mode, "alignment": alignment},
                        "arity": 11,
                        "assets": ["kernel.o"],
                    }
                )
    return recipes


OPERATOR_RECIPES = _operator_recipes()


def _fused_moe_recipes() -> list[dict[str, Any]]:
    recipes = []
    for fp8 in (False, True):
        for batch in range(1, 9):
            if fp8:
                block_m, block_n, block_k, group_m, warps = (
                    (32, 32, 128, 64, 4) if batch == 1 else (16, 64, 128, 32, 4) if batch <= 4 else (32, 64, 128, 64, 4)
                )
            else:
                block_m, block_n, block_k, group_m, warps = (
                    (32, 32, 128, 1, 4) if batch == 1 else (16, 64, 64, 32, 4) if batch <= 3 else (32, 64, 128, 1, 8)
                )
            for stage in ("gate", "down"):
                recipes.append(
                    {
                        "path": f"xllm_musa_fused_moe_{'fp8' if fp8 else 'bf16'}_aot/b{batch}_{stage}.mubin",
                        "batch_size": batch,
                        "fp8": fp8,
                        "stage": stage,
                        "block_m": block_m,
                        "block_n": block_n,
                        "block_k": block_k,
                        "group_m": group_m,
                        "num_warps": warps,
                        "num_stages": 1,
                    }
                )
    return recipes


FUSED_MOE_RECIPES = _fused_moe_recipes()


def header_content(
    bundle_id: str,
    operators: list[dict[str, Any]],
    fused_moe_kernels: list[dict[str, Any]],
    manifest_sha512: str,
) -> str:
    lines = [
        "// Copyright 2026 The xLLM Authors. All Rights Reserved.",
        "// Generated by xllm/core/kernels/musa/build.py; do not edit.",
        "#pragma once",
        "#include <array>",
        "#include <cstdint>",
        "namespace xllm::kernel::musa::aot {",
        f'inline constexpr const char* kBundleId = "{bundle_id}";',
    ]
    lines.extend(f'inline constexpr const char* {r["constant"]} = "{r["name"]}";' for r in OPERATOR_RECIPES)
    lines.append(f"inline constexpr std::array<const char*, {len(OPERATOR_RECIPES)}> kOperatorUris = {{")
    lines.extend(f"  {r['constant']}," for r in OPERATOR_RECIPES)
    lines.append("};")
    tilelang = [r for r in OPERATOR_RECIPES if r["kind"].startswith("gdn_")]
    lines.append(f"inline constexpr std::array<const char*, {len(tilelang)}> kTileLangOperatorUris = {{")
    lines.extend(f"  {r['constant']}," for r in tilelang)
    lines.append("};")
    artifacts = {"manifest.json": manifest_sha512}
    for record in operators:
        name = record["name"]
        artifacts[f"{name}/{name}.so"] = record["sha512"]
        artifacts.update({f"{name}/{filename}": digest for filename, digest in record["assets"].items()})
    artifacts.update({record["path"]: record["sha512"] for record in fused_moe_kernels})
    lines.extend(
        [
            "struct Artifact {",
            "  const char* path;",
            "  const char* sha512;",
            "};",
            f"inline constexpr std::array<Artifact, {len(artifacts)}> kArtifacts = {{{{",
        ]
    )
    lines.extend(f'  {{"{path}", "{digest}"}},' for path, digest in sorted(artifacts.items()))
    lines.append("}};")
    moe = [r for r in OPERATOR_RECIPES if r["kind"] == "moe_gemm"]
    lines.extend(
        [
            "struct MoeGemmConfig {",
            "  const char* uri;",
            "  const char* mode;",
            "  bool fp8;",
            "  bool fp16;",
            "  int64_t alignment;",
            "};",
            f"inline constexpr std::array<MoeGemmConfig, {len(moe)}> kMoeGemmOperators = {{{{",
        ]
    )
    for recipe in moe:
        config = recipe["config"]
        fp8 = "true" if config["dtype"].startswith("fp8_") else "false"
        fp16 = "true" if config["dtype"].endswith("fp16") else "false"
        lines.append(f'  {{{recipe["constant"]}, "{config["mode"]}", {fp8}, {fp16}, {config["alignment"]}}},')
    lines.append("}};")
    lines.extend(
        [
            "struct FusedMoeKernel {",
            "  const char* path;",
            "  int64_t batch_size;",
            "  int64_t block_m;",
            "  int64_t block_n;",
            "  bool use_fp8;",
            "  bool down;",
            "  uint32_t threads_per_block;",
            "  uint32_t shared_memory;",
            "};",
            f"inline constexpr std::array<FusedMoeKernel, {len(fused_moe_kernels)}> kFusedMoeKernels = {{{{",
        ]
    )
    for record in fused_moe_kernels:
        fp8 = "true" if record["fp8"] else "false"
        down = "true" if record["stage"] == "down" else "false"
        lines.append(
            f'  {{"{record["path"]}", {record["batch_size"]}, {record["block_m"]}, '
            f"{record['block_n']}, {fp8}, {down}, {record['threads_per_block']}, {record['shared_memory']}}},"
        )
    lines.append("}};")
    # Both stages reuse token IDs and expert IDs packed with GATE's block_m.
    kernel_indices = {
        (record["fp8"], record["batch_size"], record["stage"]): index for index, record in enumerate(fused_moe_kernels)
    }
    for fp8, batch in sorted({(record["fp8"], record["batch_size"]) for record in fused_moe_kernels}):
        gate = kernel_indices[fp8, batch, "gate"]
        down = kernel_indices[fp8, batch, "down"]
        lines.append(
            f"static_assert(kFusedMoeKernels[{gate}].block_m == kFusedMoeKernels[{down}].block_m, "
            '"MoE shared token packing requires matching gate/down block_m");'
        )
    lines.extend(["}  // namespace xllm::kernel::musa::aot", ""])
    return "\n".join(lines)
