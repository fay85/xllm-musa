# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Export GLM TP8-attention / EP8-expert MATE stages (layout revision 3)."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ops-path', type=Path, required=True)
    args = parser.parse_args()
    from mate.jit.mega_moe import (
        gen_mega_moe_runtime_utils_spec, MegaMoEKernelConfig,
        gen_fp8_fp8_mega_moe_stage1_spec, gen_fp8_fp8_mega_moe_stage2_spec,
    )
    from mate.jit import env as jit_env
    from mate.jit.mega_moe_prefill import (
        PrefillMegaMoEKernelConfig, gen_prefill_stage1_spec,
        gen_prefill_stage2_spec, gen_prefill_stage3_spec,
    )
    import torch
    import torch_musa
    import tvm_ffi

    props = torch.musa.get_device_properties(0)
    config = PrefillMegaMoEKernelConfig(
        num_ranks=8, num_experts=288, num_max_tokens_per_rank=512,
        num_topk=8, hidden=4096, intermediate_hidden=2048,
        num_mps=props.multi_processor_count, fast_math=False,
    )
    # Extend the installed decode ABI in a private header overlay. Do not
    # modify the installed MATE package or silently discard GLM's clamp.
    relative = Path('mate/mega_moe/decode/mp31_decode_stage1.hpp')
    header = (jit_env.MATE_INCLUDE_DIR / relative).read_text()
    old = 'const float              activation_beta,'
    if header.count(old) != 1:
        raise RuntimeError('MATE decode signature changed; revalidate clamp patch')
    header = header.replace(old, 'const float activation_clamp,\n' + old)
    old = 'up_values[i]   = static_cast<float>(ElementD(rAcc(gate_i + kPairOffset + i)));'
    if header.count(old) != 1:
        raise RuntimeError('MATE SwiGLU implementation changed; revalidate patch')
    header = header.replace(old, old + """
                if constexpr (!kUseSitu) {
                  if (activation_clamp >= 0.0f) {
                    gate_values[i] = fminf(gate_values[i], activation_clamp);
                    up_values[i] = fminf(fmaxf(up_values[i], -activation_clamp), activation_clamp);
                  }
                }
""")
    overlay = args.ops_path.parent / 'glm53_mega_overlay_v1'
    (overlay / relative).parent.mkdir(parents=True, exist_ok=True)
    (overlay / relative).write_text(header)
    decode = MegaMoEKernelConfig(**{key: getattr(config, key) for key in
                                   MegaMoEKernelConfig.__dataclass_fields__})
    specs = [gen_mega_moe_runtime_utils_spec(),
             gen_prefill_stage1_spec(config), gen_prefill_stage2_spec(config),
             gen_prefill_stage3_spec(replace(config, num_mps=2*config.num_mps)),
             gen_fp8_fp8_mega_moe_stage1_spec(decode),
             gen_fp8_fp8_mega_moe_stage2_spec(decode)]
    for index, spec in enumerate(specs):
        entry = 'mega_moe_unified_get_buffer_layout'
        if index:
            alias = (f'glm53_mega_stage{index}_v1' if index <= 3 else
                     f'glm53_mega_decode{index-3}_v1')
            sources = {path.with_name(alias+'.mu'): source.replace(spec.name, alias)
                       for path, source in spec.generated_sources.items()}
            if index == 4:
                sources = {path: source.replace('  float activation_beta,',
                                                '  float activation_clamp,\n  float activation_beta,')
                                      .replace('      activation_beta,',
                                               '      activation_clamp,\n      activation_beta,')
                           for path, source in sources.items()}
                spec = replace(spec, extra_include_dirs=[overlay, *spec.extra_include_dirs])
            spec = replace(spec, name=alias, sources=list(sources), generated_sources=sources)
            entry = alias
        spec = replace(spec, extra_cuda_cflags=[*(spec.extra_cuda_cflags or []),
                                               "-DCCCL_DISABLE_NVFP8_SUPPORT"])
        library = spec.build()
        if tvm_ffi.load_module(str(library)).get_function(entry) is None:
            raise RuntimeError(f'Missing {entry} in {library}')
        target = args.ops_path / spec.name / f'{spec.name}.so'
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f'.{os.getpid()}.tmp.so')
        shutil.copy2(library, temporary)
        temporary.replace(target)
        logger.info('Exported %s', target)


if __name__ == '__main__':
    main()
