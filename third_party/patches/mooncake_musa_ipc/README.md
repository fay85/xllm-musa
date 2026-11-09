<!-- Copyright 2026 The xLLM Authors. -->

# Pinned Mooncake overlay for MUSA IPC

The parent repository keeps the existing Mooncake gitlink at
`bb2c0bee9f7839789c947556341556b4c02aaeda` from
`https://gitcode.com/xLLM-AI/Mooncake.git`. This directory carries the complete
source overlay so publication does not depend on an unpublished submodule commit.

From the parent repository root:

```bash
git submodule update --init third_party/Mooncake
python3 scripts/mtlink_prepare_mooncake.py --apply
python3 scripts/mtlink_prepare_mooncake.py --check
```

The helper requires the exact pinned HEAD and verifies the patch SHA256 and every
affected source file. It applies only to the complete expected original state;
running it on the complete patched state verifies it without writes. Mixed states
and edits to affected files are rejected. `--check` never changes source files.
An optional `--mooncake-root /path/to/checkout` supports isolated preparation.
The check covers this overlay's files, not unrelated local modifications.

An applied overlay leaves the submodule working tree dirty while its HEAD stays
pinned. Resetting or forcing an update of the submodule can remove the overlay;
rerun the helper before building. Initialize the other project dependencies using
the normal xLLM build procedure.

`manifest.json` records before/after SHA256 hashes, file modes, the patch hash,
original experiment hashes, and golden-baseline hashes. The patch includes four
inherited baseline fixes in `common.cmake`, two store implementations, and the
IBGDA OS header, as well as the five MTLink implementation/header/kernel files.
The two `.orig` backup files are excluded. The exported C++/CUDA source contents
are preserved. Additional CMake changes make the experimental feature selectable
through the regular build graph.

With the project's existing MUSA SDK/dependency configuration, add:

```bash
cmake -S . -B build-mtlink \
  -DUSE_MUSA=ON -DUSE_INTRA_NVLINK=ON
cmake --build build-mtlink --parallel 2
```

`USE_INTRA_NVLINK` remains off by default. To compile the optional fused copy
kernel, also pass `-DUSE_MUSA_IPC_COPY_KERNEL=ON`; it defaults to off and requires
both `USE_MUSA` and `USE_INTRA_NVLINK`. `MOONCAKE_MUSA_ARCH` defaults to `mp_31`.
`MOONCAKE_MUSA_COMPILER` may select a specific native `mcc` executable. CMake
builds the kernel as a native MUSA object/archive and links it through
`transfer_engine`; the class-layout definition is public to its consumers.

Build options make the code available; the runtime environment selects the IPC
route and batch engine. See [the branch guide](../../../docs/MUSA_MTLINK_IPC.md)
for runtime configuration, measured results, and validation limits.
