# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MUSA tensor-parallel collectives (MCCL)."""

from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401

_tp_groups: dict[str, object] = {}
_tp_stores: dict[str, object] = {}


def _create_process_group(
    host: str, port: int, rank: int, world_size: int, device: str
) -> tuple[object, object]:
    from torch_musa._C._distributed_c10d import ProcessGroupMCCL

    store = dist.TCPStore(
        host,
        port,
        world_size,
        rank == 0,
        timedelta(minutes=5),
        wait_for_workers=False,
    )
    del device
    group = ProcessGroupMCCL(store, rank, world_size, timedelta(minutes=5))
    return store, group


def init_tp_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: str,
) -> object:
    device_key = str(torch.device(device))
    group = _tp_groups.get(device_key)
    if group is not None:
        if group.rank() != rank or group.size() != world_size:
            raise RuntimeError(
                f"TP group for {device_key} is already initialized as "
                f"rank {group.rank()}/{group.size()}, requested "
                f"rank {rank}/{world_size}"
            )
        return group

    store, group = _create_process_group(host, port, rank, world_size, device)
    _tp_stores[device_key] = store
    _tp_groups[device_key] = group
    return group


def _require_tp_group(x: torch.Tensor) -> object:
    group = _tp_groups.get(str(x.device))
    if group is None:
        raise RuntimeError(
            "tensor-parallel collective called before the TP process group "
            f"was initialized for {x.device}"
        )
    return group


@torch.library.custom_op("xllm_ops::all_reduce_", mutates_args={"x"})
def all_reduce_(x: torch.Tensor) -> None:
    group = _require_tp_group(x)
    dist.all_reduce(x, group=group)


@all_reduce_.register_fake
def _(x: torch.Tensor) -> None:
    del x
    return None


@torch.library.custom_op("xllm_ops::all_gather", mutates_args=())
def all_gather(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    group = _require_tp_group(x)
    if group.size() != world_size:
        raise RuntimeError(
            f"TP world-size mismatch: expected {world_size}, got {group.size()}"
        )
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(chunks, x, group=group)
    return torch.cat(chunks, dim=dim)


@all_gather.register_fake
def _(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] *= world_size
    return x.new_empty(shape)
