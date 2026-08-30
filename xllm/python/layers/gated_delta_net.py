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

"""Qwen3.5 gated delta network layer for the Python executor."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.model_executor.forward_context import (
    acquire_graph_activation,
    get_forward_context,
)


class GatedDeltaNetConfig(Protocol):
    hidden_size: int
    rms_norm_eps: float
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    tp_size: int


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self,
        cfg: GatedDeltaNetConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_k_heads = cfg.linear_num_key_heads // cfg.tp_size
        self.num_v_heads = cfg.linear_num_value_heads // cfg.tp_size
        self.key_head_dim = cfg.linear_key_head_dim
        self.value_head_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.key_head_dim
        self.value_dim = self.num_v_heads * self.value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.norm_eps = cfg.rms_norm_eps
        self.gdn_prefill_backend = kernels.resolve_gdn_prefill_backend()
        self.gdn_decode_backend = kernels.resolve_gdn_decode_backend()
        self._rms_norm_gated_out: torch.Tensor | None = None
        self._gdn_initial_state: torch.Tensor | None = None
        self._gdn_cache_indices: torch.Tensor | None = None
        self._gdn_ssm_write: torch.Tensor | None = None
        self._gdn_z: torch.Tensor | None = None

        self.merge_qkvz = cfg.tp_size == 1
        if self.merge_qkvz:
            self.in_proj_qkvz = ColumnParallelLinear(
                cfg.hidden_size,
                self.conv_dim + self.value_dim,
                cfg.tp_size,
                dtype=dtype,
                device=device,
            )
            self.in_proj_qkv = None
            self.in_proj_z = None
        else:
            self.in_proj_qkvz = None
            self.in_proj_qkv = ColumnParallelLinear(
                cfg.hidden_size,
                self.conv_dim,
                cfg.tp_size,
                dtype=dtype,
                device=device,
            )
            self.in_proj_z = ColumnParallelLinear(
                cfg.hidden_size,
                self.value_dim,
                cfg.tp_size,
                dtype=dtype,
                device=device,
            )
        # Match C++ TP=1: one ba GEMM instead of two tiny N=H_v launches.
        self.merge_ba = cfg.tp_size == 1
        if self.merge_ba:
            self.in_proj_ba = ColumnParallelLinear(
                cfg.hidden_size,
                2 * self.num_v_heads,
                cfg.tp_size,
                dtype=dtype,
                device=device,
            )
            self.in_proj_b = None
            self.in_proj_a = None
        else:
            self.in_proj_ba = None
            self.in_proj_b = ColumnParallelLinear(
                cfg.hidden_size,
                self.num_v_heads,
                cfg.tp_size,
                dtype=dtype,
                device=device,
            )
            self.in_proj_a = ColumnParallelLinear(
                cfg.hidden_size,
                self.num_v_heads,
                cfg.tp_size,
                dtype=dtype,
                device=device,
            )
        self.conv1d_weight = nn.Parameter(
            torch.empty(
                self.conv_dim,
                self.conv_kernel_size,
                dtype=dtype,
                device=device,
            )
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads, dtype=torch.float32, device=device)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.num_v_heads, dtype=torch.float32, device=device)
        )
        self.norm_weight = nn.Parameter(
            torch.ones(self.value_head_dim, dtype=dtype, device=device)
        )
        self.out_proj = RowParallelLinear(
            self.value_dim,
            cfg.hidden_size,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )

    def reserve_graph_workspace(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise RuntimeError("GDN graph reserve requires max_tokens > 0")
        if self.in_proj_qkvz is not None:
            self.in_proj_qkvz.reserve_graph_workspace(max_tokens)
        else:
            self.in_proj_qkv.reserve_graph_workspace(max_tokens)
            self.in_proj_z.reserve_graph_workspace(max_tokens)
        if self.in_proj_ba is not None:
            self.in_proj_ba.reserve_graph_workspace(max_tokens)
        else:
            if self.in_proj_b is None or self.in_proj_a is None:
                raise RuntimeError("GDN TP>1 projections were not created")
            self.in_proj_b.reserve_graph_workspace(max_tokens)
            self.in_proj_a.reserve_graph_workspace(max_tokens)
        self.out_proj.reserve_graph_workspace(max_tokens)
        self._rms_norm_gated_out = torch.empty(
            (max_tokens, self.num_v_heads, self.value_head_dim),
            dtype=self.conv1d_weight.dtype,
            device=self.conv1d_weight.device,
        )
        # C++ C=1 prefill packs z with .contiguous() so gated RMSNorm does
        # not reshape-copy a qkvz.split() view on every layer.
        self._gdn_z = torch.empty(
            (max_tokens, self.num_v_heads, self.value_head_dim),
            dtype=self.conv1d_weight.dtype,
            device=self.conv1d_weight.device,
        )
        self._gdn_initial_state = torch.zeros(
            (1, self.num_v_heads, self.value_head_dim, self.key_head_dim),
            dtype=torch.float32,
            device=self.conv1d_weight.device,
        )
        self._gdn_cache_indices = torch.empty(
            (1,),
            dtype=torch.int64,
            device=self.conv1d_weight.device,
        )
        # Qwen3.5 mamba_ssm_dtype is float32. Mate k-last state is [H, V, K].
        self._gdn_ssm_write = torch.empty(
            (1, self.num_v_heads, self.value_head_dim, self.key_head_dim),
            dtype=torch.float32,
            device=self.conv1d_weight.device,
        )

    def refresh_graph_ssm_workspace(self, ssm_state: torch.Tensor) -> None:
        if ssm_state.numel() <= 0:
            return
        slot0 = ssm_state[0]
        write_shape = (1, *tuple(slot0.shape))
        if (
            self._gdn_ssm_write is None
            or tuple(self._gdn_ssm_write.shape[1:]) != tuple(slot0.shape)
            or self._gdn_ssm_write.dtype != ssm_state.dtype
            or self._gdn_ssm_write.device != ssm_state.device
        ):
            self._gdn_ssm_write = torch.empty(
                write_shape,
                dtype=ssm_state.dtype,
                device=ssm_state.device,
            )

    def _cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        layer_caches = get_forward_context().layer_caches
        if layer_caches is None:
            raise RuntimeError("linear-attention caches are not bound")
        cache = layer_caches[self.layer_id]
        if cache.conv is None or cache.ssm is None:
            raise RuntimeError(
                f"linear-attention cache is missing for layer {self.layer_id}"
            )
        return cache.conv, cache.ssm

    def _conv_state_dim_first(self, conv_state: torch.Tensor) -> bool:
        if conv_state.size(1) == self.conv_dim:
            return True
        if conv_state.size(2) == self.conv_dim:
            return False
        raise ValueError("linear-attention conv cache has an unexpected shape")

    def _conv_prefill(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        self._conv_state_dim_first(conv_state)
        return kernels.causal_conv1d_prefill(
            mixed_qkv,
            self.conv1d_weight,
            conv_state,
            state_indices,
            has_initial_state,
            cu_seqlens,
        )

    def _conv_decode(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor:
        self._conv_state_dim_first(conv_state)
        return kernels.causal_conv1d_decode(
            mixed_qkv,
            self.conv1d_weight,
            conv_state,
            state_indices,
        )

    @staticmethod
    def _expand_sequence_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        tensor = tensor.reshape(-1)
        if tensor.numel() == batch_size:
            return tensor.contiguous()
        if tensor.numel() <= 0 or batch_size % tensor.numel():
            raise ValueError("cannot expand sequence metadata to verify batch")
        return (
            tensor.unsqueeze(1)
            .expand(tensor.numel(), batch_size // tensor.numel())
            .reshape(batch_size)
            .contiguous()
        )

    def _conv_spec_verify(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        state_indices: torch.Tensor,
        accepted_tokens: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Run MTP convolution and retain the candidate-token superstate."""
        expanded_state_len = conv_state.size(2)
        history_len = self.conv_kernel_size - 1
        if expanded_state_len != history_len + seq_len - 1:
            raise ValueError("unexpected Qwen3.5 MTP convolution cache width")
        logical_indices = self._expand_sequence_tensor(state_indices, batch_size).to(
            device=mixed_qkv.device, dtype=torch.int32
        )
        accepted = self._expand_sequence_tensor(accepted_tokens, batch_size).to(
            device=mixed_qkv.device, dtype=torch.int32
        )
        return kernels.causal_conv1d_mtp_verify(
            mixed_qkv.contiguous(),
            self.conv1d_weight,
            conv_state,
            logical_indices,
            accepted,
            self.num_k_heads,
            self.key_head_dim,
            self.value_head_dim,
        )

    def _gdn_spec_verify(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_state: torch.Tensor,
        state_indices: torch.Tensor,
        accepted_tokens: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Run fused MTP recurrence and save every candidate checkpoint."""
        logical_indices = self._expand_sequence_tensor(state_indices, batch_size).to(
            device=mixed_qkv.device, dtype=torch.int32
        )
        accepted = self._expand_sequence_tensor(accepted_tokens, batch_size).to(
            device=mixed_qkv.device, dtype=torch.int32
        )
        checkpoint_stride = ssm_state.size(0) // conv_state.size(0)
        if checkpoint_stride != seq_len:
            raise ValueError("MTP checkpoint stride must match verify width")
        return kernels.fused_gdn_mtp_checkpoint(
            mixed_qkv.contiguous(),
            a.contiguous(),
            b.contiguous(),
            self.A_log,
            self.dt_bias,
            ssm_state,
            logical_indices.contiguous(),
            accepted.contiguous(),
            checkpoint_stride,
            self.key_head_dim**-0.5,
            self.num_k_heads,
            self.key_head_dim,
            self.value_head_dim,
        )

    def _gdn_prefill(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        cache_indices = self._acquire_cache_indices(state_indices)
        batch_size = int(state_indices.numel())
        initial_state = self._acquire_gdn_initial_state(ssm_state, batch_size)
        # Graph capture records the zero-init branch used by C++ piecewise
        # prefill. Prefix restore stays on eager model() so index_select
        # and device .any() are not captured.
        if get_forward_context().graph_mode:
            initial_state.zero_()
        else:
            use_initial_state = (state_indices > 0) & has_initial_state
            if bool(use_initial_state.any()):
                selected = ssm_state.index_select(0, cache_indices)
                initial_state.copy_(selected.to(dtype=torch.float32))
                initial_state.mul_(
                    use_initial_state.to(device=initial_state.device).view(
                        batch_size, 1, 1, 1
                    )
                )
            else:
                initial_state.zero_()
        q, k, v, g, beta = kernels.fused_gdn_prefill_post_conv(
            mixed_qkv,
            a,
            b,
            self.A_log,
            self.dt_bias,
            self.num_k_heads,
            self.key_head_dim,
            self.value_head_dim,
        )
        output, final_state = kernels.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            self.gdn_prefill_backend,
        )
        live_tokens = mixed_qkv.size(0)
        if output.size(0) > live_tokens:
            # KKT alignment may extend q/k/v; recurrent Mate still stops at
            # the live cu_seqlens endpoint. Slice is a packed prefix view.
            output = output[:live_tokens]
        # Match C++ index_put_. Slot-0 backup/restore was extra D2D
        # traffic captured into the Python graph (2 copies x 48 layers).
        write_state = self._acquire_ssm_write_state(ssm_state, final_state, batch_size)
        if write_state.data_ptr() != final_state.data_ptr():
            write_state.copy_(final_state)
        ssm_state.index_put_((cache_indices,), write_state)
        return output

    def _acquire_gdn_initial_state(
        self, ssm_state: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        reserved = self._gdn_initial_state
        if reserved is not None and batch_size <= reserved.size(0):
            return reserved[:batch_size]
        if get_forward_context().graph_mode:
            raise RuntimeError(
                "GDN initial-state buffer was not reserved before capture"
            )
        return torch.zeros(
            (
                batch_size,
                self.num_v_heads,
                self.value_head_dim,
                self.key_head_dim,
            ),
            dtype=torch.float32,
            device=ssm_state.device,
        )

    def _acquire_cache_indices(self, state_indices: torch.Tensor) -> torch.Tensor:
        batch_size = int(state_indices.numel())
        reserved = self._gdn_cache_indices
        if reserved is not None and batch_size <= reserved.numel():
            view = reserved[:batch_size]
            view.copy_(state_indices)
            return view
        if get_forward_context().graph_mode:
            raise RuntimeError("GDN cache-index buffer was not reserved before capture")
        return state_indices.to(dtype=torch.long)

    def _acquire_ssm_write_state(
        self,
        ssm_state: torch.Tensor,
        final_state: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if final_state.dtype == ssm_state.dtype:
            return final_state
        reserved = self._gdn_ssm_write
        if (
            reserved is not None
            and reserved.dtype == ssm_state.dtype
            and batch_size <= reserved.size(0)
            and reserved.shape[1:] == ssm_state.shape[1:]
        ):
            return reserved[:batch_size]
        if get_forward_context().graph_mode:
            raise RuntimeError("GDN SSM write buffer was not reserved before capture")
        return final_state.to(dtype=ssm_state.dtype)

    def _pack_gdn_z(self, z_flat: torch.Tensor) -> torch.Tensor:
        tokens = int(z_flat.size(0))
        if (
            z_flat.is_contiguous()
            and z_flat.dim() == 2
            and int(z_flat.size(-1)) == self.value_dim
        ):
            return z_flat.view(tokens, self.num_v_heads, self.value_head_dim)
        packed = acquire_graph_activation(
            (tokens, self.num_v_heads, self.value_head_dim),
            z_flat.dtype,
            z_flat.device,
        )
        if packed is None and self._gdn_z is not None and tokens <= self._gdn_z.size(0):
            packed = self._gdn_z[:tokens]
        if packed is None:
            if get_forward_context().graph_mode:
                raise RuntimeError("GDN z buffer was not reserved before capture")
            return z_flat.contiguous().view(
                tokens, self.num_v_heads, self.value_head_dim
            )
        packed.view(tokens, self.value_dim).copy_(z_flat)
        return packed

    def _gdn_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor:
        decode_fn = (
            kernels.mate_gated_delta_rule_decode
            if self.gdn_decode_backend == "mate"
            else kernels.fused_recurrent_gated_delta_rule_packed_decode
        )
        output = decode_fn(
            mixed_qkv,
            a,
            b,
            self.A_log,
            self.dt_bias,
            ssm_state,
            state_indices,
            self.key_head_dim**-0.5,
            self.num_k_heads,
            self.key_head_dim,
            self.value_head_dim,
        )
        return output.view(-1, self.num_v_heads, self.value_head_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        context = get_forward_context()
        metadata = context.metadata
        if metadata is None:
            raise RuntimeError("Qwen3.5 GDN requires attention metadata")
        state_indices = metadata.linear_state_indices
        if state_indices is None:
            raise RuntimeError("linear_state_indices are required by Qwen3.5")
        if state_indices.device != hidden.device or state_indices.dtype != torch.int32:
            state_indices = state_indices.to(device=hidden.device, dtype=torch.int32)
        has_initial_state = metadata.has_initial_state
        cu_seqlens = getattr(metadata, "gdn_cu_seq_lens", None)
        if cu_seqlens is None:
            cu_seqlens = metadata.q_cu_seq_lens
        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                state_indices.numel() + 1,
                dtype=torch.int32,
                device=hidden.device,
            )

        conv_state, ssm_state = self._cache()
        if self.in_proj_qkvz is not None:
            qkvz = self.in_proj_qkvz(hidden)
            mixed_qkv, z_flat = qkvz.split([self.conv_dim, self.value_dim], dim=-1)
            z = self._pack_gdn_z(z_flat)
        else:
            mixed_qkv = self.in_proj_qkv(hidden)
            z = self._pack_gdn_z(self.in_proj_z(hidden))
        if self.in_proj_ba is not None:
            ba = self.in_proj_ba(hidden)
            b, a = ba.split([self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            if self.in_proj_b is None or self.in_proj_a is None:
                raise RuntimeError("GDN TP>1 projections were not created")
            b = self.in_proj_b(hidden)
            a = self.in_proj_a(hidden)
        is_prefill = metadata.is_prefill or metadata.is_chunked_prefill
        is_spec_verify = bool(getattr(metadata, "is_spec_verify", False))
        if is_spec_verify:
            accepted_tokens = getattr(metadata, "num_accepted_tokens", None)
            if accepted_tokens is None:
                raise RuntimeError(
                    "num_accepted_tokens are required for Qwen3.5 MTP verify"
                )
            batch_size = state_indices.numel()
            if hidden.size(0) % batch_size:
                raise ValueError("MTP verify tokens must be dense per sequence")
            seq_len = hidden.size(0) // batch_size
            mixed_qkv = self._conv_spec_verify(
                mixed_qkv,
                conv_state,
                state_indices,
                accepted_tokens,
                batch_size,
                seq_len,
            )
        elif is_prefill:
            if has_initial_state is None:
                raise RuntimeError("has_initial_state is required by Qwen3.5 prefill")
            if (
                has_initial_state.device != hidden.device
                or has_initial_state.dtype != torch.bool
            ):
                has_initial_state = has_initial_state.to(
                    device=hidden.device, dtype=torch.bool
                )
            if has_initial_state.shape != state_indices.shape:
                raise ValueError("has_initial_state must match linear_state_indices")
            mixed_qkv = self._conv_prefill(
                mixed_qkv,
                conv_state,
                state_indices,
                has_initial_state,
                cu_seqlens,
            )
        else:
            mixed_qkv = self._conv_decode(mixed_qkv, conv_state, state_indices)

        if is_spec_verify:
            output = self._gdn_spec_verify(
                mixed_qkv,
                a,
                b,
                ssm_state,
                conv_state,
                state_indices,
                accepted_tokens,
                batch_size,
                seq_len,
            )
        elif is_prefill:
            output = self._gdn_prefill(
                mixed_qkv,
                a,
                b,
                ssm_state,
                state_indices,
                has_initial_state,
                cu_seqlens,
            )
        else:
            output = self._gdn_decode(mixed_qkv, a, b, ssm_state, state_indices)
        gated_output = acquire_graph_activation(
            (
                int(output.size(0)),
                int(self.num_v_heads),
                int(self.value_head_dim),
            ),
            output.dtype,
            output.device,
        )
        if (
            gated_output is None
            and self._rms_norm_gated_out is not None
            and output.size(0) <= self._rms_norm_gated_out.size(0)
        ):
            gated_output = self._rms_norm_gated_out[: output.size(0)]
        output = kernels.rms_norm_gated(
            output,
            z,
            self.norm_weight,
            self.norm_eps,
            gated_output,
        )
        return self.out_proj(output.reshape(-1, self.value_dim))
