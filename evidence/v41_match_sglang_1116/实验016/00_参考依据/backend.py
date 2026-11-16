"""Backend boundary for W4A8 MoE weight preparation and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.utils import set_weight_attrs

from .scales import convert_cutlass_scale_to_mate, interleave_cutlass_scales

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        StandardDispatchOutput,
    )


@dataclass
class W4A8ExecutionBuffers:
    a_strides1: torch.Tensor
    b_strides1: torch.Tensor
    c_strides1: torch.Tensor
    a_strides2: torch.Tensor
    b_strides2: torch.Tensor
    c_strides2: torch.Tensor
    s_strides13: torch.Tensor
    s_strides2: torch.Tensor
    expert_offsets: torch.Tensor
    problem_sizes1: torch.Tensor
    problem_sizes2: torch.Tensor

    def kernel_args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.a_strides1,
            self.b_strides1,
            self.c_strides1,
            self.a_strides2,
            self.b_strides2,
            self.c_strides2,
            self.s_strides13,
            self.s_strides2,
            self.expert_offsets,
            self.problem_sizes1,
            self.problem_sizes2,
        )


class W4A8MoEBackend:
    """Own backend-specific W4A8 state behind SGLang's quant method seam."""

    is_musa = False

    def __init__(self, *, group_size: int = 128) -> None:
        self.group_size = group_size
        self.buffers: Optional[W4A8ExecutionBuffers] = None
        self.moe_runner_config: Optional[MoeRunnerConfig] = None

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        assert "weight_loader" in extra_weight_attrs
        w13_weight = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition * 2,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        scale_attrs = {
            **extra_weight_attrs,
            "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
        }
        w13_weight_scale = Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)

        w2_weight_scale = Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

        w13_input_scale = Parameter(
            torch.ones((num_experts, 2), dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, scale_attrs)

        w2_input_scale = Parameter(
            torch.ones(num_experts, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, scale_attrs)

        device = layer.w13_weight.device
        a_strides1 = torch.full(
            (num_experts, 3), hidden_size, device=device, dtype=torch.int64
        )
        c_strides1 = torch.full(
            (num_experts, 3),
            2 * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        a_strides2 = torch.full(
            (num_experts, 3),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        c_strides2 = torch.full(
            (num_experts, 3), hidden_size, device=device, dtype=torch.int64
        )
        self.buffers = W4A8ExecutionBuffers(
            a_strides1=a_strides1,
            b_strides1=a_strides1,
            c_strides1=c_strides1,
            a_strides2=a_strides2,
            b_strides2=a_strides2,
            c_strides2=c_strides2,
            s_strides13=c_strides1,
            s_strides2=c_strides2,
            expert_offsets=torch.empty(
                num_experts + 1, dtype=torch.int32, device=device
            ),
            problem_sizes1=torch.empty(
                (num_experts, 3), dtype=torch.int32, device=device
            ),
            problem_sizes2=torch.empty(
                (num_experts, 3), dtype=torch.int32, device=device
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        device = layer.w2_weight.device
        layer.w13_weight_scale_inv = Parameter(
            self._convert_weight_scale(
                layer.w13_weight_scale_inv.to(torch.bfloat16),
                layer.w13_weight.shape[1],
            ),
            requires_grad=False,
        )
        layer.w2_weight_scale_inv = Parameter(
            self._convert_weight_scale(
                layer.w2_weight_scale_inv.to(torch.bfloat16),
                layer.w2_weight.shape[1],
            ),
            requires_grad=False,
        )

        input_scale_dtype = torch.bfloat16 if self.is_musa else torch.float32
        w13_input_scale = torch.tensor(
            [layer.w13_input_scale.max().to(torch.float32).item()],
            dtype=input_scale_dtype,
            device=device,
        )
        w2_input_scale = torch.tensor(
            [layer.w2_input_scale.max().to(torch.float32).item()],
            dtype=input_scale_dtype,
            device=device,
        )
        layer.w13_input_scale = Parameter(w13_input_scale, requires_grad=False)
        layer.w2_input_scale = Parameter(w2_input_scale, requires_grad=False)
        self._register_runtime_scale_buffers(layer, w13_input_scale, w2_input_scale)
        # The dispatcher is created before post-load processing.  Configure it
        # here so NORMAL receives BF16 and LOW_LATENCY can carry FP8 scales on
        # MUSA W4A8.  Without this call the dispatcher keeps its default dtype
        # and the operator sees an incompatible activation contract.
        dispatcher = getattr(layer, "dispatcher", None)
        if dispatcher is not None:
            dispatcher.set_quant_config(self.get_dispatch_quant_config())

    def _convert_weight_scale(
        self, scales: torch.Tensor, n_logical: int
    ) -> torch.Tensor:
        return interleave_cutlass_scales(scales)

    def _register_runtime_scale_buffers(
        self,
        layer: Module,
        w13_input_scale: torch.Tensor,
        w2_input_scale: torch.Tensor,
    ) -> None:
        return

    def create_moe_runner(
        self, layer: Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        del layer
        self.moe_runner_config = moe_runner_config

    def get_dispatch_quant_config(self) -> dict[str, Any]:
        return {"use_per_token_fp8_comm": True}

    def validate_deepep_mode(self, mode: Any) -> None:
        return

    def _local_kernel(self):
        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe

        return cutlass_w4a8_moe

    def _deepep_normal_kernel(self):
        from sglang.srt.layers.moe.cutlass_w4a8_moe import (
            cutlass_w4a8_moe_deepep_normal,
        )

        return cutlass_w4a8_moe_deepep_normal

    def _deepep_ll_kernel(self):
        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe_deepep_ll

        return cutlass_w4a8_moe_deepep_ll

    def _kernel_args(self) -> tuple[torch.Tensor, ...]:
        assert self.buffers is not None, "W4A8 backend weights were not created"
        return self.buffers.kernel_args()

    @staticmethod
    def _activation_scales(
        layer: Module,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return layer.w13_input_scale, layer.w2_input_scale

    def apply_local(
        self, layer: Module, dispatch_output: StandardDispatchOutput
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert self.moe_runner_config is not None
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        a1_scale, a2_scale = self._activation_scales(layer)
        output = self._local_kernel()(
            dispatch_output.hidden_states,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_weights,
            topk_ids,
            *self._kernel_args(),
            a1_scale,
            a2_scale,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,
            a1_mate_scale=getattr(layer, "w13_input_mate_scale", None),
            a2_mate_scale=getattr(layer, "w2_input_mate_scale", None),
        )
        return StandardCombineInput(hidden_states=output)

    def apply_deepep_normal(
        self, layer: Module, dispatch_output: DeepEPNormalDispatchOutput
    ) -> torch.Tensor:
        hidden_states = dispatch_output.hidden_states
        hidden_states_scale = dispatch_output.hidden_states_scale
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        if hidden_states.shape[0] == 0:
            if self.is_musa:
                return hidden_states.new_empty(hidden_states.shape, dtype=torch.bfloat16)
            return hidden_states

        a1_scale, a2_scale = self._activation_scales(layer)
        moe_input = (
            (hidden_states, hidden_states_scale)
            if self.is_musa and hidden_states_scale is not None
            else hidden_states
        )
        metadata_kwargs = (
            {"num_recv_tokens_per_expert": dispatch_output.num_recv_tokens_per_expert}
            if self.is_musa
            else {}
        )
        return self._deepep_normal_kernel()(
            moe_input,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            dispatch_output.topk_weights,
            dispatch_output.topk_ids,
            *self._kernel_args(),
            a1_scale,
            a2_scale,
            **metadata_kwargs,
        )

    def apply_deepep_ll(
        self, layer: Module, dispatch_output: DeepEPLLDispatchOutput
    ) -> torch.Tensor:
        hidden_states, hidden_scales, topk_ids, _, masked_m, _ = dispatch_output
        a1_scale, a2_scale = self._activation_scales(layer)
        scale_kwargs = {}
        if self.is_musa:
            from sglang.srt.environ import envs

            if envs.SGLANG_MUSA_W4A8_LL_SCALE_REUSE.get():
                scale_kwargs = dict(
                    a1_mate_scale=getattr(layer, "w13_input_mate_scale", None),
                    a2_mate_scale=getattr(layer, "w2_input_mate_scale", None),
                )
        return self._deepep_ll_kernel()(
            hidden_states,
            hidden_scales,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_ids,
            masked_m,
            *self._kernel_args(),
            a1_scale,
            a2_scale,
            **scale_kwargs,
        )


class MusaW4A8Backend(W4A8MoEBackend):
    is_musa = True

    def _convert_weight_scale(
        self, scales: torch.Tensor, n_logical: int
    ) -> torch.Tensor:
        return (
            convert_cutlass_scale_to_mate(scales, n_logical)
            .to(torch.bfloat16)
            .contiguous()
        )

    def _register_runtime_scale_buffers(
        self,
        layer: Module,
        w13_input_scale: torch.Tensor,
        w2_input_scale: torch.Tensor,
    ) -> None:
        from sglang.srt.environ import envs

        num_experts = layer.w13_weight.shape[0]
        masked_m = int(envs.SGLANG_DEEP_GEMM_BLOCK_M.get())
        layer.register_buffer(
            "w13_input_mate_scale",
            w13_input_scale.to(torch.float32)
            .expand(num_experts, masked_m)
            .contiguous(),
            persistent=False,
        )
        layer.register_buffer(
            "w2_input_mate_scale",
            w2_input_scale.to(torch.float32)
            .expand(num_experts, masked_m)
            .contiguous(),
            persistent=False,
        )

    def get_dispatch_quant_config(self) -> dict[str, Any]:
        from sglang.srt.environ import envs

        config: dict[str, Any] = {
            "use_per_token_fp8_comm": envs.SGLANG_W4AFP8_MUSA_PER_TOKEN_FP8_COMM.get(),
            "normal_dispatcher_output_dtype": "bf16",
            "low_latency_dispatcher_output_dtype": "bf16",
        }
        return config

    def validate_deepep_mode(self, mode: Any) -> None:
        if mode.enable_low_latency() and self.group_size != 128:
            raise ValueError(
                "MUSA W4A8 DeepEP low_latency requires group size 128, "
                f"got {self.group_size}."
            )

    def _local_kernel(self):
        from sglang.srt.layers.moe.musa_w4a8_moe import musa_w4a8_moe

        return musa_w4a8_moe

    def _deepep_normal_kernel(self):
        from sglang.srt.layers.moe.musa_w4a8_moe import (
            musa_w4a8_moe_deepep_normal,
        )

        return musa_w4a8_moe_deepep_normal

    def _deepep_ll_kernel(self):
        from sglang.srt.layers.moe.musa_w4a8_moe import (
            musa_w4a8_moe_deepep_ll,
        )

        return musa_w4a8_moe_deepep_ll


def create_w4a8_moe_backend(*, group_size: int, is_musa: bool) -> W4A8MoEBackend:
    backend_cls = MusaW4A8Backend if is_musa else W4A8MoEBackend
    return backend_cls(group_size=group_size)
