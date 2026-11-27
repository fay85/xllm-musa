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

"""FlowMatchEulerDiscreteScheduler.

Port of ``xllm/models/dit/schedulers/flowmatch_euler_discrete_scheduler.h``
(diffusers-compatible). Constructed from the scheduler component's config
dict; the timestep ramp defaults to ending at ``sigma_min`` (the training
grid's smallest sigma) and ends at ``1/N`` when ``raw_sigmas`` is set (the
Wan2.2 I2V distill path).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from xllm.python.models.dit.dit_component_base import DiTComponentBase


class FlowMatchEulerDiscreteScheduler(DiTComponentBase):
    """Euler scheduler over a flow-match sigma ramp."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.shift = float(config.get("shift", 1))
        shift_terminal = config.get("shift_terminal", -1)
        if shift_terminal is None:
            shift_terminal = -1
        self.shift_terminal = None if float(shift_terminal) == -1 else float(shift_terminal)
        self.use_dynamic_shifting = bool(config.get("use_dynamic_shifting", True))
        self.base_shift = float(config.get("base_shift", 0.5))
        self.max_shift = float(config.get("max_shift", 1.15))
        self.raw_sigmas = bool(config.get("raw_sigmas", False))

        num_train_timesteps = self.num_train_timesteps
        timesteps = torch.arange(num_train_timesteps, 0, -1, dtype=torch.float32)  # [N, N-1, ..., 1]
        sigmas = timesteps / num_train_timesteps
        if not self.use_dynamic_shifting:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.cpu()
        self.sigma_min = float(self.sigmas[-1].item())
        self.sigma_max = float(self.sigmas[0].item())
        self.step_index: int | None = None
        self.begin_index: int | None = None

    # The scheduler carries no parameters; the loading convention just needs
    # the hook to exist.
    def load_weights(self, state_dicts: list) -> None:
        return

    def set_begin_index(self, begin_index: int | None) -> None:
        self.begin_index = begin_index

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        if self.shift_terminal is None:
            raise RuntimeError("shift_terminal is not set")
        one_minus_z = 1.0 - t
        scale_factor = float(one_minus_z[-1].item()) / (1.0 - self.shift_terminal)
        return 1.0 - (one_minus_z / scale_factor)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | str = "cpu",
        sigmas: torch.Tensor | None = None,
        mu: float | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> None:
        if self.use_dynamic_shifting and mu is None:
            raise RuntimeError("mu must be provided when use_dynamic_shifting is true")
        if sigmas is not None and timesteps is not None and sigmas.numel() != timesteps.numel():
            raise RuntimeError("sigmas and timesteps must have the same length")

        num_steps = num_inference_steps
        if num_steps <= 0:
            num_steps = sigmas.numel() if sigmas is not None else timesteps.numel()

        is_timesteps_provided = timesteps is not None
        ts_tensor: torch.Tensor | None = None
        sigmas_tensor: torch.Tensor

        if is_timesteps_provided:
            ts_tensor = timesteps.reshape(-1).to(torch.float32).clone()

        if sigmas is None:
            if ts_tensor is None:
                # The ramp runs from sigma_max down to a lower bound. The
                # standard schedule ends at sigma_min; raw_sigmas ends at 1/N
                # (Wan2.2 I2V distill). Only the endpoint differs.
                sigma_end = 1.0 / num_steps if self.raw_sigmas else self.sigma_min
                start = self.sigma_max * self.num_train_timesteps
                end = sigma_end * self.num_train_timesteps
                ts_tensor = start + (end - start) * torch.arange(num_steps, dtype=torch.float32) / (num_steps - 1)
            sigmas_tensor = ts_tensor / self.num_train_timesteps
        else:
            sigmas_tensor = sigmas.reshape(-1).to(torch.float32).clone()

        if self.use_dynamic_shifting:
            sigmas_tensor = self._time_shift_exponential(mu, 1.0, sigmas_tensor)
        else:
            sigmas_tensor = self.shift * sigmas_tensor / (1.0 + (self.shift - 1.0) * sigmas_tensor)

        if self.shift_terminal is not None:
            sigmas_tensor = self.stretch_shift_to_terminal(sigmas_tensor)

        sigmas_tensor = sigmas_tensor.to(device).to(torch.float32)
        if not is_timesteps_provided:
            ts_tensor = sigmas_tensor * self.num_train_timesteps
        else:
            ts_tensor = ts_tensor.to(device).to(torch.float32)

        zeros = torch.zeros(1, dtype=torch.float32, device=device)
        sigmas_tensor = torch.cat([sigmas_tensor, zeros])

        self.timesteps = ts_tensor
        self.sigmas = sigmas_tensor
        self.step_index = None
        self.begin_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        dt = sigma_next - sigma
        prev_sample = sample.to(torch.float32) + dt * model_output

        self.step_index += 1
        return prev_sample.to(model_output.dtype)

    # ---- internals ----------------------------------------------------------

    def _time_shift_exponential(self, mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
        exp_mu = math.exp(mu)
        return exp_mu / (exp_mu + torch.pow(1.0 / t - 1.0, sigma))

    def _init_step_index(self, timestep: torch.Tensor) -> None:
        if self.begin_index is None:
            self.step_index = self.index_for_timestep(timestep.to(self.timesteps.device))
        else:
            self.step_index = self.begin_index

    def index_for_timestep(
        self,
        timestep: torch.Tensor,
        schedule_timesteps: torch.Tensor | None = None,
    ) -> int:
        sched = schedule_timesteps if schedule_timesteps is not None else self.timesteps
        indices = (sched == timestep).nonzero()
        pos = 1 if indices.size(0) > 1 else 0
        return int(indices[pos, 0].item())
