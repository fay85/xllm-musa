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
# ==============================================================================

from __future__ import annotations

import json
import math
import threading
import uuid
import weakref
from pathlib import Path
from types import TracebackType
from typing import Sequence

from xllm_export import (
    DiTGenerationOutput,
    DiTMaster,
    DiTRequestKind,
    DiTRequestOutput,
    DiTRequestParams,
    DiTTensorSources,
    Options,
    StatusCode,
    get_device_count,
)

from xllm.pybind.utils import get_free_port

# The model also fixes the output modality.
_REQUEST_KINDS = {
    "QwenImageEditPlusPipeline": DiTRequestKind.IMAGE,
    "MiniMaxH3ModularPipeline": DiTRequestKind.VIDEO,
}


class DiT:
    _instance_lock = threading.Lock()
    _active: weakref.ReferenceType[DiTMaster] | None = None

    def __init__(
        self,
        model: str,
        *,
        sp_size: int = 1,
        text_encoder_tp_size: int = 1,
        tp_size: int = 1,
        cfg_size: int = 1,
        dit_vae_image_size: int = 1048576,
        vae_size: int = 1,
        npu_kernel_backend: str = "AUTO",
        max_tokens_per_batch: int = 10240,
        output_shm_size: int = 128,
        dit_cache_policy: str = "None",
        dit_enable_text_encoder: bool = True,
        dit_h3_task_type: str = "fl2va",
        dit_laser_attention_enabled: bool = False,
        enable_manual_loader: bool = False,
        enable_rolling_load: bool = False,
        rolling_load_num_cached_layers: int = 2,
        rolling_load_num_rolling_slots: int = -1,
    ) -> None:
        self.master: DiTMaster | None = None
        self._lock = threading.Lock()
        model_path = Path(model).resolve()
        with (model_path / "model_index.json").open() as source:
            model_index = json.load(source)
        self._request_kind = _REQUEST_KINDS.get(model_index.get("_class_name"))
        if self._request_kind is None:
            raise ValueError("DiT supports QwenImageEditPlusPipeline and MiniMaxH3ModularPipeline only")
        with DiT._instance_lock:
            active = DiT._active() if DiT._active is not None else None
            if active is not None:
                raise ValueError("Only one active DiT instance is supported")
            options = Options()
            options.model_path = str(model_path)
            options.backend = "dit"
            options.enable_offline_inference = True
            options.tp_size = tp_size
            options.sp_size = sp_size
            options.cfg_size = cfg_size
            options.text_encoder_tp_size = text_encoder_tp_size
            options.max_requests_per_batch = 1
            options.communication_backend = "hccl"
            options.master_node_addr = f"127.0.0.1:{get_free_port()}"
            options.spawn_worker_path = str(Path(__file__).resolve().parent.parent)
            options.enable_shm = True
            options.is_local = True
            options.enable_prefix_cache = False
            options.enable_chunked_prefill = False
            options.enable_schedule_overlap = False
            options.vae_size = vae_size
            options.npu_kernel_backend = npu_kernel_backend
            options.max_tokens_per_batch = max_tokens_per_batch
            options.output_shm_size = output_shm_size
            options.dit_vae_image_size = dit_vae_image_size
            options.dit_cache_policy = dit_cache_policy
            options.dit_enable_text_encoder = dit_enable_text_encoder
            options.dit_h3_task_type = dit_h3_task_type
            options.dit_laser_attention_enabled = dit_laser_attention_enabled
            options.enable_manual_loader = enable_manual_loader or enable_rolling_load
            options.enable_rolling_load = enable_rolling_load
            options.rolling_load_num_cached_layers = rolling_load_num_cached_layers
            options.rolling_load_num_rolling_slots = rolling_load_num_rolling_slots
            self._validate_options(options)
            self.master = DiTMaster(options)
            DiT._active = weakref.ref(self.master)

    @staticmethod
    def _validate_options(options: Options) -> None:
        device_count = get_device_count()
        if (
            device_count < 1
            or options.tp_size < 1
            or options.sp_size < 1
            or options.cfg_size not in (1, 2)
            or options.tp_size * options.sp_size * options.cfg_size != device_count
            or options.text_encoder_tp_size < 1
            or device_count % options.text_encoder_tp_size != 0
            or options.vae_size < 1
            or device_count % options.vae_size != 0
        ):
            raise ValueError(
                f"Invalid DiT topology: visible={device_count}, tp={options.tp_size}, "
                f"sp={options.sp_size}, cfg={options.cfg_size}, "
                f"text_encoder_tp={options.text_encoder_tp_size}, vae={options.vae_size}"
            )
        if options.dit_vae_image_size <= 0:
            raise ValueError("dit_vae_image_size must be positive")
        if options.enable_rolling_load:
            if options.rolling_load_num_cached_layers < 1:
                raise ValueError("rolling_load_num_cached_layers must be >= 1")
            if not -1 <= options.rolling_load_num_rolling_slots <= options.rolling_load_num_cached_layers:
                raise ValueError("rolling_load_num_rolling_slots must be between -1 and cached layers")

    def generate(
        self,
        prompt: str,
        multimedia: Sequence[tuple[str, str]] = (),
        *,
        width: int,
        height: int,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 1.0,
        max_sequence_length: int = 2048,
        seed: int = 0,
        num_frames: int | None = None,
        fps: float = 24.0,
        audio_sampling_rate: int | None = None,
        tensor_sources: DiTTensorSources | None = None,
    ) -> DiTGenerationOutput:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent DiT operations are not supported")
        try:
            master = self.master
            if master is None:
                raise RuntimeError("DiT is closed")
            is_video = self._request_kind == DiTRequestKind.VIDEO
            params = DiTRequestParams(self._request_kind)
            params.request_id = f"{'video' if is_video else 'image'}gen-{uuid.uuid4().hex}"
            params.input_params.prompt = prompt
            params.input_params.negative_prompt = negative_prompt
            if tensor_sources is not None:
                params.input_params.tensor_sources = tensor_sources
            generation_params = params.generation_params
            generation_params.width = width
            generation_params.height = height
            generation_params.num_inference_steps = num_inference_steps
            generation_params.guidance_scale = guidance_scale
            generation_params.true_cfg_scale = true_cfg_scale
            generation_params.max_sequence_length = max_sequence_length
            generation_params.num_images_per_prompt = 1
            generation_params.seed = seed
            generation_params.seed_is_set = True
            if is_video:
                # None means not provided; H3 still checks duration and sampling rate.
                if num_frames is None or audio_sampling_rate is None:
                    raise ValueError("Video requests require num_frames and audio_sampling_rate")
                if min(num_frames, audio_sampling_rate) <= 0 or not 0 < fps < math.inf:
                    raise ValueError("num_frames, audio_sampling_rate and fps must be positive")
                generation_params.num_frames = num_frames
                generation_params.video_fps = fps
                generation_params.audio_sampling_rate = audio_sampling_rate
            names = {"image": "image", "video": "prompt_video", "audio": "prompt_audio"}
            for modality, path in multimedia:
                if modality not in names:
                    raise ValueError("multimedia modality must be image, video or audio")
                params.input_params.media_sources.add(
                    names[modality], modality, Path(path).read_bytes(), generation_params.audio_sampling_rate
                )
            return self._generate(master, params)
        finally:
            self._lock.release()

    @staticmethod
    def _generate(master: DiTMaster, params: DiTRequestParams) -> DiTGenerationOutput:
        results: list[DiTRequestOutput] = []

        def _on_output(output: DiTRequestOutput) -> bool:
            results.append(output)
            return True

        master.handle_request(params, _on_output)
        master.generate()
        if len(results) != 1:
            raise RuntimeError("DiT did not complete exactly one request")
        output = results[0]
        status = output.status
        if status is not None and not status.ok:
            if status.code == StatusCode.INVALID_ARGUMENT:
                raise ValueError(status.message)
            raise RuntimeError(status.message)
        if len(output.outputs) != 1:
            raise RuntimeError("DiT did not return exactly one output")
        return output.outputs[0]

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close DiT during generation")
        try:
            with DiT._instance_lock:
                if self.master is not None:
                    self.master = None
        finally:
            self._lock.release()

    def __enter__(self) -> DiT:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()
