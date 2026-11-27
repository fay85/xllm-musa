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

"""VAE image pre/post-processing.

Port of ``xllm/models/dit/processors/vae_image_processor.h``. Pure tensor
operations; the lanczos resize delegates to PIL, whose ``Resample.c`` the C++
``lanczos_resample`` implementation was itself ported from, so both stacks
produce identical uint8 output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_PIL_MODES = {1: "L", 3: "RGB", 4: "RGBA"}


class VAEImageProcessor(nn.Module):
    """Resize / normalize / denormalize images around the VAE."""

    def __init__(
        self,
        config: dict,
        do_resize: bool = True,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
        latent_channels: int = 4,
        scale_factor: int | None = None,
    ) -> None:
        super().__init__()
        block_out_channels = config.get("block_out_channels")
        default_scale = 1 << len(block_out_channels) if block_out_channels else 8
        self.scale_factor = scale_factor if scale_factor is not None else default_scale
        self.latent_channels = latent_channels
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.do_binarize = do_binarize
        self.do_convert_rgb = do_convert_rgb
        self.do_convert_grayscale = do_convert_grayscale

    def adjust_dimensions(self, height: int, width: int) -> tuple[int, int]:
        height = height - (height % self.scale_factor)
        width = width - (width % self.scale_factor)
        return height, width

    def preprocess(
        self,
        image: torch.Tensor,
        height: int | None = None,
        width: int | None = None,
        resize_mode: str = "lanczos",
        crop_coords: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        processed = image
        if crop_coords is not None:
            x1, y1, x2, y2 = crop_coords
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(processed.size(-1), x2)
            y2 = min(processed.size(-2), y2)
            if processed.dim() == 3:
                processed = processed[:, y1:y2, x1:x2]
            elif processed.dim() == 4:
                processed = processed[:, :, y1:y2, x1:x2]

        channel = processed.size(1)
        if channel == self.latent_channels:
            return image

        target_h, target_w = self._get_default_height_width(processed, height, width)
        if self.do_resize:
            processed = self.resize(processed, target_h, target_w, resize_mode)
        if float(processed.max().item()) > 1.1:
            processed = processed / 255.0
        if self.do_normalize:
            processed = self._normalize(processed)
        if self.do_binarize:
            processed = (processed >= 0.5).to(torch.float32)
        return processed

    def postprocess(
        self,
        tensor: torch.Tensor,
        do_denormalize: list[bool] | None = None,
    ) -> torch.Tensor:
        processed = tensor.clone()
        if self.do_normalize:
            if do_denormalize is None:
                processed = self._denormalize(processed)
            else:
                for index in range(processed.size(0)):
                    if index < len(do_denormalize) and do_denormalize[index]:
                        processed[index] = self._denormalize(processed[index])
        return processed

    def resize(self, image: torch.Tensor, height: int, width: int, resize_mode: str = "lanczos") -> torch.Tensor:
        options = (image.device, image.dtype)
        image = image.cpu()

        squeeze_batch = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_batch = True
        if image.dim() != 4:
            raise ValueError(f"resize expects CHW or BCHW image, got dim={image.dim()}")

        if resize_mode == "lanczos":
            resized = torch.stack([self._resize_lanczos(image[i], height, width) for i in range(image.size(0))])
        elif resize_mode == "bicubic":
            resized = torch.nn.functional.interpolate(
                image.float(),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).to(image.dtype)
        else:
            raise ValueError(f"Currently only support 'lanczos' and 'bicubic', but got: {resize_mode}")

        if squeeze_batch:
            resized = resized.squeeze(0)
        return resized.to(device=options[0], dtype=options[1])

    # ---- internals ----------------------------------------------------------

    @staticmethod
    def _resize_lanczos(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """PIL-lanczos resize of one [C, H, W] uint8 image, kept on CPU.

        The C++ path ports PIL's Resample.c with fixed-point rounding; PIL
        itself is the reference, so routing through it is exact. The byte
        round-trip uses a single numpy memcpy (``numpy().tobytes()``) rather
        than ``bytes(untyped_storage())``, which is O(N) per element.
        """
        from PIL import Image  # local import: keeps package import torch-only

        if image.dtype != torch.uint8:
            # The C++ path requires uint8 (data_ptr<uint8_t>); be graceful
            # with float inputs by rounding into the same domain.
            image = (image.clamp(0.0, 255.0) if float(image.max()) > 1.1 else image.clamp(0.0, 1.0) * 255.0).to(
                torch.uint8
            )
        channels, src_h, src_w = image.shape
        if channels not in _PIL_MODES:
            raise ValueError(f"lanczos resize supports 1/3/4 channels, got {channels}")
        mode = _PIL_MODES[channels]
        # HWC byte order: permute, then materialize a compact storage to read.
        hwc = image.contiguous().permute(1, 2, 0).contiguous()
        # NOTE: bytes(hwc.untyped_storage()) is O(N) per element in CPython and
        # takes ~10s for a 1MP image; numpy().tobytes() is a single memcpy.
        src_bytes = hwc.numpy().tobytes()
        pil_image = Image.frombytes(mode, (src_w, src_h), src_bytes)
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        out = torch.frombuffer(bytearray(pil_image.tobytes()), dtype=torch.uint8)
        return out.reshape(height, width, channels).permute(2, 0, 1)

    def _get_default_height_width(
        self,
        image: torch.Tensor,
        height: int | None = None,
        width: int | None = None,
    ) -> tuple[int, int]:
        if image.dim() == 3:
            h, w = image.size(1), image.size(2)
        elif image.dim() == 4:
            h, w = image.size(2), image.size(3)
        else:
            raise ValueError(f"Unsupported image dimension: {image.dim()}")
        return self.adjust_dimensions(height if height is not None else h, width if width is not None else w)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return 2.0 * tensor - 1.0

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)
