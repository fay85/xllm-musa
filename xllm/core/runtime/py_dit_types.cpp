/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/runtime/py_dit_types.h"

#include <glog/logging.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/tokenizer/tokenizer.h"
#include "core/runtime/dit_forward_params.h"

namespace py = pybind11;
namespace xllm {

void bind_dit_generation_params(py::module_& module) {
  if (py::detail::get_type_info(typeid(DiTGenerationParams), false) !=
      nullptr) {
    module.attr("DiTGenerationParams") = py::type::of<DiTGenerationParams>();
    return;
  }
  py::class_<DiTGenerationParams>(module, "DiTGenerationParams")
      .def(py::init<>())
      .def_readwrite("output_type", &DiTGenerationParams::output_type)
      .def_readwrite("width", &DiTGenerationParams::width)
      .def_readwrite("height", &DiTGenerationParams::height)
      .def_readwrite("num_inference_steps",
                     &DiTGenerationParams::num_inference_steps)
      .def_readwrite("true_cfg_scale", &DiTGenerationParams::true_cfg_scale)
      .def_readwrite("guidance_scale", &DiTGenerationParams::guidance_scale)
      .def_readwrite("num_images_per_prompt",
                     &DiTGenerationParams::num_images_per_prompt)
      .def_readwrite("num_videos_per_prompt",
                     &DiTGenerationParams::num_videos_per_prompt)
      .def_readwrite("seed", &DiTGenerationParams::seed)
      .def_readwrite("seed_is_set", &DiTGenerationParams::seed_is_set)
      .def_readwrite("max_sequence_length",
                     &DiTGenerationParams::max_sequence_length)
      .def_readwrite("strength", &DiTGenerationParams::strength)
      .def_readwrite("enable_cfg_renorm",
                     &DiTGenerationParams::enable_cfg_renorm)
      .def_readwrite("cfg_renorm_min", &DiTGenerationParams::cfg_renorm_min)
      .def_readwrite("audio_duration_frames",
                     &DiTGenerationParams::audio_duration_frames)
      .def_readwrite("audio_steps", &DiTGenerationParams::audio_steps)
      .def_readwrite("audio_guidance_method",
                     &DiTGenerationParams::audio_guidance_method)
      .def_readwrite("audio_sampling_rate",
                     &DiTGenerationParams::audio_sampling_rate)
      .def_readwrite("num_frames", &DiTGenerationParams::num_frames)
      .def_readwrite("video_fps", &DiTGenerationParams::video_fps)
      .def_readwrite("guidance_scale_2", &DiTGenerationParams::guidance_scale_2)
      .def_readwrite("seconds", &DiTGenerationParams::seconds)
      .def_readwrite("boundary_ratio", &DiTGenerationParams::boundary_ratio)
      .def_readwrite("flow_shift", &DiTGenerationParams::flow_shift)
      .def_readwrite("max_new_tokens", &DiTGenerationParams::max_new_tokens)
      .def_readwrite("diffusion_steps", &DiTGenerationParams::diffusion_steps)
      .def_readwrite("temperature", &DiTGenerationParams::temperature)
      .def_readwrite("top_k", &DiTGenerationParams::top_k)
      .def_readwrite("top_p", &DiTGenerationParams::top_p)
      .def_readwrite("repetition_penalty",
                     &DiTGenerationParams::repetition_penalty)
      .def_readwrite("lora_adapter_ids",
                     &DiTGenerationParams::lora_adapter_ids);
}

void ensure_xllm_dit_runtime_module() {
  // The caller holds the GIL. No static registration into a live interpreter.
  py::dict modules = py::module_::import("sys").attr("modules");
  const py::str name("xllm_dit_runtime");
  if (modules.contains(name)) {
    return;
  }
  py::object object = py::module_::import("types").attr("ModuleType")(name);
  py::module_ module = py::reinterpret_borrow<py::module_>(object);
  bind_dit_generation_params(module);
  py::class_<DiTForwardInput>(module, "DiTForwardInput")
      .def(py::init<>())
      .def_readwrite("batch_size", &DiTForwardInput::batch_size)
      .def_readwrite("prompts", &DiTForwardInput::prompts)
      .def_readwrite("negative_prompts", &DiTForwardInput::negative_prompts)
      .def_readwrite("generation_params", &DiTForwardInput::generation_params)
      .def_property_readonly("prompt_embeds",
                             [](const DiTForwardInput& input) {
                               return input.tensor_sources.get("prompt_embed");
                             })
      .def_property_readonly(
          "negative_prompt_embeds",
          [](const DiTForwardInput& input) {
            return input.tensor_sources.get("negative_prompt_embed");
          })
      .def_property_readonly("latents", [](const DiTForwardInput& input) {
        return input.tensor_sources.get("latent");
      });
  py::class_<DiTForwardOutput>(module, "DiTForwardOutput")
      .def(py::init<>())
      .def_readwrite("tensors", &DiTForwardOutput::tensors)
      .def_readwrite("audio_tensors", &DiTForwardOutput::audio_tensors);
  // Reuse the already resolved tokenizer; no second tokenizer/config download.
  py::class_<Tokenizer, std::shared_ptr<Tokenizer>>(module, "Tokenizer")
      .def("encode", [](const Tokenizer& tokenizer, const std::string& text) {
        std::vector<int32_t> ids;
        CHECK(tokenizer.encode(text, &ids, /*add_special_tokens=*/false));
        return ids;
      });
  modules[name] = module;
}

}  // namespace xllm
