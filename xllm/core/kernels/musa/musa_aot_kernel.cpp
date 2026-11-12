/* Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
==============================================================================*/
#include "core/kernels/musa/musa_aot_kernel.h"

#include <glog/logging.h>
#include <musa.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <utility>

#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"
namespace xllm::kernel::musa {
namespace {

void check_driver(MUresult result, const char* operation) {
  if (result == MUresult::MUSA_SUCCESS) {
    return;
  }
  const char* message = nullptr;
  muGetErrorString(result, &message);
  LOG(FATAL) << operation << " failed: "
             << (message == nullptr ? "unknown MUSA error" : message);
}

class MusaAotKernel final {
 public:
  MusaAotKernel(const std::string& directory, const nlohmann::json& metadata) {
    CHECK_EQ(metadata.at("num_ctas").get<int32_t>(), 1);
    CHECK(metadata.at("cluster_dims") == nlohmann::json({1, 1, 1}));
    const std::string filename = metadata.at("file").get<std::string>();
    CHECK_EQ(std::filesystem::path(filename).filename().string(), filename);
    const std::string path = directory + "/" + filename;
    const std::string symbol = metadata.at("symbol").get<std::string>();
    check_driver(muModuleLoad(&module_, path.c_str()), "muModuleLoad");
    check_driver(muModuleGetFunction(&function_, module_, symbol.c_str()),
                 "muModuleGetFunction");
    threads_ = metadata.at("threads").get<uint32_t>();
    shared_ = metadata.at("shared").get<uint32_t>();
    CHECK_GT(threads_, 0);
    CHECK_LE(threads_, 1024);
  }

  void launch(uint32_t grid_x,
              uint32_t grid_y,
              uint32_t grid_z,
              MUstream stream,
              void** parameters) const {
    check_driver(muLaunchKernel(function_,
                                grid_x,
                                grid_y,
                                grid_z,
                                threads_,
                                1,
                                1,
                                shared_,
                                stream,
                                parameters,
                                nullptr),
                 "muLaunchKernel");
  }

 private:
  // Keep the module alive for the process/context lifetime, as in the
  // existing xLLM native fused-MoE AOT loader. No Python owns these handles.
  MUmodule module_ = nullptr;
  MUfunction function_ = nullptr;
  uint32_t threads_ = 0;
  uint32_t shared_ = 0;
};

const MusaAotKernel& kernel(const torch::Device& device,
                            const std::string& directory,
                            const std::string& key) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::unique_ptr<MusaAotKernel>> cache;
  const std::string identity =
      std::to_string(device.index()) + ":" + directory + ":" + key;
  std::lock_guard<std::mutex> lock(mutex);
  const auto existing = cache.find(identity);
  if (existing != cache.end()) {
    return *existing->second;
  }
  std::ifstream file(directory + "/manifest.json");
  CHECK(file.good()) << "Missing MUSA AOT kernel manifest: " << directory;
  nlohmann::json manifest;
  file >> manifest;
  CHECK_EQ(manifest.at("target").at("arch").get<int32_t>(), 31);
  auto value = std::make_unique<MusaAotKernel>(directory,
                                               manifest.at("kernels").at(key));
  const auto [inserted, success] = cache.emplace(identity, std::move(value));
  CHECK(success);
  return *inserted->second;
}

}  // namespace

void launch_musa_aot_kernel(const torch::Device& device,
                            const std::string& directory,
                            const std::string& key,
                            const std::array<uint32_t, 3>& grid,
                            void** arguments) {
  c10::musa::MUSAGuard guard(device.index());
  MUstream stream = reinterpret_cast<MUstream>(
      c10::musa::getCurrentMUSAStream(device.index()).stream());
  kernel(device, directory, key)
      .launch(grid[0], grid[1], grid[2], stream, arguments);
}
}  // namespace xllm::kernel::musa
