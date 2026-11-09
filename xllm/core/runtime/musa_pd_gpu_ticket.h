/* Copyright 2026 The xLLM Authors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <musa.h>

#include <array>
#include <cstdint>
#include <memory>

#include "core/runtime/pd_gpu_ticket.h"

namespace xllm::detail {

class MusaPdGpuTicket final : public PdGpuTicket {
 public:
  static std::shared_ptr<PdGpuTicket> record(uint64_t epoch,
                                             MUstream compute,
                                             MUstream prepare,
                                             bool eligible = true);
  ~MusaPdGpuTicket() override;
  MusaPdGpuTicket(const MusaPdGpuTicket&) = delete;
  MusaPdGpuTicket& operator=(const MusaPdGpuTicket&) = delete;
  MusaPdGpuTicket(MusaPdGpuTicket&&) = delete;
  MusaPdGpuTicket& operator=(MusaPdGpuTicket&&) = delete;
  PdGpuTicketObservation query() const override;

 private:
  MusaPdGpuTicket(uint64_t epoch, MUstream compute, MUstream prepare);
  PdGpuTicketObservation observation_;
  MUcontext context_ = nullptr;
  std::array<MUevent, 2> events_ = {nullptr, nullptr};
  std::array<uint32_t, 2> role_masks_ = {0, 0};
  uint32_t event_count_ = 0;
};

}  // namespace xllm::detail
