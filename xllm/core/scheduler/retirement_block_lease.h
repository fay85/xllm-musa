/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
==============================================================================*/
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/block/block.h"
#include "core/framework/block/block_manager.h"

namespace xllm {

class RetirementBlockLease final {
 public:
  ~RetirementBlockLease() { clear(); }

  void retain(const Slice<Block>& blocks) {
    blocks_.reserve(blocks_.size() + blocks.size());
    for (const Block& block : blocks) {
      if (block.is_valid()) {
        blocks_.emplace_back(block);
      }
    }
  }

  void clear() {
    std::vector<Block> release;
    release.reserve(1);
    for (Block& block : blocks_) {
      if (!block.is_valid()) {
        continue;
      }
      BlockManager* manager = block.manager();
      if (manager == nullptr) {
        block = Block();
        continue;
      }
      release.emplace_back(std::move(block));
      manager->deallocate(std::move(release));
      CHECK(release.empty());
    }
    blocks_.clear();
  }

  int64_t size() const { return static_cast<int64_t>(blocks_.size()); }

 private:
  std::vector<Block> blocks_;
};

}  // namespace xllm
