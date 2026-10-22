/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "framework/kv_cache/musa/deepseek_v4_kv_cache_impl.h"

#include <utility>
#include <vector>

namespace xllm {
namespace {

torch::Tensor swap_tensor_blocks(const torch::Tensor& tensor,
                                 const torch::Tensor& src,
                                 const torch::Tensor& dst) {
  if (!tensor.defined()) {
    return tensor;
  }
  torch::Tensor selected = torch::index_select(tensor, /*dim=*/0, src);
  torch::Tensor result = tensor.clone();
  result.index_copy_(/*dim=*/0, dst, selected);
  return result;
}

class MusaDeepSeekV4KVCacheImpl final : public KVCacheImpl {
 public:
  explicit MusaDeepSeekV4KVCacheImpl(const DeepSeekV4KVCacheTensors& tensors)
      : key_cache_(tensors.key_cache),
        mate_key_cache_(tensors.mate_key_cache),
        index_cache_(tensors.index_cache),
        indexer_cache_scale_(tensors.indexer_cache_scale),
        swa_cache_(tensors.swa_cache),
        mate_extra_k_cache_(tensors.mate_extra_k_cache),
        compress_kv_state_(tensors.compress_kv_state),
        compress_score_state_(tensors.compress_score_state),
        compress_index_kv_state_(tensors.compress_index_kv_state),
        compress_index_score_state_(tensors.compress_index_score_state),
        compressed_block_type_(tensors.compressed_block_type) {}

  torch::Tensor get_k_cache() const override { return key_cache_; }
  torch::Tensor get_index_cache() const override { return index_cache_; }
  std::optional<torch::Tensor> get_indexer_cache_scale() const override {
    if (indexer_cache_scale_.defined() && indexer_cache_scale_.numel() > 0) {
      return indexer_cache_scale_;
    }
    return std::nullopt;
  }
  torch::Tensor get_swa_cache() const override { return swa_cache_; }
  torch::Tensor get_compress_kv_state() const override {
    return compress_kv_state_;
  }
  torch::Tensor get_compress_score_state() const override {
    return compress_score_state_;
  }
  torch::Tensor get_compress_index_kv_state() const override {
    return compress_index_kv_state_;
  }
  torch::Tensor get_compress_index_score_state() const override {
    return compress_index_score_state_;
  }

  std::vector<KVCacheTensor> get_cache_tensors() const override {
    std::vector<KVCacheTensor> tensors;
    tensors.reserve(10);
    auto add_tensor = [&tensors](KVCacheTensorRole role,
                                 const torch::Tensor& tensor,
                                 BlockType block_type) {
      if (tensor.defined() && tensor.numel() > 0) {
        tensors.emplace_back(
            KVCacheTensor{role, tensor, cache_group_id(block_type)});
      }
    };

    add_tensor(KVCacheTensorRole::WINDOW, swa_cache_, BlockType::SWA);
    add_tensor(KVCacheTensorRole::VALUE, mate_extra_k_cache_, BlockType::SWA);
    add_tensor(KVCacheTensorRole::KEY, key_cache_, compressed_block_type_);
    add_tensor(
        KVCacheTensorRole::VALUE, mate_key_cache_, compressed_block_type_);
    add_tensor(KVCacheTensorRole::INDEX, index_cache_, compressed_block_type_);
    add_tensor(KVCacheTensorRole::INDEX_SCALE,
               indexer_cache_scale_,
               compressed_block_type_);
    add_tensor(KVCacheTensorRole::KV_STATE, compress_kv_state_, BlockType::SWA);
    add_tensor(
        KVCacheTensorRole::SCORE_STATE, compress_score_state_, BlockType::SWA);
    add_tensor(KVCacheTensorRole::INDEX_KV_STATE,
               compress_index_kv_state_,
               BlockType::SWA);
    add_tensor(KVCacheTensorRole::INDEX_SCORE_STATE,
               compress_index_score_state_,
               BlockType::SWA);
    return tensors;
  }

  BlockTypeTensorMap get_block_type_tensors(BlockType type) const override {
    BlockTypeTensorMap tensors;
    if (type == BlockType::SWA) {
      add_to_map(tensors, KVCacheTensorRole::SWA, swa_cache_);
      add_to_map(tensors, KVCacheTensorRole::VALUE, mate_extra_k_cache_);
    } else if (type == compressed_block_type_) {
      add_to_map(tensors, KVCacheTensorRole::KEY, key_cache_);
      add_to_map(tensors, KVCacheTensorRole::VALUE, mate_key_cache_);
      add_to_map(tensors, KVCacheTensorRole::INDEX, index_cache_);
      add_to_map(tensors, KVCacheTensorRole::INDEX_SCALE, indexer_cache_scale_);
    }
    return tensors;
  }

  bool empty() const override { return !swa_cache_.defined(); }

  std::vector<std::vector<int64_t>> get_shapes() const override {
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(5);
    add_shape(shapes, key_cache_);
    add_shape(shapes, mate_key_cache_);
    add_shape(shapes, index_cache_);
    add_shape(shapes, swa_cache_);
    add_shape(shapes, mate_extra_k_cache_);
    return shapes;
  }

  void swap_blocks(torch::Tensor& src, torch::Tensor& dst) override {
    key_cache_ = swap_tensor_blocks(key_cache_, src, dst);
    mate_key_cache_ = swap_tensor_blocks(mate_key_cache_, src, dst);
    index_cache_ = swap_tensor_blocks(index_cache_, src, dst);
    indexer_cache_scale_ = swap_tensor_blocks(indexer_cache_scale_, src, dst);
    swa_cache_ = swap_tensor_blocks(swa_cache_, src, dst);
    mate_extra_k_cache_ = swap_tensor_blocks(mate_extra_k_cache_, src, dst);
    compress_kv_state_ = swap_tensor_blocks(compress_kv_state_, src, dst);
    compress_score_state_ = swap_tensor_blocks(compress_score_state_, src, dst);
    compress_index_kv_state_ =
        swap_tensor_blocks(compress_index_kv_state_, src, dst);
    compress_index_score_state_ =
        swap_tensor_blocks(compress_index_score_state_, src, dst);
  }

 private:
  static void add_to_map(BlockTypeTensorMap& tensors,
                         KVCacheTensorRole role,
                         const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() > 0) {
      tensors.emplace(role, tensor);
    }
  }

  static void add_shape(std::vector<std::vector<int64_t>>& shapes,
                        const torch::Tensor& tensor) {
    if (tensor.defined()) {
      shapes.emplace_back(tensor.sizes().vec());
    }
  }

  torch::Tensor key_cache_;
  torch::Tensor mate_key_cache_;
  torch::Tensor index_cache_;
  torch::Tensor indexer_cache_scale_;
  torch::Tensor swa_cache_;
  torch::Tensor mate_extra_k_cache_;
  torch::Tensor compress_kv_state_;
  torch::Tensor compress_score_state_;
  torch::Tensor compress_index_kv_state_;
  torch::Tensor compress_index_score_state_;
  BlockType compressed_block_type_ = BlockType::KV;
};

}  // namespace

int64_t musa_dsv4_flashmla_cache_head_dim(int64_t model_head_dim) {
  CHECK_GT(model_head_dim, 0);
  return model_head_dim + kMusaDsv4FlashMlaRopeHeadDim;
}

int64_t musa_dsv4_model1_page_stride(int64_t block_size) {
  CHECK_GT(block_size, 0);
  const int64_t page_bytes = block_size * kMusaDsv4Model1BytesPerToken;
  return ((page_bytes + kMusaDsv4Model1PageAlignment - 1) /
          kMusaDsv4Model1PageAlignment) *
         kMusaDsv4Model1PageAlignment;
}

torch::Tensor create_musa_dsv4_model1_cache(int64_t block_count,
                                            int64_t block_size,
                                            const torch::Device& device) {
  CHECK_GE(block_count, 0);
  const int64_t page_stride = musa_dsv4_model1_page_stride(block_size);
  torch::Tensor storage =
      torch::empty({block_count * page_stride},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));
  return storage.as_strided(
      {block_count, block_size, 1, kMusaDsv4Model1BytesPerToken},
      {page_stride,
       kMusaDsv4Model1BytesPerToken,
       kMusaDsv4Model1BytesPerToken,
       1});
}

std::unique_ptr<KVCacheImpl> create_musa_dsv4_kv_cache_impl(
    const DeepSeekV4KVCacheTensors& tensors) {
  return std::make_unique<MusaDeepSeekV4KVCacheImpl>(tensors);
}

}  // namespace xllm
