/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include "core/kernels/npu/paged_attention_graph_operation.h"

#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/context.h>
#include <atb/operation.h>
#include <glog/logging.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace xllm::kernel::npu {
namespace {

uint32_t tiling_blocks_from_env(const char* name,
                                uint32_t default_value,
                                uint32_t minimum) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  uint32_t blocks = 0;
  const char* end = value + std::strlen(value);
  const auto parsed = std::from_chars(value, end, blocks);
  CHECK(parsed.ec == std::errc() && parsed.ptr == end)
      << "Invalid ATB tiling block count in " << name;
  CHECK_GE(blocks, minimum);
  CHECK_LE(blocks, 1024);
  return blocks;
}

class PaStreamResources final {
 public:
  explicit PaStreamResources(uint32_t device_blocks)
      : device_(c10::DeviceType::PrivateUse1,
                c10_npu::getCurrentNPUStream().device_index()) {
    atb::Context* context = nullptr;
    CHECK_EQ(
        atb::CreateContext(&context,
                           tiling_blocks_from_env(
                               "ATB_HOST_TILING_BUFFER_BLOCK_NUM", 128, 128),
                           device_blocks),
        atb::NO_ERROR);
    context_.reset(context);
  }

  ~PaStreamResources() {
    const c10::DeviceGuard guard(device_);
    context_.reset();
  }

  atb::Context* context() const { return context_.get(); }
  torch::Tensor workspace(uint64_t bytes, const torch::TensorOptions& options) {
    if (bytes == 0) {
      return torch::Tensor();
    }
    if (!workspace_.defined() ||
        static_cast<uint64_t>(workspace_.numel()) < bytes) {
      uint64_t capacity =
          workspace_.defined() ? static_cast<uint64_t>(workspace_.numel()) : 1;
      while (capacity < bytes) {
        CHECK_LE(
            capacity,
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 2);
        capacity *= 2;
      }
      workspace_ = torch::empty({static_cast<int64_t>(capacity)},
                                options.dtype(torch::kUInt8));
      // An older replay can still refer to the previous allocation. Retain
      // growth generations with the graph; doubling bounds their total size.
      workspaces_.emplace_back(workspace_);
    }
    return workspace_;
  }

 private:
  torch::Device device_;
  std::unique_ptr<atb::Context, decltype(&atb::DestroyContext)> context_{
      nullptr,
      atb::DestroyContext};
  torch::Tensor workspace_;
  std::vector<torch::Tensor> workspaces_;
};

atb::Tensor make_atb_tensor(const torch::Tensor& tensor) {
  CHECK(tensor.is_contiguous());
  atb::Tensor result{};
  result.desc.format = ACL_FORMAT_ND;
  if (!tensor.device().is_cpu()) {
    const aclFormat format =
        static_cast<aclFormat>(at_npu::native::get_npu_format(tensor));
    if (format != ACL_FORMAT_NCHW && format != ACL_FORMAT_NHWC &&
        format != ACL_FORMAT_NCDHW) {
      result.desc.format = format;
    }
  }
  switch (tensor.scalar_type()) {
    case torch::kBFloat16:
      result.desc.dtype = ACL_BF16;
      break;
    case torch::kHalf:
      result.desc.dtype = ACL_FLOAT16;
      break;
    case torch::kInt32:
      result.desc.dtype = ACL_INT32;
      break;
    case torch::kInt64:
      result.desc.dtype = ACL_INT64;
      break;
    case torch::kUInt8:
      result.desc.dtype = ACL_UINT8;
      break;
    case torch::kInt8:
      result.desc.dtype = ACL_INT8;
      break;
    case torch::kFloat32:
      result.desc.dtype = ACL_FLOAT;
      break;
    default:
      LOG(FATAL) << "Unsupported PA graph tensor dtype: "
                 << tensor.scalar_type();
  }
  result.desc.shape.dimNum = tensor.dim();
  for (int64_t index = 0; index < tensor.dim(); ++index) {
    result.desc.shape.dims[index] = tensor.size(index);
  }
  if (tensor.device().is_cpu()) {
    result.hostData = tensor.data_ptr();
  } else {
    result.deviceData = tensor.data_ptr();
  }
  result.dataSize = atb::Utils::GetTensorSize(result);
  return result;
}

}  // namespace

class PagedAttentionGraphResources::Impl final {
 public:
  explicit Impl(int64_t num_layers) : num_layers_(num_layers) {
    CHECK_GT(num_layers_, 0);
  }

  std::shared_ptr<PaStreamResources> acquire() {
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    std::lock_guard<std::mutex> lock(mutex_);
    auto& resources = streams_[stream];
    if (resources != nullptr) {
      return resources;
    }
    // Bound the pool budget, not the supported model depth. Keep several
    // tiling launches per layer in flight instead of wrapping the default pool.
    const int64_t layers = std::min<int64_t>(num_layers_, 256);
    const uint32_t budget =
        static_cast<uint32_t>(((4 * layers + 31) / 32) * 32);
    const uint32_t device_blocks = std::max(
        budget,
        tiling_blocks_from_env("ATB_DEVICE_TILING_BUFFER_BLOCK_NUM", 32, 32));
    resources = std::make_shared<PaStreamResources>(device_blocks);
    return resources;
  }

  size_t stream_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return streams_.size();
  }

 private:
  int64_t num_layers_;
  mutable std::mutex mutex_;
  std::unordered_map<aclrtStream, std::shared_ptr<PaStreamResources>> streams_;
};

PagedAttentionGraphResources::PagedAttentionGraphResources(int64_t num_layers)
    : impl_(std::make_unique<Impl>(num_layers)) {}
PagedAttentionGraphResources::~PagedAttentionGraphResources() = default;
size_t PagedAttentionGraphResources::stream_count() const {
  return impl_->stream_count();
}

class PagedAttentionGraphOperation::Impl final {
 public:
  Impl(std::shared_ptr<PaStreamResources> resources,
       torch::Tensor query,
       torch::Tensor key,
       torch::Tensor value,
       torch::Tensor block_table,
       torch::Tensor output,
       double scale)
      : resources_(std::move(resources)),
        query_source_(query.view({-1, query.size(-2), query.size(-1)})),
        query_(
            query_source_.is_contiguous()
                ? query_source_
                : torch::empty(query_source_.sizes(), query_source_.options())),
        key_(std::move(key)),
        value_(std::move(value)),
        block_table_source_(std::move(block_table)),
        block_table_(block_table_source_.is_contiguous()
                         ? block_table_source_
                         : torch::empty(block_table_source_.sizes(),
                                        block_table_source_.options())),
        output_(output.view(query_.sizes())),
        copy_query_(!query_source_.is_contiguous()),
        copy_block_table_(!block_table_source_.is_contiguous()) {
    lengths_ = torch::empty({block_table_.size(0)},
                            torch::TensorOptions().dtype(torch::kInt32));
    atb::infer::PagedAttentionParam param;
    param.headNum = query_.size(1);
    param.kvHeadNum = key_.size(-2);
    param.qkScale = scale;
    param.maskType = atb::infer::PagedAttentionParam::UNDEFINED;
    param.batchRunStatusEnable = false;
    param.quantType = atb::infer::PagedAttentionParam::TYPE_QUANT_UNDEFINED;
    param.outDataType = ACL_DT_UNDEFINED;
    param.hasQuantOffset = false;
    param.compressType =
        atb::infer::PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
    param.calcType = atb::infer::PagedAttentionParam::CALC_TYPE_UNDEFINED;
    param.scaleType = atb::infer::PagedAttentionParam::SCALE_TYPE_TOR;
    param.inputLayout = atb::infer::TYPE_BSND;
    param.mlaVHeadSize = 0;
    atb::Operation* operation = nullptr;
    CHECK_EQ(atb::CreateOperation(param, &operation), atb::NO_ERROR);
    operation_.reset(operation);
    pack_.inTensors = {make_atb_tensor(query_),
                       make_atb_tensor(key_),
                       make_atb_tensor(value_),
                       make_atb_tensor(block_table_),
                       make_atb_tensor(lengths_)};
    pack_.outTensors = {make_atb_tensor(output_)};
  }

  void prepare(const std::vector<int32_t>& kv_seq_lens) {
    CHECK_LE(kv_seq_lens.size(), static_cast<size_t>(lengths_.numel()));
    int32_t* lengths = lengths_.data_ptr<int32_t>();
    std::copy(kv_seq_lens.begin(), kv_seq_lens.end(), lengths);
    std::fill(lengths + kv_seq_lens.size(), lengths + lengths_.numel(), 1);
    CHECK_EQ(resources_->context()->SetExecuteStream(
                 c10_npu::getCurrentNPUStream().stream()),
             atb::NO_ERROR);
    CHECK_EQ(operation_->Setup(pack_, workspace_size_, resources_->context()),
             atb::NO_ERROR);
    workspace_ = resources_->workspace(workspace_size_, query_.options());
  }

  void execute() {
    // Refresh strided sources on every execution with stable buffer addresses.
    // Copies stay in the task group so graph-produced queries are read in order
    // and replay task replacement preserves the copy-to-PA dependency.
    if (copy_query_) {
      query_.copy_(query_source_);
    }
    if (copy_block_table_) {
      block_table_.copy_(block_table_source_);
    }
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    auto launch = [this, stream]() -> int32_t {
      const atb::Status status =
          resources_->context()->SetExecuteStream(stream);
      if (status != atb::NO_ERROR) {
        return status;
      }
      return operation_->Execute(
          pack_,
          workspace_size_ == 0 ? nullptr : workspace_.data_ptr<uint8_t>(),
          workspace_size_,
          resources_->context());
    };
    at_npu::native::OpCommand::RunOpApiV2("PagedAttentionGraph", launch);
  }

  uint64_t workspace_size() const { return workspace_size_; }

 private:
  std::shared_ptr<PaStreamResources> resources_;
  torch::Tensor query_source_;
  torch::Tensor query_;
  torch::Tensor key_;
  torch::Tensor value_;
  torch::Tensor block_table_source_;
  torch::Tensor block_table_;
  torch::Tensor output_;
  torch::Tensor lengths_;
  torch::Tensor workspace_;
  bool copy_query_ = false;
  bool copy_block_table_ = false;
  atb::VariantPack pack_;
  std::unique_ptr<atb::Operation, decltype(&atb::DestroyOperation)> operation_{
      nullptr,
      atb::DestroyOperation};
  uint64_t workspace_size_ = 0;
};

PagedAttentionGraphOperation::PagedAttentionGraphOperation(
    std::shared_ptr<PagedAttentionGraphResources> resources,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor block_table,
    torch::Tensor output,
    double scale) {
  CHECK(resources != nullptr);
  impl_ = std::make_unique<Impl>(resources->impl_->acquire(),
                                 std::move(query),
                                 std::move(key),
                                 std::move(value),
                                 std::move(block_table),
                                 std::move(output),
                                 scale);
}

PagedAttentionGraphOperation::~PagedAttentionGraphOperation() = default;

void PagedAttentionGraphOperation::prepare(
    const std::vector<int32_t>& kv_seq_lens) {
  impl_->prepare(kv_seq_lens);
}

void PagedAttentionGraphOperation::execute() { impl_->execute(); }

uint64_t PagedAttentionGraphOperation::workspace_size() const {
  return impl_->workspace_size();
}

}  // namespace xllm::kernel::npu
