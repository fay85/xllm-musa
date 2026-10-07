/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {
namespace {

torch::Tensor tracked_cpu_tensor(int32_t* releases) {
  auto storage = std::make_shared<std::vector<float>>(16, 1.0f);
  return torch::from_blob(
      storage->data(), {4, 4},
      [storage, releases](void* /*data*/) { ++*releases; },
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
}

TEST(MusaTvmffiLifetimeTest, OwningWrapperReleasesStorageAfterLastCopy) {
  int32_t releases = 0;
  std::optional<ffi::Tensor> held;
  {
    auto tensor = tracked_cpu_tensor(&releases);
    EXPECT_EQ(tensor.use_count(), 1);
    auto wrapped = to_ffi_tensor(tensor);
    EXPECT_EQ(tensor.use_count(), 2);
    EXPECT_EQ(wrapped.data_ptr(), tensor.data_ptr());
    held = wrapped;
  }
  EXPECT_EQ(releases, 0);
  held.reset();
  EXPECT_EQ(releases, 1);
}

TEST(MusaTvmffiLifetimeTest, BorrowedWrapperOwnsMetadataNotStorage) {
  int32_t releases = 0;
  std::optional<ffi::Tensor> held;
  {
    auto tensor = tracked_cpu_tensor(&releases);
    held = to_ffi_borrowed_tensor(tensor);
    EXPECT_EQ(tensor.use_count(), 1);
    EXPECT_EQ(held->data_ptr(), tensor.data_ptr());
  }
  EXPECT_EQ(releases, 1);
  // Only metadata is accessed after storage expires, never the data pointer.
  EXPECT_EQ(held->shape()[0], 4);
  EXPECT_EQ(held->strides()[1], 1);
  held.reset();
  EXPECT_EQ(releases, 1);
}

TEST(MusaTvmffiLifetimeTest, RepeatedStridedWrappingDoesNotAccumulateOwners) {
  auto tensor = torch::arange(32, torch::kFloat32)
                    .reshape({4, 8})
                    .transpose(/*dim0=*/0, /*dim1=*/1);
  const size_t original_refs = tensor.use_count();
  for (int32_t iteration = 0; iteration < 10000; ++iteration) {
    {
      auto owning = to_ffi_tensor(tensor);
      auto borrowed = to_ffi_borrowed_tensor(tensor);
      auto view = to_ffi_tensor_view(tensor);
      EXPECT_EQ(owning.shape()[0], 8);
      EXPECT_EQ(borrowed.strides()[1], 8);
      EXPECT_EQ(view.data_ptr(), tensor.data_ptr());
      EXPECT_EQ(tensor.use_count(), original_refs + 1);
    }
    ASSERT_EQ(tensor.use_count(), original_refs);
  }
}

TEST(MusaTvmffiLifetimeTest, CpuWrapperHeapUsageIsBoundedAfterWarmup) {
#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 33)
  auto tensor = torch::zeros({4, 8}, torch::kFloat32);
  const auto wrap_many = [&tensor](int32_t repetitions) {
    for (int32_t iteration = 0; iteration < repetitions; ++iteration) {
      auto owning = to_ffi_tensor(tensor);
      auto borrowed = to_ffi_borrowed_tensor(tensor);
      EXPECT_EQ(owning.data_ptr(), borrowed.data_ptr());
    }
  };
  wrap_many(/*repetitions=*/1000);
  const struct mallinfo2 before = mallinfo2();
  wrap_many(/*repetitions=*/100000);
  const struct mallinfo2 after = mallinfo2();
  constexpr size_t kAllowanceBytes = 1024 * 1024;
  EXPECT_LE(after.uordblks + after.hblkhd,
            before.uordblks + before.hblkhd + kAllowanceBytes);
#else
  GTEST_SKIP() << "The allocator accounting check requires glibc >= 2.33";
#endif
}

TEST(MusaTvmffiLifetimeTest, PackedCallReleasesNestedTensorReferences) {
  auto tensor = torch::zeros({4, 8}, torch::kFloat32);
  auto first = to_ffi_borrowed_tensor(tensor);
  auto second = to_ffi_borrowed_tensor(tensor);
  auto consume = ffi::Function::FromTyped(
      [](ffi::Tuple<ffi::Tensor, ffi::Tensor> pair) -> int64_t {
        return pair.get<0>().numel() + pair.get<1>().numel();
      });
  for (int32_t iteration = 0; iteration < 1000; ++iteration) {
#if defined(XLLM_TEST_LEGACY_STL_PACKING)
    // Diagnostic A/B build only: reproduces the old call-site ownership gap.
    const auto result = consume(std::make_tuple(first, second));
#else
    const auto result = consume(ffi::Tuple<ffi::Tensor, ffi::Tensor>(first, second));
#endif
    ASSERT_EQ(result.cast<int64_t>(), 64);
    ASSERT_EQ(first.use_count(), 1);
    ASSERT_EQ(second.use_count(), 1);
  }
}

TEST(MusaTvmffiLifetimeTest, PackedShapeTupleHeapUsageIsBounded) {
#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 33)
  auto consume = ffi::Function::FromTyped(
      [](ffi::Tuple<int64_t, int64_t, int64_t> shape) -> int64_t {
        return shape.get<0>() * shape.get<1>() * shape.get<2>();
      });
  const auto call_many = [&consume](int32_t count) {
    for (int32_t iteration = 0; iteration < count; ++iteration) {
#if defined(XLLM_TEST_LEGACY_STL_PACKING)
      const auto result = consume(std::make_tuple(int64_t{1}, int64_t{128}, int64_t{128}));
#else
      const auto result = consume(ffi::Tuple<int64_t, int64_t, int64_t>(
          int64_t{1}, int64_t{128}, int64_t{128}));
#endif
      EXPECT_EQ(result.cast<int64_t>(), 16384);
    }
  };
  call_many(/*count=*/1000);
  const struct mallinfo2 before = mallinfo2();
  call_many(/*count=*/100000);
  const struct mallinfo2 after = mallinfo2();
  EXPECT_LE(after.uordblks + after.hblkhd,
            before.uordblks + before.hblkhd + 1024 * 1024);
#else
  GTEST_SKIP() << "The allocator accounting check requires glibc >= 2.33";
#endif
}

}  // namespace
}  // namespace xllm::kernel::musa
