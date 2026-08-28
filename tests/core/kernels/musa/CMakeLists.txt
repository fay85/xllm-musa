include(cc_test)

cc_test(
  NAME
    musa_topk_test
  SRCS
    musa_topk_test.cpp
  DEPS
    :cuda_kernels
    torch
    torch_musa
    musa
    musart
    GTest::gtest_main
    glog::glog
)
