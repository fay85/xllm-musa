include(cc_test)

cc_test(
  NAME
    musa_device_test
  SRCS
    device_test.cpp
  DEPS
    :platform
    :config
    torch_musa
    GTest::gtest_main
    glog::glog
)
target_link_libraries(musa_device_test PRIVATE
  "$<LINK_GROUP:RESCAN,platform,config,util>")
