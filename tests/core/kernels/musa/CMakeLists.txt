include(cc_test)

cc_test(
  NAME musa_moe_test
  SRCS moe_test.cpp
  DEPS :cuda_kernels :platform :config torch_musa GTest::gtest_main glog::glog
  ENVIRONMENT "FLASHINFER_OPS_PATH=$<TARGET_FILE_DIR:xllm>/musa_ops"
  RUN_SERIAL
)
target_link_libraries(musa_moe_test PRIVATE
  "$<LINK_GROUP:RESCAN,platform,config,util>")

cc_test(
  NAME musa_gdn_test
  SRCS gdn_test.cpp
  DEPS :cuda_kernels :platform :config torch_musa GTest::gtest_main glog::glog
  ENVIRONMENT "FLASHINFER_OPS_PATH=$<TARGET_FILE_DIR:xllm>/musa_ops"
  RUN_SERIAL
)
target_link_libraries(musa_gdn_test PRIVATE
  "$<LINK_GROUP:RESCAN,platform,config,util>")

cc_test(
  NAME musa_ops_bundle_test
  SRCS ops_bundle_test.cpp
  DEPS :cuda_kernels :platform :config torch_musa GTest::gtest_main glog::glog
  ENVIRONMENT "FLASHINFER_OPS_PATH=$<TARGET_FILE_DIR:xllm>/musa_ops"
  CPU_ONLY
)
target_link_libraries(musa_ops_bundle_test PRIVATE
  "$<LINK_GROUP:RESCAN,platform,config,util>")

cc_test(
  NAME musa_rope_test
  SRCS rope_test.cpp
  DEPS :kernels :platform :config torch_musa GTest::gtest_main glog::glog
  RUN_SERIAL
)
target_link_libraries(musa_rope_test PRIVATE
  "$<LINK_GROUP:RESCAN,platform,config,util>")
