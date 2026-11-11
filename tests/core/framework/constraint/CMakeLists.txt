include(cc_test)

cc_test(
  NAME
    state_machine_test
  CPU_ONLY
  SRCS
    state_machine_test.cpp
  DEPS
    :constraint
    GTest::gtest_main
)

if(USE_XGRAMMAR)
  cc_test(
    NAME
      xgrammar_machine_test
    CPU_ONLY
    SRCS
      xgrammar_machine_test.cpp
    DEPS
      :constraint
      :xgrammar_constraint
      GTest::gtest_main
  )
endif()
