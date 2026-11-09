# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0

set(environment "--unset=${VARIABLE}")
if(NOT VARIABLE_VALUE STREQUAL "__UNSET__")
  list(APPEND environment "${VARIABLE}=${VARIABLE_VALUE}")
endif()
if(BINARY_ARGUMENT STREQUAL "__NONE__")
  execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env ${environment}
          "${BASH_EXECUTABLE}" -c "ulimit -c 0; exec \"$@\"" mtlink "${BINARY}"
      RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
      TIMEOUT 10)
else()
  # Keep an intentionally empty policy argument as a real argv element.
  execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env ${environment}
          "${BASH_EXECUTABLE}" -c "ulimit -c 0; exec \"$@\"" mtlink
          "${BINARY}" "${BINARY_ARGUMENT}"
      RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
      TIMEOUT 10)
endif()

if(EXPECTED STREQUAL "__FATAL__")
  if(result STREQUAL "0" OR NOT error MATCHES "must be exactly 0 or 1")
    message(FATAL_ERROR "Expected strict policy rejection; result=${result}\n${output}\n${error}")
  endif()
elseif(NOT result STREQUAL "0")
  message(FATAL_ERROR "CPU test failed; result=${result}\n${output}\n${error}")
else()
  string(FIND "${output}" "${EXPECTED}" match)
  if(match EQUAL -1)
    message(FATAL_ERROR "Missing expected output '${EXPECTED}': ${output}")
  endif()
endif()
