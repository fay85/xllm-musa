include(cc_library)

# Shared structured-output constraint building blocks, moved from
# framework/speculative/mtp_json_object_state.h and
# runtime/json_object_output_rows.h. output_rows.h symbols are unchanged;
# draft_validation.h renamed append_json_draft_step to advance_json_draft_step
# and find_json_accepted_token_mismatches to verify_json_accepted_rows.
set(CONSTRAINT_SRCS builtin_json_machine.cpp)
set(CONSTRAINT_HDRS draft_validation.h
                    output_rows.h
                    snapshot.h
                    state_machine.h
                    builtin_json_machine.h
)
set(CONSTRAINT_DEPS :json_object_grammar glog::glog)
if(USE_XGRAMMAR)
  # XGrammarConstraintMachine constructs matchers, so it needs the compiled
  # backend; without USE_XGRAMMAR the adapter sources stay out of the build.
  list(APPEND CONSTRAINT_SRCS xgrammar_machine.cpp)
  list(APPEND CONSTRAINT_HDRS xgrammar_machine.h)
  list(APPEND CONSTRAINT_DEPS :xgrammar_constraint)
endif()
cc_library(
  NAME
    constraint
  HDRS
    ${CONSTRAINT_HDRS}
  SRCS
    ${CONSTRAINT_SRCS}
  DEPS
    ${CONSTRAINT_DEPS}
)
