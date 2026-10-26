# G1 follow-up (plan 2026-10-26-kolibri-pf-partial-f16-guard.md): PF_PARTIAL riding the fp16-cache scratch must
# be refused ONCE at startup, before any model read or CUDA init: nonzero exit + a stderr message that names
# STRATA_KOLIBRI_EXPERT_F16 (the knob that fixes it).  The env vars come from the two CTest cases'
# set_tests_properties ENVIRONMENT: (1) PF_PARTIAL=1 alone, (2) PF_PARTIAL=1 with EXPERT_F16=0 (explicit zero).
#
# RED (before the guard; observed 2026-10-26): the pre-fix binary instead dies inside load ("kolibri: load failed:
# GGUF: cannot open no-such.gguf", rc=1) because the refusal message does not exist - this script's second check
# catches that, and the REAL invalid-argument crash behind it is recorded in the plan's Root cause (bench2,
# kolibri_engine.cu:2584, needs a real GGUF: too heavy for CTest, evidence in session).
execute_process(COMMAND ${ENGINE} --model no-such.gguf ENCODING UTF-8
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc TIMEOUT 60)
if(NOT rc)
  message(FATAL_ERROR "strata-kolibri ran to completion (rc=0) with PF_PARTIAL set and the fp16 cache off;\n"
                      "stderr: ${err}\nstdout: ${out}")
endif()
if(NOT err MATCHES "STRATA_KOLIBRI_EXPERT_F16")
  message(FATAL_ERROR "no refusal naming STRATA_KOLIBRI_EXPERT_F16 (rc=${rc}); the engine failed elsewhere.\n"
                      "stderr: ${err}\nstdout: ${out}")
endif()
message("kolibri_pf_partial_guard: refused at startup with rc=${rc}, message names the knob")