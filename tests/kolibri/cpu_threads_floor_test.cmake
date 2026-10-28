# T16 (plan 2026-10-28-kolibri-r9700-speed.md): --cpu-threads 1 is not a valid pool.  One lane has nobody to wait
# for at the fan-out barrier and the decode degenerates into the 51 51 51 ... repeat loop (measured on the R9700
# 2026-10-28).  The engine must clamp 1 -> 2 with a warning that says so, and do it BEFORE the model is read: the
# model here does not exist, so reaching the load failure is what proves the clamp ran instead of exiting early.
#
# 0 keeps its meaning (no pool, the opt-in absence) and 2 must not warn - a clamp that fires on a valid count would
# make the warning noise and hide the real case.
#
# Real-model evidence (the ids of a clamped run equal the ids of an explicit --cpu-threads 2 run, bitwise) needs the
# 45 GB GGUF and a card, so it lives in the session record (docs/AMD_R9700_PORT.md), not here.

execute_process(COMMAND ${ENGINE} --model no-such.gguf --cpu-threads 1 ENCODING UTF-8
                OUTPUT_VARIABLE out1 ERROR_VARIABLE err1 RESULT_VARIABLE rc1 TIMEOUT 60)
if(NOT err1 MATCHES "--cpu-threads 1 is not a valid pool")
  message(FATAL_ERROR "no clamp warning for --cpu-threads 1 (rc=${rc1}); stdout: ${out1}\nstderr: ${err1}")
endif()
if(NOT err1 MATCHES "using 2")
  message(FATAL_ERROR "the warning does not name the clamped count; stderr: ${err1}")
endif()
if(rc1 EQUAL 0)
  message(FATAL_ERROR "--cpu-threads 1 with a missing model ran to completion (rc=0); stderr: ${err1}")
endif()
if(NOT err1 MATCHES "no-such.gguf")
  message(FATAL_ERROR "the run did not reach the model load - the clamp must not exit the process; stderr: ${err1}")
endif()

execute_process(COMMAND ${ENGINE} --model no-such.gguf --cpu-threads 2 ENCODING UTF-8
                OUTPUT_VARIABLE out2 ERROR_VARIABLE err2 RESULT_VARIABLE rc2 TIMEOUT 60)
if(err2 MATCHES "is not a valid pool")
  message(FATAL_ERROR "--cpu-threads 2 wrongly warned about the lane floor; stderr: ${err2}")
endif()
if(NOT err2 MATCHES "no-such.gguf")
  message(FATAL_ERROR "--cpu-threads 2 did not reach the model load; stderr: ${err2}")
endif()

execute_process(COMMAND ${ENGINE} --model no-such.gguf ENCODING UTF-8
                OUTPUT_VARIABLE out3 ERROR_VARIABLE err3 RESULT_VARIABLE rc3 TIMEOUT 60)
if(err3 MATCHES "is not a valid pool")
  message(FATAL_ERROR "the pool-off default (no --cpu-threads) wrongly warned about the lane floor; stderr: ${err3}")
endif()

message("kolibri_cpu_threads_floor: 1 clamps to 2 with a warning (rc=${rc1}), 2 and the default are silent")
