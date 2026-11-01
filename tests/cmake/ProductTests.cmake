ninfer_add_test(ninfer_media_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_media_decode.cpp"
  LIBRARIES ninfer_media_decode)

ninfer_add_test(ninfer_prompt_input_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prompt_input.cpp"
  LIBRARIES ninfer_product_prompt_input)

ninfer_add_test(ninfer_pretty_logging_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_pretty_logging.cpp"
  LIBRARIES ninfer_product_logging)

ninfer_add_test(ninfer_perplexity_evaluation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_perplexity_evaluation.cpp"
          ${PROJECT_SOURCE_DIR}/apps/perplexity/evaluation.cpp
  LIBRARIES ninfer_core)

target_include_directories(ninfer_perplexity_evaluation_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/perplexity)

ninfer_add_test(ninfer_perplexity_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_perplexity_options.cpp"
          ${PROJECT_SOURCE_DIR}/apps/perplexity/options.cpp
  LIBRARIES ninfer_runtime_support ninfer_product_logging)

target_include_directories(ninfer_perplexity_options_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/perplexity)

ninfer_add_test(ninfer_cli_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_cli_options.cpp" ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp
  LIBRARIES ninfer_runtime_support ninfer_product_logging)

target_include_directories(ninfer_cli_options_test PRIVATE ${PROJECT_SOURCE_DIR}/apps/cli)

ninfer_add_test(ninfer_openai_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_schema.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_openai_responses_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_openai_responses_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses_store.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_anthropic_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_anthropic_schema.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_turn_anchors_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_turn_anchors.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_serve_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_serve_options.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_reasoning_retry_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_reasoning_retry.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_request_log_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_request_log.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_prefill_miss_dump_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prefill_miss_dump.cpp"
  LIBRARIES ninfer_serve)

# Real NVML on the local GPUs; skips without CUDA devices or libnvidia-ml.so.1.
ninfer_add_test(ninfer_gpu_telemetry_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gpu_telemetry_real.cpp"
  LIBRARIES ninfer_serve CUDA::cudart)
set_tests_properties(ninfer_gpu_telemetry_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_http_error_handler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_error_handler.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_http_transport_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_transport.cpp"
  LIBRARIES ninfer_serve)

# Real artifact, two devices: an injected CUDA error inside the Engine worker must stop listen()
# and mark the server as failed by the engine. Link-time wrapper, no production hooks.
ninfer_add_test(ninfer_serve_engine_failure_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_serve_engine_failure_real.cpp"
  LIBRARIES ninfer_serve ninfer_product_logging CUDA::cudart)
target_link_options(ninfer_serve_engine_failure_real_test PRIVATE
  "LINKER:--wrap=cudaStreamSynchronize")
set_tests_properties(ninfer_serve_engine_failure_real_test PROPERTIES SKIP_RETURN_CODE 77)

# Real artifact, two devices: a captured exchange that times out while serving fails only its
# request; the transport steps down and the server keeps serving with the same tokens, and a
# step-down that fails fails the Engine without telling the request it keeps serving. Link-time
# wrappers around PeerMailbox::hang_reported() and Program::recover_peer_transport(), no production
# hooks.
ninfer_add_test(ninfer_serve_tp_transport_recovery_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_serve_tp_transport_recovery_real.cpp"
  LIBRARIES ninfer_serve ninfer_product_logging CUDA::cudart)
target_link_options(ninfer_serve_tp_transport_recovery_real_test PRIVATE
  "LINKER:--wrap=_ZNK6ninfer3ops11PeerMailbox13hang_reportedEv"
  "LINKER:--wrap=_ZN6ninfer6models7qwen3_57Program22recover_peer_transportB5cxx11Ev")
set_tests_properties(ninfer_serve_tp_transport_recovery_real_test PROPERTIES SKIP_RETURN_CODE 77)

# Documentation guard: every serve flag, NINFER_* variable, conversion recipe and Tnn tag of the
# sources appears in the user docs (tools/check_docs.py).
add_test(NAME ninfer_docs_guard_test
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tools/check_docs.py ${PROJECT_SOURCE_DIR})
