add_library(ninfer_serve STATIC
  http_transport.cpp
  request_validation.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  openai_chat_http.cpp
  openai_common.cpp
  openai_responses_request.cpp
  openai_responses_state.cpp
  openai_responses_response.cpp
  anthropic_messages_request.cpp
  anthropic_messages_response.cpp
  anthropic_messages_http.cpp
  openai_responses_store.cpp
  translate.cpp
  generation_service.cpp
  reasoning_retry.cpp
  turn_anchors.cpp
  prompt_overlap.cpp
  prefill_miss_dump.cpp
  request_events.cpp
  operational_log.cpp
  gpu_telemetry.cpp
  request_log.cpp
  http_server.cpp
  openai_responses_http.cpp
  serve_options.cpp
)
ninfer_internal_includes(ninfer_serve)
target_link_libraries(ninfer_serve
  PUBLIC ninfer_engine Threads::Threads ninfer::json ninfer::httplib
  PRIVATE ninfer_media_acquire ninfer_product_logging ninfer_grammar CUDA::cudart ${CMAKE_DL_LIBS})
