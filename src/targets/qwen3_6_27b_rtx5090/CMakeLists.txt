target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/diagnostic/activation_dump_access.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/program/layouts.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/program/program.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/program/request_plan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/schedule/mtp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/schedule/text_context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/schedule/text_prefill.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/schedule/vision.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/schedule/vision_context.cpp)

target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/export)
