# --- Core library: L0 infra + L1 kernels + L2 model + runtime ----------------
file(GLOB_RECURSE QUS_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp     ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp    ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/*.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)

add_library(qus_core STATIC ${QUS_SOURCES})
# Resolve device symbols inside the archive so non-CUDA consumers (e.g. the
# qus_serve static lib) can link qus_core without a device-link step of their own.
set_target_properties(qus_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(qus_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
# Internal L1 layer headers resolve as "kernels/launcher/<op>.h" (wrapper -> launcher seam).
target_include_directories(qus_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
target_link_libraries(qus_core PUBLIC CUDA::cudart)
target_compile_options(qus_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# --- Driver executable (text in -> text out) ---------------------------------
add_executable(qus ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(qus PRIVATE qus_core)

# --- Serving layer: OpenAI-compatible HTTP server ----------------------------
# qus_serve holds the protocol/translation/service/transport code (no main), so
# it can be linked by both the qus-serve executable and the schema unit test.
add_library(qus_serve STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/openai_schema.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/translate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/generation_service.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/request_log.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/http_server.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/serve_options.cpp)
target_include_directories(qus_serve PUBLIC
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(qus_serve PUBLIC qus_core Threads::Threads)

add_executable(qus-serve ${CMAKE_CURRENT_SOURCE_DIR}/serve/main.cpp)
target_link_libraries(qus-serve PRIVATE qus_serve)
