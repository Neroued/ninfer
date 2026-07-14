# --- Core library: L0 infra + L1 kernels + L2 model + runtime ----------------
file(GLOB_RECURSE NINFER_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp     ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp    ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/media/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/*.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)

add_library(ninfer_core STATIC ${NINFER_SOURCES})
# Resolve device symbols inside the archive so non-CUDA consumers (e.g. the
# ninfer_serve static lib) can link ninfer_core without a device-link step of their own.
set_target_properties(ninfer_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(ninfer_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
# Internal L1 layer headers resolve as "kernels/launcher/<op>.h" (wrapper -> launcher seam).
target_include_directories(ninfer_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
target_link_libraries(ninfer_core PUBLIC CUDA::cudart)
target_link_libraries(ninfer_core PRIVATE PkgConfig::FFMPEG PkgConfig::LIBCURL)
target_compile_options(ninfer_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# --- Driver executable (text/image/video in -> text out) ---------------------
add_executable(ninfer ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(ninfer PRIVATE ninfer_core)

add_executable(ninfer-preprocess ${PROJECT_SOURCE_DIR}/tools/q5090/preprocess.cpp)
target_include_directories(ninfer-preprocess PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer-preprocess PRIVATE ninfer_core)

add_executable(ninfer-vision-dump ${PROJECT_SOURCE_DIR}/tools/q5090/vision_dump.cpp)
target_include_directories(ninfer-vision-dump PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer-vision-dump PRIVATE ninfer_core)

# --- Serving layer: OpenAI-compatible HTTP server ----------------------------
# ninfer_serve holds the protocol/translation/service/transport code (no main), so
# it can be linked by both the ninfer-serve executable and the schema unit test.
add_library(ninfer_serve STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/openai_schema.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/anthropic_schema.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/tool_call_parser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/translate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/generation_service.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/request_log.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/http_server.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serve/serve_options.cpp)
target_include_directories(ninfer_serve PUBLIC
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer_serve PUBLIC ninfer_core Threads::Threads)

add_executable(ninfer-serve ${CMAKE_CURRENT_SOURCE_DIR}/serve/main.cpp)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve)
