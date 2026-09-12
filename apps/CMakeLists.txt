# Embed an activeCodePage=UTF-8 manifest so the MSVC CRT hands apps UTF-8 argv on
# Windows instead of the legacy ANSI code page (breaks non-ASCII --prompt/paths).
# The manifest is appended after each add_executable via target_sources; CMake's
# MSVC rule recognizes .manifest sources and merges them into the default manifest.
if(WIN32 AND MSVC)
  set(NINFER_APP_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/windows-utf8.manifest)
endif()

add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)
if(WIN32 AND MSVC)
  target_sources(ninfer PRIVATE ${NINFER_APP_MANIFEST})
endif()

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)
if(WIN32 AND MSVC)
  target_sources(ninfer-serve PRIVATE ${NINFER_APP_MANIFEST})
endif()

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
target_include_directories(ninfer-perplexity PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/perplexity)
target_link_libraries(ninfer-perplexity PRIVATE ninfer_engine ninfer_product_logging)
if(WIN32 AND MSVC)
  target_sources(ninfer-perplexity PRIVATE ${NINFER_APP_MANIFEST})
endif()
