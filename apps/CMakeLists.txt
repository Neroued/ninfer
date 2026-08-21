add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_load_progress
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
if(DEFINED ENV{NINFER_BUILD_ID})
  set(NINFER_BUILD_ID "$ENV{NINFER_BUILD_ID}")
else()
  execute_process(
    COMMAND git describe --always --dirty --tags
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NINFER_BUILD_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()
if(NOT NINFER_BUILD_ID)
  set(NINFER_BUILD_ID "unknown")
endif()
target_compile_definitions(ninfer-serve PRIVATE NINFER_BUILD_ID="${NINFER_BUILD_ID}")
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_load_progress)
