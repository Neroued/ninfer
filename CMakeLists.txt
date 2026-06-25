cmake_minimum_required(VERSION 3.28)
project(qus LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

# --- Standards / target arch -------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)   # RTX 5090 / Blackwell (sm_120)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Core library: L0 infra + L1 kernels + L2 model + runtime ----------------
file(GLOB_RECURSE QUS_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/src/core/*.cpp    ${CMAKE_SOURCE_DIR}/src/core/*.cu
  ${CMAKE_SOURCE_DIR}/src/kernels/*.cu
  ${CMAKE_SOURCE_DIR}/src/model/*.cpp
  ${CMAKE_SOURCE_DIR}/src/runtime/*.cpp)

add_library(qus_core STATIC ${QUS_SOURCES})
target_include_directories(qus_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(qus_core PUBLIC CUDA::cudart)
set_target_properties(qus_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- Bench / driver executable (token-ids in -> token-ids out) ---------------
add_executable(qus ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(qus PRIVATE qus_core)

# --- Tests -------------------------------------------------------------------
enable_testing()

add_executable(qus_l0_tests
  ${CMAKE_SOURCE_DIR}/tests/test_dtype.cpp)
target_link_libraries(qus_l0_tests PRIVATE qus_core)
target_include_directories(qus_l0_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME qus_l0_tests COMMAND qus_l0_tests)

add_executable(qus_device_test ${CMAKE_SOURCE_DIR}/tests/test_device.cpp)
target_link_libraries(qus_device_test PRIVATE qus_core)
target_include_directories(qus_device_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME qus_device_test COMMAND qus_device_test)

add_executable(qus_tensor_test ${CMAKE_SOURCE_DIR}/tests/test_tensor.cpp)
target_link_libraries(qus_tensor_test PRIVATE qus_core)
target_include_directories(qus_tensor_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME qus_tensor_test COMMAND qus_tensor_test)

add_executable(qus_arena_test ${CMAKE_SOURCE_DIR}/tests/test_arena.cpp)
target_link_libraries(qus_arena_test PRIVATE qus_core)
target_include_directories(qus_arena_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME qus_arena_test COMMAND qus_arena_test)

add_executable(qus_kv_cache_test ${CMAKE_SOURCE_DIR}/tests/test_kv_cache.cpp)
target_link_libraries(qus_kv_cache_test PRIVATE qus_core)
target_include_directories(qus_kv_cache_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME qus_kv_cache_test COMMAND qus_kv_cache_test)

add_executable(qus_state_store_test ${CMAKE_SOURCE_DIR}/tests/test_state_store.cpp)
target_link_libraries(qus_state_store_test PRIVATE qus_core)
target_include_directories(qus_state_store_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME qus_state_store_test COMMAND qus_state_store_test)

add_executable(qus_weight_store_test ${CMAKE_SOURCE_DIR}/tests/test_weight_store.cpp)
target_link_libraries(qus_weight_store_test PRIVATE qus_core)
target_include_directories(qus_weight_store_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME qus_weight_store_test COMMAND qus_weight_store_test)
