cmake_minimum_required(VERSION 3.28)
project(qus LANGUAGES CXX CUDA)

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
set_target_properties(qus_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- Bench / driver executable (token-ids in -> token-ids out) ---------------
add_executable(qus ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(qus PRIVATE qus_core)

# --- Tests (placeholder; enable once tests exist) ----------------------------
# enable_testing()
# add_subdirectory(tests)
