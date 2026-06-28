# --- Core library: L0 infra + L1 kernels + L2 model + runtime ----------------
file(GLOB_RECURSE QUS_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp     ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp    ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/*.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)

add_library(qus_core STATIC ${QUS_SOURCES})
target_include_directories(qus_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
# Internal L1 layer headers resolve as "kernels/launcher/<op>.h" (wrapper -> launcher seam).
target_include_directories(qus_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
target_link_libraries(qus_core PUBLIC CUDA::cudart)
target_compile_options(qus_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(qus_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- Driver executable (token-ids in -> token-ids out) -----------------------
add_executable(qus ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(qus PRIVATE qus_core)
