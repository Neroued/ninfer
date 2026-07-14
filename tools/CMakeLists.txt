# Target-private diagnostics are explicit build products, not public Engine API.
add_executable(ninfer-qwen3_6_27b-dump qwen3_6_27b_dump/main.cpp)
target_include_directories(ninfer-qwen3_6_27b-dump PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b_rtx5090/impl)
target_link_libraries(ninfer-qwen3_6_27b-dump PRIVATE
  ninfer_qwen3_6_27b_rtx5090
  ninfer_product_prompt_input
  ninfer_artifact
  ninfer_core
  CUDA::cudart)
