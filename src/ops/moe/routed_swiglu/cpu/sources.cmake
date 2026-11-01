if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/unavailable.cpp")
  return()
endif()

# Baseline entry/grouping code never inherits worker ISA flags.
set(ninfer_moe_cpu_sources
  "${CMAKE_CURRENT_LIST_DIR}/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q6/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/h2560_i640_e512_k10/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../../kernels/cpu/groupwise_int/launch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../../kernels/cpu/nvfp4/launch.cpp")

foreach(isa IN ITEMS avx2 avx512 avx512_bf16)
  add_library(ninfer_moe_cpu_${isa} OBJECT
    "${CMAKE_CURRENT_LIST_DIR}/../../kernels/cpu/groupwise_int/${isa}.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../../kernels/cpu/nvfp4/${isa}.cpp")
  ninfer_internal_includes(ninfer_moe_cpu_${isa})
  set_target_properties(ninfer_moe_cpu_${isa} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(ninfer_moe_cpu_${isa} PRIVATE ninfer_core)
  target_compile_options(ninfer_moe_cpu_${isa} PRIVATE -mavx2 -mfma -mf16c)
  if(NOT isa STREQUAL "avx2")
    target_compile_options(ninfer_moe_cpu_${isa} PRIVATE
      -mavx512f -mavx512bw -mavx512dq -mavx512vl)
  endif()
  if(isa STREQUAL "avx512_bf16")
    target_compile_options(ninfer_moe_cpu_${isa} PRIVATE -mavx512bf16)
  endif()
endforeach()

add_library(ninfer_moe_cpu SHARED ${ninfer_moe_cpu_sources})
ninfer_internal_includes(ninfer_moe_cpu)
target_link_libraries(ninfer_moe_cpu PUBLIC ninfer_core PRIVATE ninfer_cpu_executor
  ninfer_moe_cpu_avx2 ninfer_moe_cpu_avx512 ninfer_moe_cpu_avx512_bf16)
target_link_libraries(ninfer_ops PUBLIC ninfer_moe_cpu)
