target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q4/q4_linear_swiglu_gemm_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/q4_linear_swiglu_gemv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/q4_linear_swiglu_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_linear_swiglu_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_linear_swiglu_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_linear_swiglu_a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_linear_swiglu_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_linear_swiglu_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_linear_swiglu_a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_linear_swiglu_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_linear_swiglu_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_linear_swiglu_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_dflash2_linear_swiglu.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_linear_swiglu_gemm_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_linear_swiglu_gemm_splitk.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_linear_swiglu_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/linear_swiglu.cpp"
)

target_sources(ninfer_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_linear_swiglu_a4_tma.cu"
)
