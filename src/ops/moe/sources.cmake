target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/weights.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/block_scaled/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/block_scaled/shared.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/groupwise_int/grouped_swiglu.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/groupwise_int/grouping.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/routing/route.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/groupwise_int/token_swiglu.cu"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_route/bf16/shapes/h2048_e256_k8.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_route/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/q4_q5/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/q4_q6/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/q8_q8/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/q4_q5/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/q4_q6/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/q8_q8/shapes/h2048_i512_e256_k8/dispatch.cpp"
)
include("${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cpu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/block_scaled/nvfp4/${NINFER_TARGET}/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/nvfp4/${NINFER_TARGET}/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/nvfp4/${NINFER_TARGET}/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/block_scaled/fp8/${NINFER_TARGET}/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/fp8/${NINFER_TARGET}/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/fp8/${NINFER_TARGET}/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/topk_softmax_route/bf16/${NINFER_TARGET}/sources.cmake")
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/swiglu/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_sigmoid_route/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_sigmoid_swiglu_moe/entry.cpp")
if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/swiglu/dgx_spark/nvfp4.cu"
    "${CMAKE_CURRENT_LIST_DIR}/swiglu/dgx_spark/nvfp4_gate_up.cu"
    "${CMAKE_CURRENT_LIST_DIR}/swiglu/dgx_spark/bf16.cu"
    "${CMAKE_CURRENT_LIST_DIR}/swiglu/dgx_spark/bf16_gate_up.cu"
    "${CMAKE_CURRENT_LIST_DIR}/topk_sigmoid_route/bf16/dgx_spark/shapes/h4096_e288_k8.cu"
    "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/nvfp4/dgx_spark/shapes/h4096_i2048_e288_k8/dispatch.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/fp8/dgx_spark/shapes/h4096_i2048_e288_k8/dispatch.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/block_scaled/h4096_i2048_e288_k8/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/block_scaled/h4096_i2048_e288_k8/fp8/dgx_spark/launch.cu")
  include("${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/block_scaled/h4096_i2048_e288_k8/nvfp4/dgx_spark/sources.cmake")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/swiglu/dgx_spark/q8.cu"
    "${CMAKE_CURRENT_LIST_DIR}/swiglu/dgx_spark/nvfp4_tma.cu")
endif()
