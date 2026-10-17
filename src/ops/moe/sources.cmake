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
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_route/bf16/shapes/h2560_e512_k10.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_route/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/fp8/shapes/h2560_i640_e512_k10/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/q4_q5/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/q4_q6/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/q8_q8/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/fp8/shapes/h2560_i640_e512_k10/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/q4_q5/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/q4_q6/shapes/h2048_i512_e256_k8/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/q8_q8/shapes/h2048_i512_e256_k8/dispatch.cpp"
)
include("${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cpu/sources.cmake")

target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/block_scaled/fp8/launch.cu"
)

include("${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/block_scaled/nvfp4/${NINFER_TARGET}/sources.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/nvfp4/${NINFER_TARGET}/sources.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/nvfp4/${NINFER_TARGET}/sources.cmake")
