ninfer_add_op_test(ninfer_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_weights.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/fixture.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_complete.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_route.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_routed.cpp"
  LIBRARIES ninfer_ops)

if(NINFER_TARGET STREQUAL "rtx5090")
  ninfer_add_op_test(ninfer_moe_mixed_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mixed.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/fixture.cpp"
    LIBRARIES ninfer_ops ninfer_cpu_executor)

  ninfer_add_op_test(ninfer_moe_cpu_groupwise_int_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/cpu/test_groupwise_int.cpp"
    LIBRARIES ninfer_moe_cpu ninfer_cpu_executor)
  set_tests_properties(ninfer_moe_cpu_groupwise_int_test PROPERTIES
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" RESOURCE_LOCK cpu_team)

  ninfer_add_op_test(ninfer_moe_cpu_nvfp4_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/cpu/test_nvfp4.cpp"
    LIBRARIES ninfer_moe_cpu ninfer_cpu_executor)
  set_tests_properties(ninfer_moe_cpu_nvfp4_test PROPERTIES
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" RESOURCE_LOCK cpu_team)

  # Qualify every compiled arithmetic route through the public CPU entry. Restricted
  # dispatch belongs only to these test archives, never to the production interface.
  get_target_property(moe_cpu_sources ninfer_moe_cpu SOURCES)
  foreach(isa IN ITEMS avx2 fma)
    add_library(ninfer_moe_cpu_${isa}_qualification STATIC ${moe_cpu_sources})
    ninfer_internal_includes(ninfer_moe_cpu_${isa}_qualification)
    string(TOUPPER ${isa} isa_upper)
    target_compile_definitions(ninfer_moe_cpu_${isa}_qualification PRIVATE
      NINFER_MOE_CPU_${isa_upper}_ONLY)
    target_link_libraries(ninfer_moe_cpu_${isa}_qualification
      PUBLIC ninfer_core
      PRIVATE ninfer_cpu_executor ninfer_moe_cpu_avx2 ninfer_moe_cpu_avx512 ninfer_moe_cpu_avx512_bf16)
    ninfer_add_op_test(ninfer_moe_cpu_groupwise_int_${isa}_test
      SOURCES "${CMAKE_CURRENT_LIST_DIR}/cpu/test_groupwise_int.cpp"
      LIBRARIES ninfer_moe_cpu_${isa}_qualification ninfer_cpu_executor)
    set_tests_properties(ninfer_moe_cpu_groupwise_int_${isa}_test PROPERTIES
      ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" RESOURCE_LOCK cpu_team)
    ninfer_add_op_test(ninfer_moe_cpu_nvfp4_${isa}_test
      SOURCES "${CMAKE_CURRENT_LIST_DIR}/cpu/test_nvfp4.cpp"
      LIBRARIES ninfer_moe_cpu_${isa}_qualification ninfer_cpu_executor)
    set_tests_properties(ninfer_moe_cpu_nvfp4_${isa}_test PROPERTIES
      ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" RESOURCE_LOCK cpu_team)
  endforeach()
endif()

ninfer_add_op_test(ninfer_moe_block_scaled_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_block_scaled.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_moe_ep_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ep.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/ep_fixture.cpp"
  LIBRARIES ninfer_ops)
if(NINFER_TARGET STREQUAL "dgx_spark")
  ninfer_add_op_test(ninfer_swiglu_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_swiglu.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/test_swiglu_bf16.cpp"
    LIBRARIES ninfer_ops ninfer_linear_test_support)
endif()
if(NINFER_TARGET STREQUAL "dgx_spark")
  ninfer_add_op_test(ninfer_sigmoid_route_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sigmoid_route.cpp"
    LIBRARIES ninfer_ops ninfer_linear_test_support)
endif()
