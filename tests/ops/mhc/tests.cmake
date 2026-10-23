if(NINFER_TARGET STREQUAL "dgx_spark")
  ninfer_add_op_test(ninfer_mhc_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/fixture.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/oracle.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/cases.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/validation.cpp"
    LIBRARIES ninfer_ops)
endif()
