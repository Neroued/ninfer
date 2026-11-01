foreach(form IN ITEMS snapshot record)
  ninfer_add_op_test(ninfer_gdn_input_proj_conv_${form}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/${form}_cases.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/${form}.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/weights.cpp"
    LIBRARIES ninfer_ops)
endforeach()
