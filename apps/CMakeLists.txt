add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_load_progress
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_load_progress)

# On Windows the FFmpeg/libcurl runtime DLLs (from the vcpkg tree or an explicit
# prefix) are not on the default search path, so copy them next to each
# executable to make the build-tree binaries run in place.
if(WIN32)
  set(_ninfer_runtime_dll_dirs "")
  foreach(_prefix ${NINFER_FFMPEG_PREFIX} ${NINFER_CURL_PREFIX})
    if(_prefix AND EXISTS "${_prefix}/bin")
      list(APPEND _ninfer_runtime_dll_dirs "${_prefix}/bin")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _ninfer_runtime_dll_dirs)
  foreach(_app ninfer ninfer-serve)
    foreach(_dll_dir IN LISTS _ninfer_runtime_dll_dirs)
      add_custom_command(TARGET ${_app} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${_dll_dir}"
                "$<TARGET_FILE_DIR:${_app}>"
        COMMENT "Copying ${_app} runtime dependencies")
    endforeach()
  endforeach()
endif()
