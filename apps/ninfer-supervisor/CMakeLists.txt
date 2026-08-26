if(NOT WIN32)
  return()
endif()

add_executable(ninfer-supervisor
  main.cpp
  engine_child.cpp
  collector.cpp
  server.cpp
  tray.cpp)
target_include_directories(ninfer-supervisor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-supervisor PRIVATE CUDA::cudart)
target_link_options(ninfer-supervisor PRIVATE
  "LINKER:/DEFAULTLIB:dxgi"
  "LINKER:/DEFAULTLIB:ole32"
  "LINKER:/DEFAULTLIB:shell32"
  "LINKER:/DEFAULTLIB:user32"
  "LINKER:/DEFAULTLIB:gdi32"
  "LINKER:/DEFAULTLIB:advapi32"
  "LINKER:/DEFAULTLIB:ws2_32"
  "LINKER:/DEFAULTLIB:crypt32")
target_compile_definitions(ninfer-supervisor PRIVATE
  WIN32_LEAN_AND_MEAN
  NOMINMAX
  _WIN32_WINNT=0x0A00)
