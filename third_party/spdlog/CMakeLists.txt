add_library(ninfer_third_party_spdlog STATIC
  src/async.cpp
  src/bundled_fmtlib_format.cpp
  src/cfg.cpp
  src/color_sinks.cpp
  src/file_sinks.cpp
  src/spdlog.cpp
  src/stdout_sinks.cpp)
add_library(spdlog::spdlog ALIAS ninfer_third_party_spdlog)

target_compile_features(ninfer_third_party_spdlog PUBLIC cxx_std_11)
target_compile_definitions(ninfer_third_party_spdlog PUBLIC SPDLOG_COMPILED_LIB)
target_include_directories(ninfer_third_party_spdlog SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ninfer_third_party_spdlog PUBLIC Threads::Threads)
