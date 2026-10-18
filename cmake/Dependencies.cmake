find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
if(NINFER_ENABLE_DISTRIBUTED)
  find_path(NCCL_INCLUDE_DIR nccl.h REQUIRED)
  find_library(NCCL_LIBRARY NAMES nccl REQUIRED)
  add_library(ninfer::nccl UNKNOWN IMPORTED GLOBAL)
  set_target_properties(ninfer::nccl PROPERTIES
    IMPORTED_LOCATION "${NCCL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(HWLOC REQUIRED IMPORTED_TARGET hwloc>=2.5)
pkg_check_modules(URING REQUIRED IMPORTED_TARGET liburing>=2.5)

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

if(NINFER_BUILD_ENGINE)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
  add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)
  add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)
endif()

if(NINFER_BUILD_PRODUCT_SUPPORT)
  add_library(ninfer::toml INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::toml INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/tomlplusplus)
  target_compile_features(ninfer::toml INTERFACE cxx_std_17)

  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
