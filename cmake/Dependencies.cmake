find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

if(WIN32)
  # --- FFmpeg (Windows, vcpkg) ---------------------------------------------
  # Dependencies are declared in vcpkg.json; the vcpkg toolchain (activated via
  # -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake) provides
  # the Find module.  See README for setup instructions.
  find_package(FFMPEG REQUIRED)
  message(STATUS "FFmpeg: ${FFMPEG_VERSION}")
  if(NOT TARGET PkgConfig::FFMPEG)
    add_library(PkgConfig::FFMPEG INTERFACE IMPORTED)
    # FFMPEG_LIBRARIES may contain optimized/debug keywords, which
    # target_link_libraries expands per-configuration (set_target_properties
    # would misparse the keyword pairs as property/value arguments).
    target_include_directories(PkgConfig::FFMPEG INTERFACE ${FFMPEG_INCLUDE_DIRS})
    target_link_libraries(PkgConfig::FFMPEG INTERFACE ${FFMPEG_LIBRARIES})
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  if(WIN32)
    # --- libcurl (Windows, vcpkg) ------------------------------------------
    # vcpkg installs curl with Schannel (SSPI) TLS; see vcpkg.json.
    find_package(CURL REQUIRED)
    message(STATUS "libcurl: ${CURL_VERSION_STRING}")
    if(NOT TARGET PkgConfig::LIBCURL)
      add_library(PkgConfig::LIBCURL INTERFACE IMPORTED)
      set_target_properties(PkgConfig::LIBCURL PROPERTIES
        INTERFACE_LINK_LIBRARIES CURL::libcurl)
    endif()
  else()
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  endif()
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
