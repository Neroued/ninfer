cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# CUDA 13's CCCL headers require MSVC's standard-conforming preprocessor.
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  # windows.h defines max/min as macros that break std::max/std::min.
  add_compile_definitions(NOMINMAX)
endif()

# Media (vision) decode and acquisition need FFMPEG and libcurl. Those are not
# part of the default Windows toolchain, so the text-only build
# (NINFER_BUILD_MEDIA=OFF) compiles API-compatible stubs instead and rejects
# vision requests at runtime.
if(WIN32)
  set(NINFER_MEDIA_DEFAULT OFF)
else()
  set(NINFER_MEDIA_DEFAULT ON)
endif()
option(NINFER_BUILD_MEDIA "Build FFMPEG/libcurl media decode and acquisition"
       ${NINFER_MEDIA_DEFAULT})

# ninfer_serve and prompt_input link ninfer_media_acquire unconditionally, so
# the target must exist whenever apps/tests build; NINFER_BUILD_MEDIA only
# selects the real (libcurl) vs stub implementation.
set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
if(NINFER_BUILD_MEDIA)
  if(WIN32)
    # Windows: use find_path/find_library for FFMPEG + libcurl (no pkg-config)
    set(FFMPEG_ROOT "${PROJECT_SOURCE_DIR}/../third_party/ffmpeg/ffmpeg-master-latest-win64-gpl-shared")
    set(CURL_ROOT "${PROJECT_SOURCE_DIR}/../third_party/curl-inst")

    find_path(FFMPEG_INCLUDE_DIR NAMES libavformat/avformat.h PATHS "${FFMPEG_ROOT}/include" NO_DEFAULT_PATH)
    find_library(AVFORMAT_LIBRARY NAMES avformat.lib PATHS "${FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
    find_library(AVCODEC_LIBRARY NAMES avcodec.lib PATHS "${FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
    find_library(AVUTIL_LIBRARY NAMES avutil.lib PATHS "${FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
    find_library(SWSCALE_LIBRARY NAMES swscale.lib PATHS "${FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)

    if(NOT FFMPEG_INCLUDE_DIR OR NOT AVFORMAT_LIBRARY OR NOT AVCODEC_LIBRARY OR NOT AVUTIL_LIBRARY OR NOT SWSCALE_LIBRARY)
      message(FATAL_ERROR "FFMPEG libraries not found. Install BtbN ffmpeg-win64-gpl-shared into third_party/ffmpeg/")
    endif()

    add_library(PkgConfig::FFMPEG INTERFACE IMPORTED)
    set_target_properties(PkgConfig::FFMPEG PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${AVFORMAT_LIBRARY};${AVCODEC_LIBRARY};${AVUTIL_LIBRARY};${SWSCALE_LIBRARY}"
    )

    if(NINFER_BUILD_MEDIA_ACQUIRE)
      find_path(CURL_INCLUDE_DIR NAMES curl/curl.h PATHS "${CURL_ROOT}/include" NO_DEFAULT_PATH)
      find_library(CURL_LIBRARY NAMES libcurl_imp.lib PATHS "${CURL_ROOT}/lib" NO_DEFAULT_PATH)

      if(NOT CURL_INCLUDE_DIR OR NOT CURL_LIBRARY)
        message(FATAL_ERROR "libcurl not found. Build curl with MSVC into third_party/curl-inst/")
      endif()

      add_library(PkgConfig::LIBCURL INTERFACE IMPORTED)
      set_target_properties(PkgConfig::LIBCURL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CURL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${CURL_LIBRARY}"
      )
    endif()
  else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
      libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
    if(NINFER_BUILD_MEDIA_ACQUIRE)
      pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    endif()
  endif()
endif()
find_package(Threads REQUIRED)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
