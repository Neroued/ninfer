# NInferMediaDeps.cmake
#
# Locates FFmpeg and libcurl and exposes them as INTERFACE imported targets
# NInfer::FFmpeg and NInfer::Curl, which the ninfer_media_* libraries link.
#
#   * POSIX   : discovered through pkg-config (unchanged behaviour).
#   * Windows : located from a vcpkg install tree or an explicit prefix, because
#               pkg-config is not part of the Windows toolchain. The prefix is
#               chosen, per package, in this order: NINFER_FFMPEG_ROOT /
#               NINFER_CURL_ROOT (per package), then NINFER_MEDIA_ROOT, then
#               $ENV{VCPKG_ROOT}/installed/<VCPKG_TARGET_TRIPLET>.
#
# On Windows the resolved prefixes are cached as NINFER_FFMPEG_PREFIX and
# NINFER_CURL_PREFIX so the app targets can copy the dependency DLLs next to the
# built executables at build time.

if(NOT WIN32)
  find_package(PkgConfig REQUIRED)
endif()

function(ninfer_media_resolve_prefix _out)
  set(result "")
  if(DEFINED NINFER_MEDIA_ROOT)
    set(result "${NINFER_MEDIA_ROOT}")
  elseif(DEFINED ENV{NINFER_MEDIA_ROOT})
    set(result "$ENV{NINFER_MEDIA_ROOT}")
  elseif(DEFINED ENV{VCPKG_ROOT})
    set(triplet "x64-windows")
    if(DEFINED ENV{VCPKG_TARGET_TRIPLET})
      set(triplet "$ENV{VCPKG_TARGET_TRIPLET}")
    endif()
    if(EXISTS "$ENV{VCPKG_ROOT}/installed/${triplet}/include")
      set(result "$ENV{VCPKG_ROOT}/installed/${triplet}")
    endif()
  endif()
  set(${_out} "${result}" PARENT_SCOPE)
endfunction()

function(ninfer_require_prefix _label _root _hint)
  if(NOT _root OR NOT EXISTS "${_root}/include")
    message(FATAL_ERROR
      "${_label} not found. Point the build at an install prefix containing "
      "include/ and lib/. ${_hint}")
  endif()
endfunction()

# Bundle an include directory and a link list (imported libs plus, on Windows,
# the system libraries the package needs) into an INTERFACE imported target.
function(ninfer_make_imported_target interface include_dir libs)
  add_library(${interface} INTERFACE IMPORTED)
  set_target_properties(${interface} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${include_dir}"
    INTERFACE_LINK_LIBRARIES "${libs}")
endfunction()

function(ninfer_find_ffmpeg)
  if(TARGET NInfer::FFmpeg)
    return()
  endif()
  if(NOT WIN32)
    pkg_check_modules(_ninfer_ffmpeg REQUIRED IMPORTED_TARGET
      libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
    ninfer_make_imported_target(NInfer::FFmpeg
      "${_ninfer_ffmpeg_INCLUDE_DIRS}" "${_ninfer_ffmpeg_LIBRARIES}")
    return()
  endif()

  if(DEFINED NINFER_FFMPEG_ROOT)
    set(root "${NINFER_FFMPEG_ROOT}")
  elseif(DEFINED ENV{NINFER_FFMPEG_ROOT})
    set(root "$ENV{NINFER_FFMPEG_ROOT}")
  else()
    ninfer_media_resolve_prefix(root)
  endif()
  ninfer_require_prefix("FFmpeg" "${root}"
    "set NINFER_FFMPEG_ROOT or NINFER_MEDIA_ROOT to an FFmpeg prefix, or install it "
    "with `vcpkg install ffmpeg --triplet x64-windows` and set VCPKG_ROOT.")

  find_path(NINFER_FFMPEG_INCLUDE_DIR libavcodec/avcodec.h
    PATHS "${root}/include" NO_DEFAULT_PATH REQUIRED)
  foreach(module IN ITEMS avformat avcodec swscale avutil)
    find_library(NINFER_FFMPEG_${module}_LIB NAMES ${module}
      PATHS "${root}/lib" NO_DEFAULT_PATH REQUIRED)
  endforeach()

  # The FFmpeg import libraries reference the Windows multimedia/network API.
  # (The list contains only import libraries that exist in the Windows SDK; e.g.
  # there is no Ntmapi.lib, and Nt*/Rtl* symbols, if ever needed, come from ntdll.)
  ninfer_make_imported_target(NInfer::FFmpeg "${NINFER_FFMPEG_INCLUDE_DIR}"
    "${NINFER_FFMPEG_avformat_LIB};${NINFER_FFMPEG_avcodec_LIB};${NINFER_FFMPEG_swscale_LIB};${NINFER_FFMPEG_avutil_LIB};Strmiids;Ws2_32;Advapi32;Ole32;Oleaut32;User32;Iphlpapi;Userenv")
  set(NINFER_FFMPEG_PREFIX "${root}" CACHE INTERNAL "Resolved FFmpeg install prefix")
endfunction()

function(ninfer_find_curl)
  if(TARGET NInfer::Curl)
    return()
  endif()
  if(NOT WIN32)
    pkg_check_modules(_ninfer_curl REQUIRED IMPORTED_TARGET libcurl>=7.85)
    ninfer_make_imported_target(NInfer::Curl
      "${_ninfer_curl_INCLUDE_DIRS}" "${_ninfer_curl_LIBRARIES}")
    return()
  endif()

  if(DEFINED NINFER_CURL_ROOT)
    set(root "${NINFER_CURL_ROOT}")
  elseif(DEFINED ENV{NINFER_CURL_ROOT})
    set(root "$ENV{NINFER_CURL_ROOT}")
  else()
    ninfer_media_resolve_prefix(root)
  endif()
  ninfer_require_prefix("libcurl" "${root}"
    "set NINFER_CURL_ROOT or NINFER_MEDIA_ROOT to a libcurl prefix, or install it "
    "with `vcpkg install curl --triplet x64-windows` and set VCPKG_ROOT.")

  find_path(NINFER_CURL_INCLUDE_DIR curl/curl.h
    PATHS "${root}/include" NO_DEFAULT_PATH REQUIRED)
  find_library(NINFER_CURL_LIB NAMES curl libcurl
    PATHS "${root}/lib" NO_DEFAULT_PATH REQUIRED)

  ninfer_make_imported_target(NInfer::Curl "${NINFER_CURL_INCLUDE_DIR}"
    "${NINFER_CURL_LIB};Ws2_32;Crypt32;Cryptui;Normaliz;Bcrypt;Advapi32;Userenv")
  set(NINFER_CURL_PREFIX "${root}" CACHE INTERNAL "Resolved libcurl install prefix")
endfunction()