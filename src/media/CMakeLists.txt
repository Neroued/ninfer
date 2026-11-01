add_library(ninfer_media_decode STATIC
  decode/decode.cpp
  decode/jpeg.cpp
)
ninfer_internal_includes(ninfer_media_decode)
target_link_libraries(ninfer_media_decode PRIVATE PkgConfig::FFMPEG JPEG::JPEG)
