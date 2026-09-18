add_library(ninfer_media_acquire STATIC
  media_acquire/acquire.cpp
)
ninfer_internal_includes(ninfer_media_acquire)
target_link_libraries(ninfer_media_acquire PRIVATE PkgConfig::LIBCURL)
if(WIN32)
  # Winsock (htons/getaddrinfo/inet_ntop/ntohl in acquire.cpp).
  target_link_libraries(ninfer_media_acquire PRIVATE ws2_32)
endif()

add_library(ninfer_product_prompt_input STATIC
  prompt_input/prompt_input.cpp
)
ninfer_internal_includes(ninfer_product_prompt_input)
target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire ninfer::json)

add_library(ninfer_product_logging STATIC
  logging/logging.cpp
  logging/pretty_format.cpp
  logging/startup_log.cpp
)
ninfer_internal_includes(ninfer_product_logging)
target_link_libraries(ninfer_product_logging PUBLIC spdlog::spdlog)
