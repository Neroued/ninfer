add_library(ninfer_media_acquire STATIC
  media_acquire/acquire.cpp
)
ninfer_internal_includes(ninfer_media_acquire)
target_link_libraries(ninfer_media_acquire PRIVATE PkgConfig::LIBCURL)

add_library(ninfer_product_prompt_input STATIC
  prompt_input/prompt_input.cpp
)
ninfer_internal_includes(ninfer_product_prompt_input)
target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire ninfer::json)

add_library(ninfer_product_logging STATIC
  logging/logging.cpp
  logging/configuration.cpp
  logging/pretty_format.cpp
  logging/startup_log.cpp
)
ninfer_internal_includes(ninfer_product_logging)
target_link_libraries(ninfer_product_logging PUBLIC spdlog::spdlog)

add_library(ninfer_product_config STATIC
  config/options.cpp
  config/schema.cpp
  config/file.cpp
  config/configuration.cpp)
ninfer_internal_includes(ninfer_product_config)
target_link_libraries(ninfer_product_config
  PUBLIC ninfer_product_logging
  PRIVATE ninfer::toml)
