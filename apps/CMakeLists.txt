add_executable(ninfer cli/main.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_config
  ninfer_product_logging
  ninfer_product_prompt_input)

if(NINFER_ENABLE_DISTRIBUTED)
  add_executable(ninfer-worker worker/main.cpp)
  ninfer_internal_includes(ninfer-worker)
  target_link_libraries(ninfer-worker PRIVATE ninfer_engine ninfer_runtime_distributed)
endif()

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/options.cpp
  perplexity/report.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_config ninfer_product_logging ninfer::json)
