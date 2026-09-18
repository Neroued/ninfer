add_library(ninfer_jinja STATIC
  jinja/json.cpp
  jinja/lexer.cpp
  jinja/parser.cpp
  jinja/runtime.cpp
  jinja/string.cpp
  jinja/value.cpp
  jinja/unicode.cpp
)
target_include_directories(ninfer_jinja
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
