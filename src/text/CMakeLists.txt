add_library(ninfer_text STATIC
  unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c
)
ninfer_internal_includes(ninfer_text)
target_include_directories(ninfer_text PRIVATE ${PROJECT_SOURCE_DIR}/third_party)

# Language evaluation reuses the engine's Unicode and JSON dependencies.
target_sources(ninfer_jinja PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jinja.cpp)
ninfer_internal_includes(ninfer_jinja)
target_include_directories(ninfer_jinja PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer_jinja PRIVATE ninfer_text ninfer::json)

# Structured output: GBNF / JSON Schema / regex / choice compiled by XGrammar (Neroued/ninfer
# 77c2e5e2). Linked by the Qwen frontend (models/qwen3_5/frontend_sources.cmake).
add_library(ninfer_grammar STATIC grammar.cpp json_schema.cpp schema_composition.cpp schema_types.cpp json_input.cpp
  constraint_limits.cpp)
ninfer_internal_includes(ninfer_grammar)
target_include_directories(ninfer_grammar PRIVATE ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)
target_include_directories(ninfer_grammar SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson)
target_link_libraries(ninfer_grammar PRIVATE ninfer_xgrammar ninfer_text ninfer::json)
