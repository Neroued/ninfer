add_library(ninfer_text STATIC
  unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c
)
ninfer_internal_includes(ninfer_text)
target_include_directories(ninfer_text PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
if(WIN32)
  # utf8proc.h defaults to __declspec(dllimport); force static linkage.
  target_compile_definitions(ninfer_text PRIVATE UTF8PROC_STATIC)
endif()

# Language evaluation reuses the engine's Unicode and JSON dependencies.
target_sources(ninfer_jinja PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jinja.cpp)
ninfer_internal_includes(ninfer_jinja)
target_include_directories(ninfer_jinja PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer_jinja PRIVATE ninfer_text ninfer::json)

add_library(ninfer_grammar STATIC grammar.cpp json_schema.cpp schema_composition.cpp json_input.cpp)
ninfer_internal_includes(ninfer_grammar)
target_include_directories(ninfer_grammar PRIVATE ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)
target_include_directories(ninfer_grammar SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson)
target_link_libraries(ninfer_grammar PRIVATE ninfer_xgrammar ninfer_text ninfer::json)
