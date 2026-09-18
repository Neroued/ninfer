add_library(ninfer_text STATIC
  unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c
)
ninfer_internal_includes(ninfer_text)
target_include_directories(ninfer_text PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
# utf8proc is compiled into this static library; without UTF8PROC_STATIC its
# header marks every symbol __declspec(dllimport) under MSVC, which breaks the
# build. PUBLIC so consumers that include the header (llama-jinja) get it too.
# GCC-only visibility path (upstream) needs no macro.
target_compile_definitions(ninfer_text PUBLIC
  $<$<PLATFORM_ID:Windows>:UTF8PROC_STATIC>)

# Language evaluation reuses the engine's Unicode and JSON dependencies.
target_sources(ninfer_jinja PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jinja.cpp)
ninfer_internal_includes(ninfer_jinja)
target_include_directories(ninfer_jinja PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer_jinja PRIVATE ninfer_text ninfer::json)
