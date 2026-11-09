add_library(ninfer_xgrammar STATIC
  cpp/compiled_grammar.cc
  cpp/config.cc
  cpp/converter_ext/cohere.cc
  cpp/converter_ext/deepseek.cc
  cpp/converter_ext/gemma.cc
  cpp/converter_ext/glm.cc
  cpp/converter_ext/kimi_k3.cc
  cpp/converter_ext/minimax.cc
  cpp/converter_ext/minimax_m3.cc
  cpp/converter_ext/qwen.cc
  cpp/earley_parser.cc
  cpp/fsm.cc
  cpp/fsm_builder.cc
  cpp/grammar.cc
  cpp/grammar_builder.cc
  cpp/grammar_compiler.cc
  cpp/grammar_functor.cc
  cpp/grammar_matcher.cc
  cpp/grammar_parser.cc
  cpp/grammar_printer.cc
  cpp/json_schema_converter.cc
  cpp/json_number.cc
  cpp/json_string_grammar.cc
  cpp/lark_converter.cc
  cpp/regex_converter.cc
  cpp/structural_tag.cc
  cpp/suffix_automata.cc
  cpp/support/logging.cc
  cpp/support/recursion_guard.cc
  cpp/testing.cc
  cpp/tokenizer_info.cc
)
target_compile_features(ninfer_xgrammar PUBLIC cxx_std_20)
target_include_directories(ninfer_xgrammar
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(ninfer_xgrammar SYSTEM
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/picojson)
target_compile_definitions(ninfer_xgrammar PRIVATE
  XGRAMMAR_ENABLE_CPPTRACE=0
  XGRAMMAR_ENABLE_INTERNAL_CHECK=0)
target_link_libraries(ninfer_xgrammar PRIVATE Threads::Threads ninfer::json)
