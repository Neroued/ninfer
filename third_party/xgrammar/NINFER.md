Vendored XGrammar v0.2.7, commit 82505d0d987c36a4209fb3d8571cf6b0f28b5acd.
Source: https://github.com/mlc-ai/xgrammar
Apache-2.0; see LICENSE. Native C++ sources only; no Python/TVM dependency.
DLPack commit bbd2f4d32427e548797929af08cfe2a9cbb3cf12, see its LICENSE.
PicoJSON and its license are included from the pinned XGrammar tree.
NInfer supplies a target-scoped CMake build without downloads. Local correctness patch in cpp/json_schema_converter.cc: bounded JSON strings exclude all
U+0000..U+001F control characters, matching the unbounded JSON string rule.
Regression: tests/text/test_structured_output.cpp. The additional-property exclusion trie uses Unicode codepoints and excludes escaped aliases of
declared properties before divergence, preventing duplicate-key overwrites from bypassing a
property schema. This restricts some otherwise valid key spellings.
Other upstream sources are unmodified.
