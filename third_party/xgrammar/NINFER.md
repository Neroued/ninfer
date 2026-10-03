Vendored XGrammar v0.2.7, commit 82505d0d987c36a4209fb3d8571cf6b0f28b5acd.
Source: https://github.com/mlc-ai/xgrammar
Apache-2.0; see LICENSE. Native C++ sources only; no Python/TVM dependency.
DLPack commit bbd2f4d32427e548797929af08cfe2a9cbb3cf12, see its LICENSE.
PicoJSON and its license are included from the pinned XGrammar tree.
NInfer supplies a target-scoped CMake build without downloads. Local correctness patch in cpp/json_schema_converter.cc: bounded and unbounded JSON
strings share a Unicode scalar FSM; raw control bytes are excluded while equivalent
legal JSON escapes remain available.
Regression: tests/text/test_structured_output.cpp. The additional-property exclusion trie uses Unicode codepoints and excludes escaped aliases of
declared properties before divergence, preventing duplicate-key overwrites from bypassing a
property schema. This restricts some otherwise valid key spellings.
The local NInfer changes below also modify the FSM string path.

Numeric-range correctness patch: GenerateNumber rejects the empty-range regex sentinel instead of
interpreting it as an empty JSON number. This matters when the six-decimal generation grid contains
no value inside an otherwise nonempty real interval. Covered by the structured-output grammar tests.

NInfer supplements: JSON-string regex compilation never falls back to plain EBNF; pattern
search semantics are preserved with JSON-escaped prefix/suffix matching. Temporal regexes
validate Gregorian month lengths, leap years and RFC 3339 offsets. Native entry validation
rejects unsupported conjunctions instead of dropping constraints. Regression coverage is
in tests/text/test_structured_output.cpp.

NInfer scalar and escape supplement: Unicode FSM transitions accept only legal scalar
values and canonical UTF-8. JSON-string regexes match decoded logical characters through
raw UTF-8, short JSON escapes and Unicode escape spellings. ECMA262 dot and whitespace
sets are normalized by the schema converter; unsupported regex assertions fail closed.
CPU coverage includes test_unicode_scalar_output.cpp and the native_schema_probe.

Schema compatibility supplement: the text compiler distributes proven
anyOf type/required intersections before XGrammar compilation. String-array
uniqueness is enforced by request-owned incremental semantic masks, not by dropping
the assertion from public JSON or by rewriting completed arrays. Object upper
bounds use the existing vendored object generator. CPU equivalence, prefix
reachability and committed/speculative-state regressions guard these contracts.
