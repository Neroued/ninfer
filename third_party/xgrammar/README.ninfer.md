# NInfer XGrammar source base

This implementation originates from [XGrammar](https://github.com/mlc-ai/xgrammar),
commit [`3641b5c21e70a6c5515ad9eca24fcbee21ee6ad0`](https://github.com/mlc-ai/xgrammar/commit/3641b5c21e70a6c5515ad9eca24fcbee21ee6ad0),
under the Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

NInfer owns the CMake target and maintains its adaptations here, adopting upstream changes
selectively. Cold compilation is limited to two concurrent callers per compiler; the adapter
uses one caller thread per compile, without creating a thread pool. Python/TVM bindings, GPU kernels,
Web code, upstream tests and build machinery are excluded.

Grammar string escaping preserves embedded NUL bytes, including in compilation cache keys.
The prepared-grammar cache shares one budget and single-flight mechanism across GBNF and JSON;
its miss factory covers validation, conversion and output framing within the cold-compile limit.
Cache access reports hit, build or an in-flight wait to the caller without adding another cache.
JSON adaptations preserve literal data in cache keys, reject unsupported unions, resolve local
JSON Pointers and carry typed errors with schema locations. String grammars constrain decoded
Unicode values before JSON encoding; property-name exclusion also covers escaped spellings.
String intersections combine patterns, Unicode length and tool-delimiter exclusion in one automaton.
General schema conjunction reduction and source diagnostics belong to NInfer's text adapter.
Positional arrays preserve optional prefix positions and impossible-position length caps.
Bounded numbers use decimal interval compilation and the product JSON serializer's binary64
round-trip boundaries, with a separate exact int64 path. The numeric adapter shares the
repository's nlohmann JSON dependency for this contract.
Regex and schema patterns share character/escape normalization with explicit full/search matching;
regex hexadecimal escapes consume exactly two digits. Product constraints reject unsupported
escapes and anchor positions before grammar conversion.
The Qwen converter uses canonical tool framing, preserves raw-string assertions with delimiter
exclusion, and emits argument numbers that the protocol JSON adapters can represent.

The CPU dependency closure includes Lark and all format converters referenced by the shared
factories. `cpp/testing.cc` also supplies token-formatting diagnostics used by the matcher and
compiled grammar; it is part of this library, not a test executable.

Bundled header dependencies:

- `3rdparty/picojson/picojson.h`: XGrammar's modified version, including ordered-object support;
  its license is in the header.
- `3rdparty/dlpack/include/dlpack/dlpack.h`: from the upstream submodule at
  [`bbd2f4d32427e548797929af08cfe2a9cbb3cf12`](https://github.com/dmlc/dlpack/commit/bbd2f4d32427e548797929af08cfe2a9cbb3cf12);
  see its [LICENSE](3rdparty/dlpack/LICENSE).

`ninfer_xgrammar` is a C++20 static library linked explicitly by consumers. It uses standard
threads and has no Python, TVM or CUDA runtime dependency. Cpptrace is disabled.
