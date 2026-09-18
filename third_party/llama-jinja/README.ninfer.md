# NInfer Jinja source base

This maintained implementation originates from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`7609846557c50f9d984719a9e1e8c5f3d02f807b`](https://github.com/ggml-org/llama.cpp/commit/7609846557c50f9d984719a9e1e8c5f3d02f807b),
`common/jinja/`, under the MIT license; see [LICENSE](LICENSE).

NInfer maintains this fork for chat-template rendering and adopts upstream fixes selectively.

Unicode case data is generated from Unicode 17.0.0 `SpecialCasing.txt` and
`DerivedCoreProperties.txt`, under [UNICODE-LICENSE](UNICODE-LICENSE):

```bash
python3 tools/generate_jinja_unicode.py /path/to/unicode-17.0.0
```
