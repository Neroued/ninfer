# Legacy q5090 `.qus` support

This directory contains only the Python-side pieces still needed around the current C++ Engine's
q5090 v4.2 `.qus` route:

- `reader.py` is the mmap-backed v4.2 reader and structural-dump producer;
- `codec.py` decodes the legacy Q4/Q5/Q6/W8 payloads used by parity tools;
- `diagnostics/structure.py` compares canonical q5090 structural dumps;
- `preprocess.cpp` and `vision_dump.cpp` are current-Engine parity executables.

The complete Python reference no longer lives here. Native `.ninfer` conversion and reference
inference are under:

- `tools/convert/qwen3_6_27b_rtx5090/`;
- `tools/reference/qwen3_6_27b_rtx5090/`;
- `tools/parity/qwen3_6_27b_rtx5090/`.

The `.qus` reader and converter remain only because the current C++ Engine has not yet migrated to
`.ninfer`. They are removed together with that Engine route, not used as a compatibility layer for
new targets.

To compare two current-Engine structural dumps:

```bash
python -m tools.q5090.diagnostics.structure reference.json candidate.json
```
